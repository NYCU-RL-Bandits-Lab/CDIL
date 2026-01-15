import os
import gym
import sys
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
import time
import random
import pickle
import math

# from QAvatar.target_domain.core.flow.real_nvp import RealNvp
import utils.utils as utils
from utils.real_nvp import RealNvp

EPS = np.finfo(np.float32).eps
EPS2 = 1e-3


class Avatar(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        is_discrete_action: bool,
        src_critic,
        src_cost,
        src_state_dim,
        src_action_dim,
        config,
    ):
        super(Avatar, self).__init__()
        hidden_size = config["hidden_size"]
        critic_lr = config["critic_lr"]
        actor_lr = config["actor_lr"]
        self.is_discrete_action = is_discrete_action
        self.grad_reg_coeffs = config["grad_reg_coeffs"]
        self.discount = config["gamma"]
        self.non_expert_regularization = config["alpha"] + 1.0
        self.flow_in_decoder = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cost = utils.Critic(
            state_dim,
            action_dim,
            hidden_size=hidden_size,
            output_activation_fn=torch.sigmoid,
        ).to(self.device)
        self.critic = utils.Critic(state_dim, 0, hidden_size=hidden_size).to(
            self.device
        )
        if config["flow_model_path"]:
            self.flow_model = (
                RealNvp.load_module(config["flow_model_path"]).to(self.device).eval()
            )
            self.action_flow_model = (
                RealNvp.load_module(config["flow_model_action_path"])
                .to(self.device)
                .eval()
            )
            self.flow_in_decoder = True
        self.decoder = utils.decoder_network(
            src_state_dim - 1, state_dim, hidden_size, self.device
        ).to(self.device)
        self.action_decoder = utils.action_decoder_network(
            src_action_dim, state_dim + action_dim, hidden_size, self.device
        ).to(self.device)

        self.src_critic = src_critic.eval()
        self.src_cost = src_cost.eval()

        if self.is_discrete_action:
            self.actor = utils.DiscreteActor(state_dim, action_dim).to(self.device)
        else:
            self.actor = utils.TanhActor(
                state_dim, action_dim, hidden_size=hidden_size
            ).to(self.device)

        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=critic_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=critic_lr)
        self.action_decoder_optimizer = optim.Adam(
            self.action_decoder.parameters(), lr=critic_lr
        )

        self.c1_1 = 0.0
        self.c2_smooth_1 = 0.0
        self.c1_2 = 0.0
        self.c2_smooth_2 = 0.0
        self.l2 = 0.0

    def update(
        self,
        init_states,
        expert_states,
        expert_actions,
        expert_next_states,
        union_states,
        union_actions,
        union_next_states,
        union_init_indices,
        union_indices_half,
        union_indices,
        timestep,
        init_len,
        state_len,
    ):  # ,source_shift=None, source_scale=None):
        self.cost_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.action_decoder_optimizer.zero_grad()

        all_union_states = union_states
        all_union_actions = union_actions
        all_union_next_states = union_next_states

        init_states_1 = init_states[union_init_indices]
        union_states_1 = union_states[union_indices_half]
        union_actions_1 = union_actions[union_indices_half]
        union_next_states_1 = union_next_states[union_indices_half]

        init_states_2 = init_states[init_len + union_init_indices]
        union_states_2 = union_states[state_len + union_indices_half]
        union_actions_2 = union_actions[state_len + union_indices_half]
        union_next_states_2 = union_next_states[state_len + union_indices_half]

        union_states = torch.cat(
            [
                union_states[union_indices_half],
                union_states[state_len + union_indices_half],
            ],
            dim=0,
        )
        union_actions = torch.cat(
            [
                union_actions[union_indices_half],
                union_actions[state_len + union_indices_half],
            ],
            dim=0,
        )
        union_next_states = torch.cat(
            [
                union_next_states[union_indices_half],
                union_next_states[state_len + union_indices_half],
            ],
            dim=0,
        )
        # union_actions = union_actions[union_indices]
        # union_next_states = union_next_states[union_indices]

        expert_inputs = torch.cat([expert_states, expert_actions], -1)
        union_inputs = torch.cat([union_states, union_actions], -1)
        union_inputs_1 = torch.cat([union_states_1, union_actions_1], -1)
        union_inputs_2 = torch.cat([union_states_2, union_actions_2], -1)
        state_mapping_output_1 = self.decoder(union_states_1).double()
        state_mapping_output_2 = self.decoder(union_states_2).double()
        if self.flow_in_decoder:
            state_mapping_output_1 = self.flow_model.g(
                self.decoder(union_states_1).double()
            )[0].float()
            state_mapping_output_2 = self.flow_model.g(
                self.decoder(union_states_2).double()
            )[0].float()
        state_mapping_output_1 = torch.cat(
            [
                state_mapping_output_1,
                torch.zeros([state_mapping_output_1.shape[0], 1]).to(self.device),
            ],
            -1,
        )
        action_mapping_output_1 = self.action_decoder(union_inputs_1)
        state_mapping_output_2 = torch.cat(
            [
                state_mapping_output_2,
                torch.zeros([state_mapping_output_2.shape[0], 1]).to(self.device),
            ],
            -1,
        )
        action_mapping_output_2 = self.action_decoder(union_inputs_2)
        if self.flow_in_decoder:
            action_mapping_output_1 = self.action_flow_model.g(
                self.action_decoder(union_inputs_1).double()
            )[0].float()
            action_mapping_output_2 = self.action_flow_model.g(
                self.action_decoder(union_inputs_2).double()
            )[0].float()
        src_union_inputs_1 = torch.cat(
            [state_mapping_output_1, action_mapping_output_1], -1
        ).float()
        src_union_inputs_2 = torch.cat(
            [state_mapping_output_2, action_mapping_output_2], -1
        ).float()

        expert_cost_val = self.cost(expert_inputs)
        union_cost_val = self.cost(union_inputs)
        union_cost_val_1 = self.cost(union_inputs_1)
        union_cost_val_2 = self.cost(union_inputs_2)

        unif_rand = torch.rand(expert_states.shape[0], 1).to(self.device)
        mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * union_inputs
        mixed_inputs2 = (
            unif_rand * union_inputs[torch.randperm(union_inputs.size(0))]
            + (1 - unif_rand) * union_inputs
        )
        mixed_inputs = torch.cat([mixed_inputs1, mixed_inputs2], 0)

        # Gradient penalty for cost
        mixed_inputs.requires_grad_()
        cost_output = self.cost(mixed_inputs)
        cost_output = torch.log(1 / (cost_output + EPS2) - 1 + EPS2)
        cost_mixed_grad = (
            torch.autograd.grad(
                outputs=cost_output,
                inputs=mixed_inputs,
                grad_outputs=torch.ones_like(cost_output),
                create_graph=True,
                retain_graph=True,
            )[0]
            + EPS
        )
        cost_grad_penalty = torch.mean((cost_mixed_grad.norm(2, dim=-1) - 1) ** 2)
        cost_loss = (
            nn.BCEWithLogitsLoss()(expert_cost_val, torch.ones_like(expert_cost_val))
            + nn.BCEWithLogitsLoss()(union_cost_val, torch.zeros_like(union_cost_val))
            + self.grad_reg_coeffs[0] * cost_grad_penalty
        )

        union_cost_1 = torch.log(1 / (union_cost_val_1 + EPS2) - 1 + EPS2)
        union_cost_2 = torch.log(1 / (union_cost_val_2 + EPS2) - 1 + EPS2)

        # nu learning
        init_nu_1 = self.critic(init_states_1)
        union_nu_1 = self.critic(union_states_1)
        union_next_nu_1 = self.critic(union_next_states_1)
        union_adv_nu_1 = (
            -union_cost_1.detach() + self.discount * union_next_nu_1 - union_nu_1
        )

        init_nu_2 = self.critic(init_states_2)
        union_nu_2 = self.critic(union_states_2)
        union_next_nu_2 = self.critic(union_next_states_2)
        union_adv_nu_2 = (
            -union_cost_2.detach() + self.discount * union_next_nu_2 - union_nu_2
        )

        # non_linear_loss = self.non_expert_regularization * torch.logsumexp(
        #     (union_adv_nu_1 / self.non_expert_regularization + union_adv_nu_2 / self.non_expert_regularization)/2.0, dim=0)
        union_adv_nu_full = torch.cat([union_adv_nu_1, union_adv_nu_2], dim=0)
        non_linear_loss = self.non_expert_regularization * torch.logsumexp(
            union_adv_nu_full / self.non_expert_regularization, dim=0
        )
        linear_loss = (1 - self.discount) * (init_nu_1 + init_nu_2).mean() / 2.0
        nu_loss = non_linear_loss + linear_loss

        # regularization of nu loss
        lambda_ = 0.001
        # nu_loss = nu_loss + lambda_*(init_nu_1**2).mean()/2.0
        init_nu_full = torch.cat([init_nu_1, init_nu_2], dim=0)
        nu_loss = nu_loss + lambda_ * (init_nu_full**2).mean()

        # mapping function learning
        next_actions_1 = self.actor(union_next_states_1)[0]
        next_union_inputs_1 = torch.cat([union_next_states_1, next_actions_1], -1)
        src_qvalue_1 = self.src_critic(src_union_inputs_1)
        next_state_output_1 = self.decoder(union_next_states_1)
        next_actions_2 = self.actor(union_next_states_2)[0]
        next_union_inputs_2 = torch.cat([union_next_states_2, next_actions_2], -1)
        src_qvalue_2 = self.src_critic(src_union_inputs_2)
        next_state_output_2 = self.decoder(union_next_states_2)
        if self.flow_in_decoder:
            next_state_output_1 = self.flow_model.g(
                self.decoder(union_next_states_1).double()
            )[0].float()
            next_state_output_2 = self.flow_model.g(
                self.decoder(union_next_states_2).double()
            )[0].float()
        next_state_output_1 = torch.cat(
            [
                next_state_output_1,
                torch.zeros([next_state_output_1.shape[0], 1]).to(self.device),
            ],
            -1,
        )
        next_action_output_1 = self.action_decoder(next_union_inputs_1)
        next_state_output_2 = torch.cat(
            [
                next_state_output_2,
                torch.zeros([next_state_output_2.shape[0], 1]).to(self.device),
            ],
            -1,
        )
        next_action_output_2 = self.action_decoder(next_union_inputs_2)
        if self.flow_in_decoder:
            next_action_output_1 = self.action_flow_model.g(
                self.action_decoder(next_union_inputs_1).double()
            )[0].float()
            next_action_output_2 = self.action_flow_model.g(
                self.action_decoder(next_union_inputs_2).double()
            )[0].float()
        src_next_qvalue_1 = self.src_critic(
            torch.cat([next_state_output_1, next_action_output_1], -1)
        )
        src_union_adv_nu_1 = (
            -union_cost_1.detach() + self.discount * src_next_qvalue_1 - src_qvalue_1
        )
        src_next_qvalue_2 = self.src_critic(
            torch.cat([next_state_output_2, next_action_output_2], -1)
        )
        src_union_adv_nu_2 = (
            -union_cost_2.detach() + self.discount * src_next_qvalue_2 - src_qvalue_2
        )

        src_union_adv_nu_full = torch.cat(
            [src_union_adv_nu_1, src_union_adv_nu_2], dim=0
        )
        mapping_loss = torch.mean(src_union_adv_nu_full**2)

        # weighted BC
        weight_1 = torch.exp(
            (union_adv_nu_1 - torch.max(union_adv_nu_1))
            / self.non_expert_regularization
        ).unsqueeze(1)
        weight_1 = weight_1 / weight_1.mean()
        weight_2 = torch.exp(
            (union_adv_nu_2 - torch.max(union_adv_nu_2))
            / self.non_expert_regularization
        ).unsqueeze(1)
        weight_2 = weight_2 / weight_2.mean()

        src_union_adv_nu_1 = src_union_adv_nu_1.detach()
        src_weight_1 = torch.exp(
            (src_union_adv_nu_1 - torch.max(src_union_adv_nu_1))
            / self.non_expert_regularization
        ).unsqueeze(1)
        src_weight_1 = src_weight_1 / src_weight_1.mean()
        src_union_adv_nu_2 = src_union_adv_nu_2.detach()
        src_weight_2 = torch.exp(
            (src_union_adv_nu_2 - torch.max(src_union_adv_nu_2))
            / self.non_expert_regularization
        ).unsqueeze(1)
        src_weight_2 = src_weight_2 / src_weight_2.mean()

        # Adaptive decay weight1
        if timestep % 10 == 0:
            with torch.no_grad():
                all_union_states_1 = all_union_states[:state_len]
                all_union_next_states_1 = all_union_next_states[:state_len]
                all_union_actions_1 = all_union_actions[:state_len]

                all_state_output_1 = self.decoder(all_union_states_1)
                if self.flow_in_decoder:
                    all_state_output_1 = self.flow_model.g(
                        self.decoder(all_union_states_1).double()
                    )[0].float()
                all_state_output_1 = torch.cat(
                    [
                        all_state_output_1,
                        torch.zeros([all_state_output_1.shape[0], 1]).to(self.device),
                    ],
                    -1,
                )
                all_next_state_output_1 = self.decoder(all_union_next_states_1)
                if self.flow_in_decoder:
                    all_next_state_output_1 = self.flow_model.g(
                        self.decoder(all_union_next_states_1).double()
                    )[0].float()
                all_next_state_output_1 = torch.cat(
                    [
                        all_next_state_output_1,
                        torch.zeros([all_next_state_output_1.shape[0], 1]).to(
                            self.device
                        ),
                    ],
                    -1,
                )

                all_input_1 = torch.cat([all_union_states_1, all_union_actions_1], -1)
                all_action_output_1 = self.action_decoder(all_input_1)
                if self.flow_in_decoder:
                    all_action_output_1 = self.action_flow_model.g(
                        self.action_decoder(all_input_1).double()
                    )[0].float()
                all_src_input_1 = torch.cat(
                    [all_state_output_1, all_action_output_1], -1
                ).float()

                all_union_cost_val_1 = self.cost(all_input_1)
                all_union_cost_1 = torch.log(
                    1 / (all_union_cost_val_1 + EPS2) - 1 + EPS2
                )

                all_union_nu_1 = self.critic(all_union_states_1)
                all_union_next_nu_1 = self.critic(all_union_next_states_1)
                all_union_adv_nu_1 = (
                    -all_union_cost_1
                    + self.discount * all_union_next_nu_1
                    - all_union_nu_1
                )

                next_all_input_1 = torch.cat(
                    [all_union_next_states_1, self.actor(all_union_next_states_1)[0]],
                    -1,
                )
                all_next_action_output_1 = self.action_decoder(next_all_input_1)
                if self.flow_in_decoder:
                    all_next_action_output_1 = self.action_flow_model.g(
                        self.action_decoder(next_all_input_1).double()
                    )[0].float()
                all_src_next_input_1 = torch.cat(
                    [all_next_state_output_1, all_next_action_output_1], -1
                )

                all_src_qvalue_1 = self.src_critic(all_src_input_1)
                all_src_next_qvalue_1 = self.src_critic(all_src_next_input_1)
                all_src_union_adv_nu_1 = (
                    -all_union_cost_1
                    + self.discount * all_src_next_qvalue_1
                    - all_src_qvalue_1
                )

                self.c1_1 = (
                    torch.abs(
                        torch.exp(
                            (all_src_union_adv_nu_1 - torch.max(all_src_union_adv_nu_1))
                            / self.non_expert_regularization
                        )
                        - torch.exp(
                            (all_union_adv_nu_1 - torch.max(all_union_adv_nu_1))
                            / self.non_expert_regularization
                        )
                    )
                    .mean()
                    .item()
                )
                if hasattr(self, "prev_union_adv_nu_1"):
                    c2_1 = (
                        torch.abs(
                            torch.exp(
                                (all_union_adv_nu_1 - torch.max(all_union_adv_nu_1))
                                / self.non_expert_regularization
                            )
                            - torch.exp(
                                (
                                    self.prev_union_adv_nu_1
                                    - torch.max(self.prev_union_adv_nu_1)
                                )
                                / self.non_expert_regularization
                            )
                        )
                        .mean()
                        .item()
                    )
                    self.c2_smooth_1 = (
                        0.9 * self.c2_smooth_1 + 0.1 * c2_1
                        if hasattr(self, "c2_smooth_1")
                        else c2_1
                    )
                else:
                    self.c2_smooth_1 = 1.0
                self.prev_union_adv_nu_1 = all_union_adv_nu_1.detach().clone()

                all_union_states_2 = all_union_states[:-state_len]
                all_union_next_states_2 = all_union_next_states[:-state_len]
                all_union_actions_2 = all_union_actions[:-state_len]

                all_state_output_2 = self.decoder(all_union_states_2)
                if self.flow_in_decoder:
                    all_state_output_2 = self.flow_model.g(
                        self.decoder(all_union_states_2).double()
                    )[0].float()
                all_state_output_2 = torch.cat(
                    [
                        all_state_output_2,
                        torch.zeros([all_state_output_2.shape[0], 1]).to(self.device),
                    ],
                    -1,
                )
                all_next_state_output_2 = self.decoder(all_union_next_states_2)
                if self.flow_in_decoder:
                    all_next_state_output_2 = self.flow_model.g(
                        self.decoder(all_union_next_states_2).double()
                    )[0].float()
                all_next_state_output_2 = torch.cat(
                    [
                        all_next_state_output_2,
                        torch.zeros([all_next_state_output_2.shape[0], 1]).to(
                            self.device
                        ),
                    ],
                    -1,
                )

                all_input_2 = torch.cat([all_union_states_2, all_union_actions_2], -1)
                all_action_output_2 = self.action_decoder(all_input_2)
                if self.flow_in_decoder:
                    all_action_output_2 = self.action_flow_model.g(
                        self.action_decoder(all_input_2).double()
                    )[0].float()
                all_src_input_2 = torch.cat(
                    [all_state_output_2, all_action_output_2], -1
                ).float()

                all_union_cost_val_2 = self.cost(all_input_2)
                all_union_cost_2 = torch.log(
                    1 / (all_union_cost_val_2 + EPS2) - 1 + EPS2
                )

                all_union_nu_2 = self.critic(all_union_states_2)
                all_union_next_nu_2 = self.critic(all_union_next_states_2)
                all_union_adv_nu_2 = (
                    -all_union_cost_2
                    + self.discount * all_union_next_nu_2
                    - all_union_nu_2
                )

                next_all_input_2 = torch.cat(
                    [all_union_next_states_2, self.actor(all_union_next_states_2)[0]],
                    -1,
                )
                all_next_action_output_2 = self.action_decoder(next_all_input_2)
                if self.flow_in_decoder:
                    all_next_action_output_2 = self.action_flow_model.g(
                        self.action_decoder(next_all_input_2).double()
                    )[0].float()
                all_src_next_input_2 = torch.cat(
                    [all_next_state_output_2, all_next_action_output_2], -1
                )

                all_src_qvalue_2 = self.src_critic(all_src_input_2)
                all_src_next_qvalue_2 = self.src_critic(all_src_next_input_2)
                all_src_union_adv_nu_2 = (
                    -all_union_cost_2
                    + self.discount * all_src_next_qvalue_2
                    - all_src_qvalue_2
                )

                self.c1_2 = (
                    torch.abs(
                        torch.exp(
                            (all_src_union_adv_nu_2 - torch.max(all_src_union_adv_nu_2))
                            / self.non_expert_regularization
                        )
                        - torch.exp(
                            (all_union_adv_nu_2 - torch.max(all_union_adv_nu_2))
                            / self.non_expert_regularization
                        )
                    )
                    .mean()
                    .item()
                )
                if hasattr(self, "prev_union_adv_nu_2"):
                    c2_2 = (
                        torch.abs(
                            torch.exp(
                                (all_union_adv_nu_2 - torch.max(all_union_adv_nu_2))
                                / self.non_expert_regularization
                            )
                            - torch.exp(
                                (
                                    self.prev_union_adv_nu_2
                                    - torch.max(self.prev_union_adv_nu_2)
                                )
                                / self.non_expert_regularization
                            )
                        )
                        .mean()
                        .item()
                    )
                    self.c2_smooth_2 = (
                        0.9 * self.c2_smooth_2 + 0.1 * c2_2
                        if hasattr(self, "c2_smooth_2")
                        else c2_2
                    )
                else:
                    self.c2_smooth_2 = 1.0
                self.prev_union_adv_nu_2 = all_union_adv_nu_2.detach().clone()

        time_weight_decay_1 = self.c2_smooth_1 / (self.c1_1 + self.c2_smooth_1 + 1e-6)
        time_weight_decay_2 = self.c2_smooth_2 / (self.c1_2 + self.c2_smooth_2 + 1e-6)

        l2_loss = sum(p.norm(2).sum() for p in self.actor.parameters()) * 1e-2
        pi_loss = (
            -torch.mean(
                (
                    time_weight_decay_1 * src_weight_2.detach()
                    + (1 - time_weight_decay_1) * weight_2.detach()
                )
                * self.actor.get_log_prob(union_states_2, union_actions_2)
                + (
                    time_weight_decay_2 * src_weight_1.detach()
                    + (1 - time_weight_decay_2) * weight_1.detach()
                )
                * self.actor.get_log_prob(union_states_1, union_actions_1)
            )
            / 2.0
            + l2_loss
        )
        # pi_loss = - torch.mean(
        #     ((time_weight_decay_1 * src_weight_2.detach() + (1 - time_weight_decay_1) * weight_2.detach())
        #     + (time_weight_decay_2 * src_weight_1.detach() + (1 - time_weight_decay_2) * weight_1.detach())) * self.actor.get_log_prob(union_states, union_actions)
        # )+ l2_loss
        self.l2 = l2_loss.item()

        # Gradient penalty for nu
        if self.grad_reg_coeffs[1] is not None:
            unif_rand2 = torch.rand(expert_states.shape[0], 1).to(self.device)
            nu_inter = unif_rand2 * expert_states + (1 - unif_rand2) * union_states
            nu_next_inter = (
                unif_rand2 * expert_next_states + (1 - unif_rand2) * union_next_states
            )
            nu_inter = torch.cat([union_states, nu_inter, nu_next_inter], 0)

            nu_inter.requires_grad_()
            nu_output = self.critic(nu_inter)
            nu_mixed_grad = (
                torch.autograd.grad(
                    outputs=nu_output,
                    inputs=nu_inter,
                    grad_outputs=torch.ones_like(nu_output),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                + EPS
            )
            nu_grad_penalty = torch.mean(nu_mixed_grad.norm(2, dim=-1) ** 2)
            nu_loss += self.grad_reg_coeffs[1] * nu_grad_penalty

        mapping_loss.backward()
        nu_loss.backward()
        cost_loss.backward()
        pi_loss.backward()
        self.critic_optimizer.step()
        self.cost_optimizer.step()
        self.actor_optimizer.step()
        self.decoder_optimizer.step()
        self.action_decoder_optimizer.step()

        info_dict = {
            "actor_loss": pi_loss.item(),
            "mapping_loss": mapping_loss.item(),
        }
        return info_dict

    def step(self, observation, deterministic: bool = True):
        self.actor.eval()
        observation = torch.tensor([observation], dtype=torch.float32).to(self.device)
        all_actions = self.actor(observation)
        if deterministic:
            actions = all_actions[0]
        else:
            actions = all_actions[1]
        self.actor.train()
        return actions.detach().cpu()

    def get_training_state(self):
        training_state = {
            "cost_params": [
                (name, param.detach().cpu().numpy())
                for name, param in self.cost.named_parameters()
            ],
            "critic_params": [
                (name, param.detach().cpu().numpy())
                for name, param in self.critic.named_parameters()
            ],
            "actor_params": [
                (name, param.detach().cpu().numpy())
                for name, param in self.actor.named_parameters()
            ],
            "decoder_params": [
                (name, param.detach().cpu().numpy())
                for name, param in self.decoder.named_parameters()
            ],
            "action_decoder_params": [
                (name, param.detach().cpu().numpy())
                for name, param in self.action_decoder.named_parameters()
            ],
            "cost_optimizer_state": self.cost_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "decoder_optimizer_state": self.decoder_optimizer.state_dict(),
            "action_decoder_optimizer_state": self.action_decoder_optimizer.state_dict(),
        }
        return training_state

    def set_training_state(self, training_state):
        self.cost.load_state_dict(
            {name: torch.tensor(value) for name, value in training_state["cost_params"]}
        )
        self.critic.load_state_dict(
            {
                name: torch.tensor(value)
                for name, value in training_state["critic_params"]
            }
        )
        self.actor.load_state_dict(
            {
                name: torch.tensor(value)
                for name, value in training_state["actor_params"]
            }
        )
        self.decoder.load_state_dict(
            {
                name: torch.tensor(value)
                for name, value in training_state["decoder_params"]
            }
        )
        self.action_decoder.load_state_dict(
            {
                name: torch.tensor(value)
                for name, value in training_state["action_decoder_params"]
            }
        )
        self.cost_optimizer.load_state_dict(training_state["cost_optimizer_state"])
        self.critic_optimizer.load_state_dict(training_state["critic_optimizer_state"])
        self.actor_optimizer.load_state_dict(training_state["actor_optimizer_state"])
        self.decoder_optimizer.load_state_dict(
            training_state["decoder_optimizer_state"]
        )
        self.action_decoder_optimizer.load_state_dict(
            training_state["action_decoder_optimizer_state"]
        )

    def init_dummy(self, state_dim, action_dim):
        # Dummy train_step (to create optimizer variables)
        dummy_state = torch.zeros((1, state_dim), dtype=torch.float32)
        dummy_action = torch.zeros((1, action_dim), dtype=torch.float32)
        dummy_next_state = torch.zeros((1, state_dim), dtype=torch.float32)
        self.update(
            dummy_state,
            dummy_state,
            dummy_action,
            dummy_next_state,
            dummy_state,
            dummy_action,
            dummy_next_state,
        )  # type: ignore

    def save(self, filepath, training_info):
        print("Save checkpoint: ", filepath)
        training_state = self.get_training_state()
        data = {
            "training_state": training_state,
            "training_info": training_info,
        }
        with open(filepath + ".tmp", "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(filepath + ".tmp", filepath)
        print("Saved!")

    def load(self, filepath):
        print("Load checkpoint:", filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.set_training_state(data["training_state"])
        return data
