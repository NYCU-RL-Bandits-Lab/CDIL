import os
import gym
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import pickle
import pandas as pd

import utils.utils as utils


EPS = np.finfo(np.float32).eps
EPS2 = 1e-3


class DemoDICE(nn.Module):
    """ Class that implements DemoDICE training in PyTorch """
    def __init__(self, state_dim, action_dim, is_discrete_action: bool, config):
        super(DemoDICE, self).__init__()
        hidden_size = config['hidden_size']
        critic_lr = config['critic_lr']
        actor_lr = config['actor_lr']
        self.is_discrete_action = is_discrete_action
        self.grad_reg_coeffs = config['grad_reg_coeffs']
        self.discount = config['gamma']
        self.non_expert_regularization = config['alpha'] + 1.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cost = utils.Critic(state_dim, action_dim, hidden_size=hidden_size, output_activation_fn=torch.sigmoid).to(self.device)
        self.critic = utils.Critic(state_dim, 0, hidden_size=hidden_size).to(self.device)
        self.q_function = utils.Critic(state_dim, action_dim, hidden_size=hidden_size).to(self.device)
        
        if self.is_discrete_action:
            self.actor = utils.DiscreteActor(state_dim, action_dim).to(self.device)
        else:
            self.actor = utils.TanhActor(state_dim, action_dim, hidden_size=hidden_size).to(self.device)

        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=critic_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_optimizer = optim.Adam(self.q_function.parameters(), lr=critic_lr)  # Q function optimizer

    def update(self, init_states, expert_states, expert_actions, expert_next_states,
               union_states, union_actions, union_next_states, timestep):
        self.cost_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.q_optimizer.zero_grad()

        expert_inputs = torch.cat([expert_states, expert_actions], -1)
        union_inputs = torch.cat([union_states, union_actions], -1)

        expert_cost_val = self.cost(expert_inputs)
        union_cost_val = self.cost(union_inputs)

        unif_rand = torch.rand(expert_states.shape[0], 1).to(self.device)
        mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * union_inputs
        mixed_inputs2 = unif_rand * union_inputs[torch.randperm(union_inputs.size(0))] + (1 - unif_rand) * union_inputs
        mixed_inputs = torch.cat([mixed_inputs1, mixed_inputs2], 0)

        # Gradient penalty for cost
        mixed_inputs.requires_grad_()
        cost_output = self.cost(mixed_inputs)
        cost_output = torch.log(1 / (cost_output + EPS2) - 1 + EPS2)
        cost_mixed_grad = torch.autograd.grad(
            outputs=cost_output,
            inputs=mixed_inputs,
            grad_outputs=torch.ones_like(cost_output),
            create_graph=True,
            retain_graph=True)[0] + EPS
        cost_grad_penalty = torch.mean((cost_mixed_grad.norm(2, dim=-1) - 1) ** 2)
        cost_loss = (nn.BCEWithLogitsLoss()(expert_cost_val, torch.ones_like(expert_cost_val)) +
                     nn.BCEWithLogitsLoss()(union_cost_val, torch.zeros_like(union_cost_val)) +
                     self.grad_reg_coeffs[0] * cost_grad_penalty)

        union_cost = torch.log(1 / (union_cost_val + EPS2) - 1 + EPS2)

        # nu learning
        init_nu = self.critic(init_states)
        expert_nu = self.critic(expert_states)
        expert_next_nu = self.critic(expert_next_states)
        union_nu = self.critic(union_states)
        union_next_nu = self.critic(union_next_states)
        union_adv_nu = - union_cost.detach() + self.discount * union_next_nu - union_nu

        non_linear_loss = self.non_expert_regularization * torch.logsumexp(
            union_adv_nu / self.non_expert_regularization, dim=0)
        linear_loss = (1 - self.discount) * init_nu.mean()
        nu_loss = non_linear_loss + linear_loss

        # Q function learning
        q_value = self.q_function(union_inputs)
        q_loss = torch.mean((- union_cost.detach() + self.discount * union_next_nu.detach() - q_value) ** 2)

        # weighted BC
        weight = torch.exp((union_adv_nu - torch.max(union_adv_nu)) / self.non_expert_regularization).unsqueeze(1)
        weight = weight / weight.mean()
        pi_loss = - torch.mean(weight.detach() * self.actor.get_log_prob(union_states, union_actions))

        # Gradient penalty for nu
        if self.grad_reg_coeffs[1] is not None:
            unif_rand2 = torch.rand(expert_states.shape[0], 1).to(self.device)
            nu_inter = unif_rand2 * expert_states + (1 - unif_rand2) * union_states
            nu_next_inter = unif_rand2 * expert_next_states + (1 - unif_rand2) * union_next_states
            nu_inter = torch.cat([union_states, nu_inter, nu_next_inter], 0)

            nu_inter.requires_grad_()
            nu_output = self.critic(nu_inter)
            nu_mixed_grad = torch.autograd.grad(
                outputs=nu_output,
                inputs=nu_inter,
                grad_outputs=torch.ones_like(nu_output),
                create_graph=True,
                retain_graph=True)[0] + EPS
            nu_grad_penalty = torch.mean(nu_mixed_grad.norm(2, dim=-1) ** 2)
            nu_loss += self.grad_reg_coeffs[1] * nu_grad_penalty

        nu_loss.backward()
        cost_loss.backward()
        q_loss.backward()
        self.critic_optimizer.step()
        self.cost_optimizer.step()
        self.q_optimizer.step()

        pi_loss.backward()
        self.actor_optimizer.step()
        
        info_dict = {
            'cost_loss': cost_loss.item(),
            'nu_loss': nu_loss.item(),
            'actor_loss': pi_loss.item(),
            'q_loss': q_loss.item(),
            'expert_nu': expert_nu.mean().item(),
            'union_nu': union_nu.mean().item(),
            'init_nu': init_nu.mean().item(),
            'union_adv': union_adv_nu.mean().item(),
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
            'cost_params': [(name, param.detach().cpu().numpy()) for name, param in self.cost.named_parameters()],
            'critic_params': [(name, param.detach().cpu().numpy()) for name, param in self.critic.named_parameters()],
            'actor_params': [(name, param.detach().cpu().numpy()) for name, param in self.actor.named_parameters()],
            'q_params': [(name, param.detach().cpu().numpy()) for name, param in self.q_function.named_parameters()],
            'cost_optimizer_state': self.cost_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'q_optimizer_state': self.q_optimizer.state_dict(),
        }
        return training_state

    def set_training_state(self, training_state):
        self.cost.load_state_dict({name: torch.tensor(value) for name, value in training_state['cost_params']})
        self.critic.load_state_dict({name: torch.tensor(value) for name, value in training_state['critic_params']})
        self.actor.load_state_dict({name: torch.tensor(value) for name, value in training_state['actor_params']})
        self.q_function.load_state_dict({name: torch.tensor(value) for name, value in training_state['q_params']})
        self.cost_optimizer.load_state_dict(training_state['cost_optimizer_state'])
        self.critic_optimizer.load_state_dict(training_state['critic_optimizer_state'])
        self.actor_optimizer.load_state_dict(training_state['actor_optimizer_state'])
        self.q_optimizer.load_state_dict(training_state['q_optimizer_state'])

    def init_dummy(self, state_dim, action_dim):
        # Dummy train_step (to create optimizer variables)
        dummy_state = torch.zeros((1, state_dim), dtype=torch.float32).to(self.device)
        dummy_action = torch.zeros((1, action_dim), dtype=torch.float32).to(self.device)
        dummy_next_state = torch.zeros((1, state_dim), dtype=torch.float32).to(self.device)
        self.update(dummy_state, dummy_state, dummy_action, dummy_next_state,
                    dummy_state, dummy_action, dummy_next_state, 0)
        
    def save(self, filepath, training_info):
        print('Save checkpoint: ', filepath)
        training_state = self.get_training_state()
        data = {
            'training_state': training_state,
            'training_info': training_info,
        }
        with open(filepath + '.tmp', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(filepath + '.tmp', filepath)
        print('Saved!')

    def load(self, filepath):
        print('Load checkpoint:', filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.set_training_state(data['training_state'])
        return data