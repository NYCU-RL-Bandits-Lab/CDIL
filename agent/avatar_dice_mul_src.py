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

#from QAvatar.target_domain.core.flow.real_nvp import RealNvp
import utils.utils as utils
from utils.real_nvp import RealNvp

EPS = np.finfo(np.float32).eps
EPS2 = 1e-3

class Avatar(nn.Module):
    """ Class that implements DemoDICE training in PyTorch """
                                            
    def __init__(self, state_dim, action_dim, is_discrete_action: bool, src1_critic, src2_critic, src1_state_dim, src1_action_dim, src2_state_dim, src2_action_dim, config):
        super(Avatar, self).__init__()
        hidden_size = config['hidden_size']
        critic_lr = config['critic_lr']
        actor_lr = config['actor_lr']
        self.is_discrete_action = is_discrete_action
        self.grad_reg_coeffs = config['grad_reg_coeffs']
        self.discount = config['gamma']
        self.non_expert_regularization = config['alpha'] + 1.
        self.flow_in_decoder = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cost = utils.Critic(state_dim, action_dim, hidden_size=hidden_size, output_activation_fn=torch.sigmoid).to(self.device)
        self.critic = utils.Critic(state_dim, 0, hidden_size=hidden_size).to(self.device)

        self.decoder1 = utils.decoder_network(src1_state_dim-1, state_dim, hidden_size, self.device).to(self.device)
        self.action_decoder1 = utils.action_decoder_network(src1_action_dim, state_dim+action_dim, hidden_size, self.device).to(self.device)
        self.decoder2 = utils.decoder_network(src2_state_dim-1, state_dim, hidden_size, self.device).to(self.device)
        self.action_decoder2 = utils.action_decoder_network(src2_action_dim, state_dim+action_dim, hidden_size, self.device).to(self.device)
        
        self.src1_critic = src1_critic.eval()
        self.src2_critic = src2_critic.eval()
        
        if self.is_discrete_action:
            self.actor = utils.DiscreteActor(state_dim, action_dim).to(self.device)
        else:
            self.actor = utils.TanhActor(state_dim, action_dim, hidden_size=hidden_size).to(self.device)

        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=critic_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.decoder_optimizer1 = optim.Adam(self.decoder1.parameters(), lr=critic_lr)
        self.action_decoder_optimizer1 = optim.Adam(self.action_decoder1.parameters(), lr=critic_lr)
        self.decoder_optimizer2 = optim.Adam(self.decoder2.parameters(), lr=critic_lr)
        self.action_decoder_optimizer2 = optim.Adam(self.action_decoder2.parameters(), lr=critic_lr)

        self.c1 = 1.0
        self.c2 = 1.0
        self.c3_smooth = 1.0
        self.l2 = 0.0

        self.d1 = 1.0
        self.d2 = 1.0
        self.d3 = 1.0


    def update(self, init_states, expert_states, expert_actions, expert_next_states,
               union_states, union_actions, union_next_states, union_indices, timestep, power_weight_decay=1):#,source_shift=None, source_scale=None):
        self.cost_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.decoder_optimizer1.zero_grad()
        self.action_decoder_optimizer1.zero_grad()
        self.decoder_optimizer2.zero_grad()
        self.action_decoder_optimizer2.zero_grad()

        all_union_states = union_states
        all_union_actions = union_actions
        all_union_next_states = union_next_states
        union_states = union_states[union_indices]
        union_actions = union_actions[union_indices]
        union_next_states = union_next_states[union_indices]
        
        expert_inputs = torch.cat([expert_states, expert_actions], -1)
        union_inputs = torch.cat([union_states, union_actions], -1)

        state_mapping_output1 = self.decoder1(union_states).double()
        state_mapping_output1 = torch.cat([state_mapping_output1, torch.zeros([state_mapping_output1.shape[0], 1]).to(self.device)], -1)
        action_mapping_output1 = self.action_decoder1(union_inputs)
        src_union_inputs1 = torch.cat([state_mapping_output1, action_mapping_output1], -1)
        state_mapping_output2 = self.decoder2(union_states).double()
        state_mapping_output2 = torch.cat([state_mapping_output2, torch.zeros([state_mapping_output2.shape[0], 1]).to(self.device)], -1)
        action_mapping_output2 = self.action_decoder2(union_inputs)
        src_union_inputs = torch.cat([state_mapping_output2, action_mapping_output2], -1)

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
        union_nu = self.critic(union_states)
        union_next_nu = self.critic(union_next_states)
        union_adv_nu = - union_cost.detach() + self.discount * union_next_nu - union_nu

        non_linear_loss = self.non_expert_regularization * torch.logsumexp(
            union_adv_nu / self.non_expert_regularization, dim=0)
        linear_loss = (1 - self.discount) * init_nu.mean()
        nu_loss = non_linear_loss + linear_loss
        # regularization of nu loss
        lambda_ = 0.001
        nu_loss = nu_loss + lambda_*(init_nu**2).mean()/2.0

        # mapping function learning
        next_actions = self.actor(union_next_states)[0]
        next_union_inputs = torch.cat([union_next_states, next_actions], -1)        
        src1_qvalue = self.src1_critic(src_union_inputs)
        src2_qvalue = self.src2_critic(src_union_inputs)
        next_state_output1 = self.decoder1(union_next_states)
        next_state_output1 = torch.cat([next_state_output1, torch.zeros([next_state_output1.shape[0], 1]).to(self.device)], -1)
        next_action_output1 = self.action_decoder1(next_union_inputs)
        next_state_output2 = self.decoder2(union_next_states)
        next_state_output2 = torch.cat([next_state_output2, torch.zeros([next_state_output2.shape[0], 1]).to(self.device)], -1)
        next_action_output2 = self.action_decoder2(next_union_inputs)
        src1_next_qvalue = self.src1_critic(torch.cat([next_state_output1, next_action_output1], -1))
        src2_next_qvalue = self.src2_critic(torch.cat([next_state_output2, next_action_output2], -1))
        src1_union_adv_nu = - union_cost.detach() + self.discount*src1_next_qvalue - src1_qvalue
        src2_union_adv_nu = - union_cost.detach() + self.discount*src2_next_qvalue - src2_qvalue

        # mapping_loss1 = torch.mean(src1_union_adv_nu**2)
        # mapping_loss2 = torch.mean(src2_union_adv_nu**2)
        mapping_loss = torch.mean(src1_union_adv_nu**2) + torch.mean(src2_union_adv_nu**2)
        
        # weighted BC
        weight = torch.exp((union_adv_nu - torch.max(union_adv_nu)) / self.non_expert_regularization).unsqueeze(1)
        weight = weight / weight.mean()

        src1_union_adv_nu = src1_union_adv_nu.detach()
        src1_weight = torch.exp((src1_union_adv_nu - torch.max(src1_union_adv_nu)) / self.non_expert_regularization).unsqueeze(1)
        src1_weight = src1_weight / src1_weight.mean()

        src2_union_adv_nu = src2_union_adv_nu.detach()
        src2_weight = torch.exp((src2_union_adv_nu - torch.max(src2_union_adv_nu)) / self.non_expert_regularization).unsqueeze(1)
        src2_weight = src2_weight / src2_weight.mean()

        # Adaptive decay weight
        if timestep % 10 == 0:
            with torch.no_grad():
                all_state_output1 = self.decoder1(all_union_states)
                all_state_output1 = torch.cat([all_state_output1, torch.zeros([all_state_output1.shape[0], 1]).to(self.device)], -1)
                all_next_state_output1 = self.decoder1(all_union_next_states)
                all_next_state_output1 = torch.cat([all_next_state_output1, torch.zeros([all_next_state_output1.shape[0], 1]).to(self.device)], -1)
                all_state_output2 = self.decoder2(all_union_states)
                all_state_output2 = torch.cat([all_state_output2, torch.zeros([all_state_output2.shape[0], 1]).to(self.device)], -1)
                all_next_state_output2 = self.decoder2(all_union_next_states)
                all_next_state_output2 = torch.cat([all_next_state_output2, torch.zeros([all_next_state_output2.shape[0], 1]).to(self.device)], -1)
                
                all_input = torch.cat([all_union_states, all_union_actions], -1)
                all_action_output1 = self.action_decoder1(all_input)
                all_src_input1 = torch.cat([all_state_output1, all_action_output1], -1)
                all_action_output2 = self.action_decoder2(all_input)
                all_src_input2 = torch.cat([all_state_output2, all_action_output2], -1)

                all_union_cost_val = self.cost(all_input)
                all_union_cost = torch.log(1 / (all_union_cost_val + EPS2) - 1 + EPS2)

                all_union_nu = self.critic(all_union_states)
                all_union_next_nu = self.critic(all_union_next_states)
                all_union_adv_nu = - all_union_cost + self.discount * all_union_next_nu - all_union_nu

                next_all_input = torch.cat([all_union_next_states, self.actor(all_union_next_states)[0]], -1)
                all_next_action_output1 = self.action_decoder1(next_all_input)
                all_src_next_input1 = torch.cat([all_next_state_output1, all_next_action_output1], -1)
                all_next_action_output2 = self.action_decoder2(next_all_input)
                all_src_next_input2 = torch.cat([all_next_state_output2, all_next_action_output2], -1)

                all_src1_qvalue = self.src1_critic(all_src_input1)
                all_src1_next_qvalue = self.src1_critic(all_src_next_input1)
                all_src2_qvalue = self.src2_critic(all_src_input2)
                all_src2_next_qvalue = self.src2_critic(all_src_next_input2)
                all_src1_union_adv_nu = - all_union_cost + self.discount * all_src1_next_qvalue - all_src1_qvalue
                all_src2_union_adv_nu = - all_union_cost + self.discount * all_src2_next_qvalue - all_src2_qvalue

                self.c1 = torch.abs(
                    torch.exp((all_src1_union_adv_nu - torch.max(all_src1_union_adv_nu)) / self.non_expert_regularization) - 
                    torch.exp((all_union_adv_nu - torch.max(all_union_adv_nu)) / self.non_expert_regularization)
                ).mean().item()
                self.c2 = torch.abs(
                    torch.exp((all_src2_union_adv_nu - torch.max(all_src2_union_adv_nu)) / self.non_expert_regularization) - 
                    torch.exp((all_union_adv_nu - torch.max(all_union_adv_nu)) / self.non_expert_regularization)
                ).mean().item()
                if hasattr(self, 'prev_union_adv_nu'):
                    c3 = torch.abs(
                        torch.exp((all_union_adv_nu - torch.max(all_union_adv_nu)) / self.non_expert_regularization) - 
                        torch.exp((self.prev_union_adv_nu - torch.max(self.prev_union_adv_nu)) / self.non_expert_regularization)
                    ).mean().item()
                    self.c3_smooth = 0.9 * self.c3_smooth + 0.1 * c3 if hasattr(self, 'c2_smooth') else c3
                else:
                    self.c3_smooth = 1.0
                self.prev_union_adv_nu = all_union_adv_nu.detach().clone()
        # 計算 alpha(t) = c2c3 / (c2c3 + c1c3 + c1c2)
        # time_weight_decay = self.c3_smooth**power_weight_decay / (self.c1**power_weight_decay + self.c2_smooth**power_weight_decay + 1e-6)
        d1 = self.c2 * self.c3_smooth
        d2 = self.c1 * self.c3_smooth
        d3 = self.c1 * self.c2
        total_weight = d1 + d2 + d3 + 1e-6

        l2_loss = sum(p.norm(2).sum() for p in self.actor.parameters()) * 1e-2
        pi_loss = - torch.mean(
            (d1/total_weight * src1_weight.detach() + d2/total_weight * src2_weight.detach() + d3/total_weight * weight.detach()) * self.actor.get_log_prob(union_states, union_actions)
        ) + l2_loss
        self.l2 = l2_loss.item()
                
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

        # mapping_loss1.backward()
        # mapping_loss2.backward()
        mapping_loss.backward()
        nu_loss.backward()
        cost_loss.backward()
        pi_loss.backward()
        self.critic_optimizer.step()
        self.cost_optimizer.step()
        self.actor_optimizer.step()
        self.decoder_optimizer1.step()
        self.action_decoder_optimizer1.step()
        self.decoder_optimizer2.step()
        self.action_decoder_optimizer2.step()

        info_dict = {
            'actor_loss': pi_loss.item(),
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
            'decoder_params': [(name, param.detach().cpu().numpy()) for name, param in self.decoder.named_parameters()],
            'action_decoder_params': [(name, param.detach().cpu().numpy()) for name, param in self.action_decoder.named_parameters()],
            'cost_optimizer_state': self.cost_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'decoder_optimizer_state': self.decoder_optimizer.state_dict(),
            'action_decoder_optimizer_state': self.action_decoder_optimizer.state_dict(),
        }
        return training_state

    def set_training_state(self, training_state):
        self.cost.load_state_dict({name: torch.tensor(value) for name, value in training_state['cost_params']})
        self.critic.load_state_dict({name: torch.tensor(value) for name, value in training_state['critic_params']})
        self.actor.load_state_dict({name: torch.tensor(value) for name, value in training_state['actor_params']})
        self.decoder.load_state_dict({name: torch.tensor(value) for name, value in training_state['decoder_params']})
        self.action_decoder.load_state_dict({name: torch.tensor(value) for name, value in training_state['action_decoder_params']})
        self.cost_optimizer.load_state_dict(training_state['cost_optimizer_state'])
        self.critic_optimizer.load_state_dict(training_state['critic_optimizer_state'])
        self.actor_optimizer.load_state_dict(training_state['actor_optimizer_state'])
        self.decoder_optimizer.load_state_dict(training_state['decoder_optimizer_state'])
        self.action_decoder_optimizer.load_state_dict(training_state['action_decoder_optimizer_state'])

    def init_dummy(self, state_dim, action_dim):
        # Dummy train_step (to create optimizer variables)
        dummy_state = torch.zeros((1, state_dim), dtype=torch.float32)
        dummy_action = torch.zeros((1, action_dim), dtype=torch.float32)
        dummy_next_state = torch.zeros((1, state_dim), dtype=torch.float32)
        self.update(dummy_state, dummy_state, dummy_action, dummy_next_state,
                    dummy_state, dummy_action, dummy_next_state)
        
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
    





# ant
"""
# 2 sources
export CUDA_VISIBLE_DEVICES=0 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=1 \
  --algorithm=multisrc \
  --env_id=Ant \
  --load_hdf5_dataset=1 \
  --log_interval=10000 \
  --dataset_file_names="['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']" \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=tfboard/eye_catching/multisrc \
  --seed 0


# no source
export CUDA_VISIBLE_DEVICES=1 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=1 \
  --algorithm=demodice \
  --env_id=Ant \
  --load_hdf5_dataset=1 \
  --log_interval=10000 \
  --dataset_file_names="['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']" \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=tfboard/eye_catching/nosrc \
  --seed 0


python train_il.py \
  --env_is_gym=0 \
  --algorithm=demodice \
  --env_id=Ant \
  --xml_path=./env/ant_target.xml \
  --dataset_file_names="['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=demodice_ablation/ant_set1_seed0 \
  --seed 0
"""