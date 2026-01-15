### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import utils.utils as utils
import abc
import pickle
import os


class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""


class GWIL(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, config,
                 discount=0.99, init_temperature=0.1, alpha_lr=1e-4, alpha_betas=[0.9, 0.999],
                 actor_betas=[0.9, 0.999], actor_update_frequency=1,
                 critic_betas=[0.9, 0.999], critic_tau=0.005, critic_target_update_frequency=2,
                 learnable_temperature=True):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(config['device'])
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = config['batch_size']
        self.learnable_temperature = learnable_temperature

        self.actor = utils.DiagGaussianActor(obs_dim, action_dim, config['hidden_dim'], config['hidden_depth'], config['log_std_bounds']).to(self.device)
        self.critic = utils.DoubleQCritic(obs_dim, action_dim, config['hidden_dim'], config['hidden_depth']).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=config['actor_lr'],
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=config['critic_lr'],
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done,
                      step, result={}):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        # logger.log('train_critic/loss', critic_loss, step)
        result.update({
            'train_critic/loss': critic_loss,
        })
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(step)
        return result

    def update_actor_and_alpha(self, obs, step, result={}):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # logger.log('train_actor/loss', actor_loss, step)
        # logger.log('train_actor/target_entropy', self.target_entropy, step)
        # logger.log('train_actor/entropy', -log_prob.mean(), step)
        result.update({
            'train_actor/loss': actor_loss,
        })
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # self.actor.log(step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            # logger.log('train_alpha/loss', alpha_loss, step)
            # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        
        return result

    def update(self, replay_buffer, step, gw=False, normalize_reward=False, normalize_reward_batch=False, include_external_reward=False, weight_external_reward=1, weight_gw_reward=1):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size, gw=gw, normalize_reward=normalize_reward, normalize_reward_batch=normalize_reward_batch, include_external_reward=include_external_reward, weight_external_reward=weight_external_reward, weight_gw_reward=weight_gw_reward)

        loss_result = self.update_critic(obs, action, reward, next_obs, not_done_no_max, step, result={})

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step, result=loss_result)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
            
        return loss_result
    
    def get_training_state(self):
        training_state = {
            'critic_params': [(name, param.detach().cpu().numpy()) for name, param in self.critic.named_parameters()],
            'critic_target_params': [(name, param.detach().cpu().numpy()) for name, param in self.critic_target.named_parameters()],
            'actor_params': [(name, param.detach().cpu().numpy()) for name, param in self.actor.named_parameters()],
            'log_alpha': self.log_alpha.detach().cpu().numpy(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'log_alpha_optimizer_state': self.log_alpha_optimizer.state_dict(),
        }
        return training_state

    def set_training_state(self, training_state):
        self.critic.load_state_dict({name: torch.tensor(value, device=self.device) for name, value in training_state['critic_params']})
        self.critic_target.load_state_dict({name: torch.tensor(value, device=self.device) for name, value in training_state['critic_target_params']})
        self.actor.load_state_dict({name: torch.tensor(value, device=self.device) for name, value in training_state['actor_params']})
        self.log_alpha.data = torch.tensor(training_state['log_alpha'], device=self.device, requires_grad=True)
        self.critic_optimizer.load_state_dict(training_state['critic_optimizer_state'])
        self.actor_optimizer.load_state_dict(training_state['actor_optimizer_state'])
        self.log_alpha_optimizer.load_state_dict(training_state['log_alpha_optimizer_state'])

    def init_dummy(self, state_dim, action_dim):
        # Create a dummy replay buffer to match the update method's expected input
        class DummyReplayBuffer:
            def __init__(self) -> None:
                self.device = "cuda"
            def sample(self, batch_size, gw=False, normalize_reward=False, normalize_reward_batch=False, include_external_reward=False, weight_external_reward=1, weight_gw_reward=1):
                obs = torch.zeros((batch_size, state_dim), dtype=torch.float32, device=self.device)
                action = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=self.device)
                reward = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
                next_obs = torch.zeros((batch_size, state_dim), dtype=torch.float32, device=self.device)
                not_done = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
                not_done_no_max = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
                return obs, action, reward, next_obs, not_done, not_done_no_max

        dummy_replay_buffer = DummyReplayBuffer()
        self.update(dummy_replay_buffer, step=0)

    def save(self, filepath, training_info):
        print(f'Saving checkpoint to: {filepath}')
        training_state = self.get_training_state()
        data = {
            'training_state': training_state,
            'training_info': training_info,
        }
        try:
            # Write to a temporary file first to avoid corruption
            temp_filepath = filepath + '.tmp'
            with open(temp_filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Atomically rename the temporary file to the final filepath
            os.replace(temp_filepath, filepath)
            print(f'Successfully saved checkpoint to: {filepath}')
        except Exception as e:
            print(f'Error saving checkpoint to {filepath}: {str(e)}')
            raise

    def load(self, filepath):
        print(f'Loading checkpoint from: {filepath}')
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.set_training_state(data['training_state'])
            print(f'Successfully loaded checkpoint from: {filepath}')
            return data
        except Exception as e:
            print(f'Error loading checkpoint from {filepath}: {str(e)}')
            raise