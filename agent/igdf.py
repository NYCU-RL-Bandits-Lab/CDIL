import numpy as np
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils
from utils.contrastiveoi import ContrastiveInfo
from tqdm import trange

TensorBatch = List[torch.Tensor]

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        # self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        # rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, next_states, dones]#rewards, 

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        if self._pointer >= self._buffer_size:
            self._pointer = 0

        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        # self._rewards[self._pointer] = self._to_tensor(np.array([reward], dtype=np.float32))
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(np.array([done], dtype=np.float32))

        self._pointer += 1
        if self._size < self._buffer_size:
            self._size += 1


def merge_batch(source_batch, target_batch) -> TensorBatch:
    merged_states  = torch.concat((source_batch[0], target_batch[0]), axis = 0)
    merged_actions = torch.concat((source_batch[1], target_batch[1]), axis = 0)
    # merged_rewards = torch.concat((source_batch[2], target_batch[2]), axis = 0)
    merged_next_states = torch.concat((source_batch[2], target_batch[2]), axis = 0)
    merged_dones = torch.concat((source_batch[3], target_batch[3]), axis = 0)

    return [merged_states, merged_actions, merged_next_states, merged_dones]#merged_rewards,


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def load_data_and_train_contras(
    env_id, dataset_dir, config, observation_dim, action_dim, 
    union_states, union_actions, union_next_states, union_dones,
    expert_states, expert_actions, expert_next_states, expert_dones
):
    """
    Load source and target datasets, process dimensions, create replay buffers, 
    and train ContrastiveInfo model.

    Args:
        env_id (str): Environment ID (e.g., 'Hopper-v2').
        dataset_dir (str): Directory containing dataset files.
        config (dict): Configuration dictionary containing settings like 
                      'src_expert_path', 'src_random_path', 'src_env_robot', 
                      'buffer_size', 'device', 'info_batch_size', 'info_lr', 
                      'repr_dim', 'ensemble_size', 'num_update'.
        observation_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        union_states (np.ndarray): Target union states (expert + non-expert).
        union_actions (np.ndarray): Target union actions.
        union_next_states (np.ndarray): Target union next states.
        union_dones (np.ndarray): Target union done flags.
        expert_states (np.ndarray): Target expert states.
        expert_actions (np.ndarray): Target expert actions.
        expert_next_states (np.ndarray): Target expert next states.
        expert_dones (np.ndarray): Target expert done flags.
        igdf: Module containing ReplayBuffer class.
        ContrastiveInfo: Class for contrastive information model.

    Returns:
        tuple: (target_buffer, target_expert_buffer, source_buffer, source_expert_buffer, info, total_steps)
    """
    # xml_path = config['xml_path']
    # if xml_path:
    (src_expert_initial_states, src_expert_states, src_expert_actions,
        src_expert_next_states, src_expert_dones) = utils.load_d4rl_data(
        dataset_dir, env_id + '-v2', 'expert-v2', 400, start_idx=0)
    (src_random_initial_states, src_random_states, src_random_actions,
        src_random_next_states, src_random_dones) = utils.load_d4rl_data(
        dataset_dir, env_id + '-v2', 'random-v2', 1600, start_idx=0)
    # else:
        # src_dataset_path = os.path.join(dataset_dir, config['src_expert_path'])
        # (src_expert_initial_states, src_expert_states, src_expert_actions,
        #     src_expert_next_states, src_expert_dones) = utils.sample_demonstrations(
        #     env_id=env_id, num_trajectories=400, load_path=src_dataset_path,
        #     max_episode_steps=500, difficulty='expert', dtype=np.float32,
        #     env_robot=config['src_env_robot'])
        # src_dataset_path = os.path.join(dataset_dir, config['src_random_path'])
        # (src_random_initial_states, src_random_states, src_random_actions,
        #     src_random_next_states, src_random_dones) = utils.sample_demonstrations(
        #     env_id=env_id, num_trajectories=1600, load_path=src_dataset_path,
        #     max_episode_steps=500, difficulty='random', dtype=np.float32,
        #     env_robot=config['src_env_robot'])

    # Concatenate source expert and random data
    src_union_states = np.concatenate([src_expert_states, src_random_states]).astype(np.float32)
    src_union_actions = np.concatenate([src_expert_actions, src_random_actions]).astype(np.float32)
    src_union_next_states = np.concatenate([src_expert_next_states, src_random_next_states]).astype(np.float32)
    src_union_dones = np.concatenate([src_expert_dones, src_random_dones]).astype(np.float32)

    # Pad or truncate source states and actions to match target dimensions
    if observation_dim - src_union_states.shape[1] >= 0:
        s_padding = np.zeros((src_union_states.shape[0], observation_dim - src_union_states.shape[1]))
        src_union_states = np.concatenate([src_union_states, s_padding], axis=1)
        src_union_next_states = np.concatenate([src_union_next_states, s_padding], axis=1)
    else:
        src_union_states = src_union_states[:, :observation_dim]
        src_union_next_states = src_union_next_states[:, :observation_dim]

    if action_dim - src_union_actions.shape[1] >= 0:
        a_padding = np.zeros((src_union_actions.shape[0], action_dim - src_union_actions.shape[1]))
        src_union_actions = np.concatenate([src_union_actions, a_padding], axis=1)
    else:
        src_union_actions = src_union_actions[:, :action_dim]

    if observation_dim - src_expert_states.shape[1] >= 0:
        s_padding = np.zeros((src_expert_states.shape[0], observation_dim - src_expert_states.shape[1]))
        src_expert_states = np.concatenate([src_expert_states, s_padding], axis=1)
        src_expert_next_states = np.concatenate([src_expert_next_states, s_padding], axis=1)
    else:
        src_expert_states = src_expert_states[:, :observation_dim]
        src_expert_next_states = src_expert_next_states[:, :observation_dim]

    if action_dim - src_expert_actions.shape[1] >= 0:
        a_padding = np.zeros((src_expert_actions.shape[0], action_dim - src_expert_actions.shape[1]))
        src_expert_actions = np.concatenate([src_expert_actions, a_padding], axis=1)
    else:
        src_expert_actions = src_expert_actions[:, :action_dim]

    # Create data dictionaries for replay buffers
    target_data = {
        'observations': union_states,
        'actions': union_actions,
        'next_observations': union_next_states,
        'terminals': union_dones
    }
    target_expert_data = {
        'observations': expert_states,
        'actions': expert_actions,
        'next_observations': expert_next_states,
        'terminals': expert_dones
    }
    source_data = {
        'observations': src_union_states,
        'actions': src_union_actions,
        'next_observations': src_union_next_states,
        'terminals': src_union_dones
    }
    source_expert_data = {
        'observations': src_expert_states,
        'actions': src_expert_actions,
        'next_observations': src_expert_next_states,
        'terminals': src_expert_dones
    }

    # Initialize replay buffers
    target_buffer = ReplayBuffer(
        observation_dim, action_dim, config['buffer_size'], config['device']
    )
    target_expert_buffer = ReplayBuffer(
        observation_dim, action_dim, 500000, config['device']
    )
    source_buffer = ReplayBuffer(
        observation_dim, action_dim, config['buffer_size'], config['device']
    )
    source_expert_buffer = ReplayBuffer(
        observation_dim, action_dim, 500000, config['device']
    )

    # Load data into replay buffers
    target_buffer.load_d4rl_dataset(target_data)
    target_expert_buffer.load_d4rl_dataset(target_expert_data)
    source_buffer.load_d4rl_dataset(source_data)
    source_expert_buffer.load_d4rl_dataset(source_expert_data)

    # Initialize ContrastiveInfo model and optimizer
    info = ContrastiveInfo(
        observation_dim, action_dim, config['repr_dim'], config['ensemble_size'],
        False, False, False, None
    ).to(config['device'])
    info_optimizer = torch.optim.Adam(info.parameters(), lr=config['info_lr'])

    # Training loop
    total_steps = 0
    for train_step in trange(config['num_update'], desc="Training"):
        # Sample data from buffers
        tar_s, tar_a, tar_ss, _ = target_buffer.sample(config['info_batch_size'])
        _, _, src_ss, _ = source_buffer.sample(config['info_batch_size'] - 1)

        # Reshape tensors
        tar_s = tar_s.unsqueeze(1)  # [batch_size, 1, state_dim]
        tar_a = tar_a.unsqueeze(1)  # [batch_size, 1, action_dim]
        tar_ss = tar_ss.unsqueeze(1)  # [batch_size, 1, state_dim]
        src_ss = src_ss.unsqueeze(0)  # [1, batch_size-1, state_dim]
        src_ss = src_ss.expand(config['info_batch_size'], -1, -1)  # [batch_size, batch_size-1, state_dim]
        ss = torch.concat((tar_ss, src_ss), dim=1)  # [batch_size, batch_size, state_dim]

        # Compute contrastive loss
        logits = info(tar_s, tar_a, ss)  # [batch_size, 1, batch_size]
        logits = logits.squeeze(1)
        matrix = torch.zeros((config['info_batch_size'], config['info_batch_size']), 
                            dtype=torch.float32, device=config['device'])
        matrix[:, 0] = 1

        info_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, matrix)
        info_loss = torch.mean(info_loss)

        # Backpropagation and optimization
        info_optimizer.zero_grad()
        info_loss.backward()
        info_optimizer.step()

        total_steps += 1

    return target_buffer, target_expert_buffer, source_buffer, source_expert_buffer, info

class IQ_Learn:
    def __init__(
        self,
        agent,
        discount: float = 0.99,
        device: str = "cuda",
    ):
        self.agent = agent
        self.discount = discount
        self.device = device
              
    def _update_q(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        log_dict,
        mask,
    ):
        current_V = self.agent.getV(obs)
        next_V = self.agent.getV(next_obs)
        current_Q1, current_Q2 = self.agent.critic(obs, actions, both=True)

        gamma = self.discount
        y = (1 - done) * gamma * next_V

        reward_1 = (current_Q1 - y)
        reward_2 = (current_Q2 - y)
        loss_1 = -(mask * reward_1).mean()
        loss_2 = -(mask * reward_2).mean()

        value_loss = (mask * (current_V - y)).mean()
        loss_1 += value_loss
        loss_2 += value_loss

        chi2_loss_1 = 1/(4 * 0.5) * (mask * reward_1**2).mean()
        chi2_loss_2 = 1/(4 * 0.5) * (mask * reward_2**2).mean()
        loss_1 += chi2_loss_1
        loss_2 += chi2_loss_2

        log_dict["q1_loss"] = loss_1.item()
        log_dict["q2_loss"] = loss_2.item()

        critic_loss = 1/2 * (loss_1 + loss_2)

        self.agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.agent.critic_optimizer.step()

        return log_dict

    def train(self, batch: TensorBatch, mask: TensorBatch, step) -> Dict[str, float]:
        (
            obs,
            actions,
            next_observations,
            dones,
        ) = batch

        log_dict = {}
        # Update Q function
        self._update_q(obs, actions, next_observations, dones, log_dict, mask)

        self.agent.update_actor_and_alpha(obs, log_dict, step)
        soft_update(self.agent.critic_net, self.agent.critic_target_net,
                    self.agent.critic_tau)

        return log_dict