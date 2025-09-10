from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gym
import math
import gymnasium
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributions as distributions
import h5py
from urllib import request
import d4rl
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper


LOG_STD_MIN = -5
LOG_STD_MAX = 2
SCALE_DIAG_MIN_MAX = (LOG_STD_MIN, LOG_STD_MAX)
MEAN_MIN_MAX = (-7, 7)
EPS = np.finfo(np.float32).eps
KEYS = ['observations', 'actions', 'rewards', 'terminals']


# Inverse tanh torch function
def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))

# demodice/avatar_dice ====================================================
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class TanhActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, mean_range=(-7., 7.), logstd_range=(-5., 2.), initial_std_scaler=1):
        super(TanhActor, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_logstd = nn.Linear(hidden_size, action_dim)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.initial_std_scaler = initial_std_scaler

    def forward(self, x):
        h = self.fc_layers(x)
        mean = self.fc_mean(h).clamp(self.mean_min, self.mean_max)
        logstd = self.fc_logstd(h).clamp(self.logstd_min, self.logstd_max)
        std = torch.exp(logstd) * self.initial_std_scaler

        pretanh_action_dist = distributions.Normal(mean, std)
        pretanh_action = pretanh_action_dist.rsample()
        action = torch.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return torch.tanh(mean), action, log_prob

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = torch.tanh(pretanh_action)
        else:
            pretanh_action = torch.atanh(action.clamp(-1 + EPS, 1 - EPS))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action).sum(-1)
        log_prob = pretanh_log_prob - torch.sum(torch.log(1 - action.pow(2) + EPS), dim=-1)
        return log_prob, pretanh_log_prob

    def get_log_prob(self, states, actions):
        h = self.fc_layers(states)
        mean = self.fc_mean(h).clamp(self.mean_min, self.mean_max)
        logstd = self.fc_logstd(h).clamp(self.logstd_min, self.logstd_max)
        std = torch.exp(logstd) * self.initial_std_scaler

        pretanh_action_dist = distributions.Normal(mean, std)
        pretanh_actions = torch.atanh(actions.clamp(-1 + EPS, 1 - EPS))
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_actions).sum(-1)
        log_probs = pretanh_log_prob - torch.sum(torch.log(torch.clamp(1 - actions.pow(2), EPS, 2.0)), dim=-1)
        log_probs = log_probs.unsqueeze(-1)
        return log_probs


class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DiscreteActor, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.logit_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        h = self.fc_layers(x)
        logits = self.logit_layer(h)
        dist = distributions.Categorical(logits=logits)
        action = dist.sample().float()
        greedy_action = F.one_hot(logits.argmax(dim=-1), self.logit_layer.out_features).float()
        log_prob = dist.log_prob(action.long())
        return greedy_action, action, log_prob.unsqueeze(-1)

    def get_log_prob(self, states, actions):
        h = self.fc_layers(states)
        logits = self.logit_layer(h)
        dist = distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions.argmax(dim=-1).long()).unsqueeze(-1)
        return log_probs


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, output_activation_fn=None, use_last_layer_bias=False, output_dim=None):
        super(Critic, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        if use_last_layer_bias:
            self.last_layer = nn.Linear(hidden_size, output_dim or 1, bias=True)
            nn.init.uniform_(self.last_layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(self.last_layer.bias, -3e-3, 3e-3)
        else:
            self.last_layer = nn.Linear(hidden_size, output_dim or 1, bias=False)

        self.output_activation_fn = output_activation_fn

    def forward(self, x):
        h = self.fc_layers(x)
        h = self.last_layer(h)
        if self.output_activation_fn is not None:
            h = self.output_activation_fn(h)
        if self.last_layer.out_features == 1:
            h = h.view(-1)
        return h
    

class action_decoder_network(nn.Module):
    def __init__(self, source_action_dim, target_action_dim, hidden_size, device):
        super(action_decoder_network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(target_action_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, source_action_dim),
            nn.Tanh()
        )
        self.device = device
        self.target_action_dim = target_action_dim

    def forward(self, target_action):
        source_action = self.model(target_action.float()) * 0.5
        return source_action


class decoder_network(nn.Module):
    def __init__(self, source_state_dim, target_state_dim, hidden_size, device):
        super(decoder_network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(target_state_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, source_state_dim),
            nn.Tanh()
        )
        self.device = device
        self.target_state_dim = target_state_dim

    def forward(self, target_state):
        source_state = self.model(target_state.float()) * 0.5
        return source_state
#==========================================================================
# smodice =================================================================
class TanhNormalPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_sizes=(256,256), action_space=None,
                 mean_range=(-7.24, 7.24), logstd_range=(-5., 2.), eps=1e-6):
        super(TanhNormalPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        self.mean_linear = nn.Linear(hidden_sizes[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        
        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def forward(self, inputs, step_type=(), network_state=(), training=False):
        inputs = torch.cat(inputs, 1)
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        logstd = self.log_std_linear(x)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd)
        pretanh_action_dist = distributions.Normal(mean, std)
        pretanh_action = pretanh_action_dist.rsample()
        action = torch.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return (action, pretanh_action, log_prob, pretanh_log_prob, pretanh_action_dist), network_state

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = torch.tanh(pretanh_action)
        else:
            pretanh_action = atanh(torch.clamp(action, -1 + self.eps, 1 - self.eps))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - torch.log(1 - action ** 2 + self.eps)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob, pretanh_log_prob

    def deterministic_action(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        action = torch.tanh(mean)
        return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(TanhNormalPolicy, self).to(device)
    

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_sizes):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = nn.Linear(hidden_sizes[1], 1)

        self.apply(weights_init_)

    def forward(self, state):
        state = torch.cat(state, 1)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x, None
    
# =========================================================================
# gwil ====================================================================
class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
        
        
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
    

class TanhTransform(distributions.transforms.Transform):
    domain = distributions.constraints.real
    codomain = distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(distributions.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = distributions.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)

# =========================================================================
# function of dataasets ===================================================
def load_d4rl_data(dirname, env_id, dataname, num_trajectories, start_idx=0, dtype=np.float32):
    MAX_EPISODE_STEPS = 1000

    original_env_id = env_id
    filename = ''
    filepath = ''
    env = None
    if env_id in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Ant-v2']:
        env_id = env_id.split('-v2')[0].lower()
    filename = f'{env_id}_{dataname}'
    filepath = os.path.join(dirname, filename + '.hdf5')
    # if not exists
    if (not os.path.exists(filepath)):
        os.makedirs(dirname, exist_ok=True)
        # Download the dataset
        remote_url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{filename}.hdf5'
        print(f'Download dataset from {remote_url} into {filepath} ...')
        request.urlretrieve(remote_url, filepath)
        print(f'Done!')

    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys
    
    dataset_file = h5py.File(filepath, 'r')
    dataset_keys = KEYS
    use_timeouts = False
    use_next_obs = False
    if 'timeouts' in get_keys(dataset_file):
        if 'timeouts' not in dataset_keys:
            dataset_keys.append('timeouts')
        use_timeouts = True
    dataset = {k: dataset_file[k][:] for k in dataset_keys}
    dataset_file.close()

    N = dataset['observations'].shape[0]
    init_obs_, init_action_, obs_, action_, next_obs_, rew_, done_ = [], [], [], [], [], [], []
    episode_steps = 0
    num_episodes = 0
    total_reward = 0
    total_len = 0
    
    for i in range(N - 1):
        if env_id == 'ant':
            obs = dataset['observations'][i][:27]
            if use_next_obs:
                next_obs = dataset['next_observations'][i][:27]
            else:
                next_obs = dataset['observations'][i + 1][:27]
        else:
            obs = dataset['observations'][i]
            if use_next_obs:
                next_obs = dataset['next_observations'][i]
            else:
                next_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        total_reward += dataset['rewards'][i]
        total_len += 1
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            is_final_timestep = dataset['timeouts'][i]
        else:
            is_final_timestep = (episode_steps == MAX_EPISODE_STEPS - 1)

        if is_final_timestep:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break
            continue

        if num_episodes >= start_idx:
            if episode_steps == 0:
                init_obs_.append(obs)
            obs_.append(obs)
            next_obs_.append(next_obs)
            action_.append(action)
            done_.append(done_bool)

        episode_steps += 1
        if done_bool:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break
    
    env = gym.make(original_env_id)
    if env.action_space.dtype == int:
        action_ = np.eye(env.action_space.n)[np.array(action_, dtype=np.int)]  # integer to one-hot encoding

    avg_reward = total_reward / num_episodes
    avg_len = total_len / num_episodes

    print(f'{num_trajectories} trajectories are sampled, average reward: {avg_reward}, average length: {avg_len}')
    return np.array(init_obs_, dtype=dtype), np.array(obs_, dtype=dtype), np.array(action_, dtype=dtype), np.array(
        next_obs_, dtype=dtype), np.array(done_)


def sample_demonstrations(env_id, xml_path=None, num_trajectories=1, load_path=None, max_episode_steps=1000, difficulty='random', pair_num = 0, dtype=np.float32, env_robot=None):
    # for saved file for datasets
    if load_path:
        data = np.load(load_path)
        num = max_episode_steps
        if pair_num != 0:
            num = pair_num
        return data['init_obs'][:num_trajectories], data['obs'][:num_trajectories*num], data['action'][:num_trajectories*num], data['next_obs'][:num_trajectories*num], data['done'][:num_trajectories*num]
    
    # for collection of random trajectories
    if env_robot:
        controller_config = load_controller_config(default_controller='JOINT_VELOCITY')
        env_suite = suite.make(
            env_id,
            robots=env_robot,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_object_obs=True,
            use_camera_obs=False,
            reward_shaping=True,
            horizon=500,
        )
        keys = ["object-state"]
        for idx in range(len(env_suite.robots)):
            keys.append(f"robot{idx}_proprio-state")
        env = GymWrapper(env_suite, keys=keys)
    else:
        env = gymnasium.make(env_id, xml_file=xml_path)

    env.seed = 0
    
    init_obs_, obs_, action_, next_obs_, done_ = [], [], [], [], []
    
    num_episodes = 0
    rewards = 0
    total_len = 0
    
    while num_episodes < num_trajectories:
        print("episodes: ", num_episodes)
        obs = env.reset()
        if 'ant' in env_id.lower():
            obs = obs[0][:31]
        else:
            obs = obs[0]
        
        init_obs_.append(obs)
        episode_steps = 0
        episode_return = 0
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info, _ = env.step(action)
            if 'ant' in env_id.lower():
                next_obs = next_obs[:31]
            else:
                next_obs = next_obs
            rewards += reward
            episode_return += reward
            
            obs_.append(obs)
            action_.append(action)
            next_obs_.append(next_obs)
            done_.append(done)
            
            obs = next_obs
            episode_steps += 1

            total_len += 1
            if done or (episode_steps >= max_episode_steps):
                num_episodes += 1
                break

    avg_return = rewards / num_episodes
    avg_len = total_len / num_episodes

    if env_robot:
        np.savez(f'dataset/{env_id}_{env_robot}_{difficulty}_{num_trajectories}', 
             init_obs=np.array(init_obs_, dtype=dtype), 
             obs=np.array(obs_, dtype=dtype), 
             action=np.array(action_, dtype=dtype), 
             next_obs=np.array(next_obs_, dtype=dtype), 
             done=np.array(done_, dtype=dtype))
    else:
        np.savez(f'dataset/target_{env_id}_{difficulty}_{num_trajectories}', 
             init_obs=np.array(init_obs_, dtype=dtype), 
             obs=np.array(obs_, dtype=dtype), 
             action=np.array(action_, dtype=dtype), 
             next_obs=np.array(next_obs_, dtype=dtype), 
             done=np.array(done_, dtype=dtype))

    print(f'{num_trajectories} trajectories sampled ({difficulty} difficulty), average return: {avg_return}., average length: {avg_len}')
    return (np.array(init_obs_, dtype=dtype), np.array(obs_, dtype=dtype),
            np.array(action_, dtype=dtype), np.array(next_obs_, dtype=dtype),
            np.array(done_, dtype=dtype))


def add_absorbing_states(expert_states, expert_actions, expert_next_states,
                         expert_dones, env, dtype=np.float32):
    """Adds absorbing states to trajectories.
    Args:
      expert_states: A numpy array with expert states.
      expert_actions: A numpy array with expert states.
      expert_next_states: A numpy array with expert states.
      expert_dones: A numpy array with expert states.
      env: A gym environment.
    Returns:
        Numpy arrays that contain states, actions, next_states and dones.
    """

    # First add 0 indicator to all non-absorbing states.
    expert_states = np.pad(expert_states, ((0, 0), (0, 1)), mode='constant')
    expert_next_states = np.pad(
        expert_next_states, ((0, 0), (0, 1)), mode='constant')

    expert_states = [x for x in expert_states]
    expert_next_states = [x for x in expert_next_states]
    expert_actions = [x for x in expert_actions]
    expert_dones = [x for x in expert_dones]

    # Add absorbing states.
    i = 0
    current_len = 0
    while i < len(expert_states):
        current_len += 1
        if expert_dones[i] and current_len < env._max_episode_steps:  # pylint: disable=protected-access
            current_len = 0
            expert_states.insert(i + 1, env.get_absorbing_state())
            expert_next_states[i] = env.get_absorbing_state()
            expert_next_states.insert(i + 1, env.get_absorbing_state())
            action_dim = env.action_space.n if env.action_space.dtype == int else env.action_space.shape[0]
            expert_actions.insert(i + 1, np.zeros((action_dim,), dtype=dtype))
            expert_dones[i] = 0.0
            expert_dones.insert(i + 1, 1.0)
            i += 1
        i += 1

    expert_states = np.stack(expert_states)
    expert_next_states = np.stack(expert_next_states)
    expert_actions = np.stack(expert_actions)
    expert_dones = np.stack(expert_dones)

    return expert_states.astype(dtype), expert_actions.astype(dtype), expert_next_states.astype(dtype), expert_dones.astype(dtype)

# =========================================================================