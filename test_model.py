from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import ast
from torch.utils.tensorboard import SummaryWriter
import test
import pickle
import time
from utils import utils
# from config.lfd_default_config import get_parser
from config.config import get_parser
from agent import avatar_dice, demodice
# import demodice_pytorch as demodice
import copy
import math
import wrappers
from tqdm import tqdm
import torch
import os
import random
import numpy as np
# np.bool = np.bool_
# import tensorflow as tf
# import robosuite as suite


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate_d4rl(config, env_id, actor, shift_env, scale_env, num_seed=5, num_episodes=10):
    """Evaluates the policy.
    Args:
        actor: A policy to evaluate
        env: Environment to evaluate the policy on
        train_env_id: train_env_id to compute normalized score
        num_episodes: A number of episodes to average the policy on
    Returns:
        Averaged reward and a total number of steps.
    """
    total_timesteps = 0
    total_returns = 0
    seeds = 10
    env_is_gym = config['env_is_gym']
    xml_path = config['xml_path']
    env_robot = config['env_robot']

    if xml_path:
        eval_env = wrappers.create_il_env(env_name=env_id+'-v3', shift=shift_env,
                                          scale=scale_env, normalized_box_actions=False, xml_path=config['xml_path'])
    elif env_robot:
        eval_env = wrappers.create_il_env(
            env_name=env_id, shift=shift_env, scale=scale_env, normalized_box_actions=False, robot=config['env_robot'])
    else:
        eval_env = wrappers.create_il_env(
            env_id+'-v2', shift=shift_env, scale=scale_env, normalized_box_actions=False)

    for _ in range(num_seed):
        for _ in range(num_episodes):
            if env_is_gym:
                state = eval_env.reset()
            else:
                state = eval_env.reset(seed=seeds)[0]

            done = False
            length = 0
            while not done:
                if 'ant' in env_id.lower():
                    if env_is_gym:
                        state = np.concatenate((state[:27], [0.]), -1)
                    else:
                        state = np.concatenate((state[:31], [0.]), -1)

                if config['algorithm'] == 'smodice':
                    actions = actor.step(
                        (np.array([state])).astype(np.float32))
                    action = actions[0][0].numpy()
                elif config['algorithm'] == 'gwil':
                    with utils.eval_mode(actor):
                        action = actor.act(state, sample=False)
                elif config['algorithm'] == 'igdf':
                    with utils.eval_mode(actor):
                        action = actor.choose_action(state, sample=False)
                else:
                    action = actor.step(state)[0].numpy()

                if env_is_gym:
                    next_state, reward, done, _ = eval_env.step(action)
                else:
                    next_state, reward, done, _, _ = eval_env.step(action)

                total_returns += reward
                total_timesteps += 1
                state = next_state
                length += 1

                if length > 1000:
                    break
        seeds += 1

    mean_score = total_returns / (num_episodes*num_seed)
    mean_timesteps = total_timesteps / (num_episodes*num_seed)

    return mean_score, mean_timesteps


def run(config):

    seed = config['seed']
    seed_torch(seed)
    np.random.seed(seed)
    random.seed(seed)

    env_id = config['env_id']
    tb_path = config['tb_path']
    env_is_gym = config['env_is_gym']
    xml_path = config['xml_path']
    env_robot = config['env_robot']
    dataset_file_names = config['dataset_file_names']
    load_hdf5_dataset = config['load_hdf5_dataset']
    algorithm = config['algorithm']
    batch_size = config['batch_size']

    writer = SummaryWriter(tb_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # expert data info
    expert_dataset_name = config['expert_dataset_name']
    expert_num_traj = config['expert_num_traj']
    # imperfect data info
    imperfect_dataset_names = config['imperfect_dataset_names']
    imperfect_num_trajs = config['imperfect_num_trajs']
    if len(imperfect_dataset_names) == 0:
        imperfect_dataset_names, imperfect_num_trajs = config['imperfect_dataset_default_info']
    assert len(imperfect_dataset_names) == len(imperfect_num_trajs)

    dataset_dir = config['dataset_dir']
    if not load_hdf5_dataset:
        dataset_path = os.path.join(dataset_dir, dataset_file_names[0])
    traj_nums = []
    traj_lens = []
    if load_hdf5_dataset:
        (expert_initial_states, expert_states, expert_actions, expert_next_states, expert_dones) = utils.load_d4rl_data(
            dataset_dir, env_id+'-v2', expert_dataset_name, expert_num_traj, start_idx=0)
    elif xml_path:
        (expert_initial_states, expert_states, expert_actions, expert_next_states, expert_dones) = utils.sample_demonstrations(
            env_id+'-v3', xml_path, expert_num_traj, dataset_path, difficulty='expert', dtype=np.float32)
    elif env_robot:
        (expert_initial_states, expert_states, expert_actions, expert_next_states, expert_dones) = utils.sample_demonstrations(
            env_id=env_id, num_trajectories=expert_num_traj, load_path=dataset_path, max_episode_steps=500, difficulty='expert', dtype=np.float32, env_robot=env_robot)

    traj_nums.append(expert_num_traj)
    traj_lens.append(math.ceil(expert_states.shape[0]/expert_num_traj))

    # load non-expert dataset
    imperfect_init_states, imperfect_states, imperfect_actions, imperfect_next_states, imperfect_dones = [], [], [], [], []
    if len(imperfect_dataset_names) > 0:
        if not load_hdf5_dataset:
            index = 0
            load_paths = dataset_file_names[-2:]
            for i in range(len(load_paths)):
                if load_paths[i] == None:
                    load_paths[i] = None
                else:
                    load_paths[i] = os.path.join(dataset_dir, load_paths[i])

        for imperfect_datatype_idx, (imperfect_dataset_name, imperfect_num_traj) in enumerate(
                zip(imperfect_dataset_names, imperfect_num_trajs)):
            start_idx = expert_num_traj if (
                expert_dataset_name == imperfect_dataset_name) else 0

            if load_hdf5_dataset:
                (initial_states, states, actions, next_states, dones) = utils.load_d4rl_data(dataset_dir, env_id+'-v2',
                                                                                             imperfect_dataset_name,
                                                                                             imperfect_num_traj,
                                                                                             start_idx=start_idx)
            elif xml_path:
                (initial_states, states, actions, next_states, dones) = utils.sample_demonstrations(env_id+'-v3',
                                                                                                    xml_path,
                                                                                                    imperfect_num_traj,
                                                                                                    load_paths[index],
                                                                                                    difficulty=imperfect_dataset_name[
                                                                                                        :6],
                                                                                                    dtype=np.float32)
            elif env_robot:
                (initial_states, states, actions, next_states, dones) = utils.sample_demonstrations(env_id=env_id,
                                                                                                    num_trajectories=imperfect_num_traj,
                                                                                                    load_path=load_paths[index],
                                                                                                    max_episode_steps=500,
                                                                                                    difficulty=imperfect_dataset_name[
                                                                                                        :6],
                                                                                                    dtype=np.float32,
                                                                                                    env_robot=env_robot)
            if not load_hdf5_dataset:
                index += 1

            imperfect_init_states.append(initial_states)
            imperfect_states.append(states)
            imperfect_actions.append(actions)
            imperfect_next_states.append(next_states)
            imperfect_dones.append(dones)

            traj_nums.append(imperfect_num_traj)
            traj_lens.append(math.ceil(states.shape[0]/imperfect_num_traj))

    imperfect_init_states = np.concatenate(
        imperfect_init_states).astype(np.float32)
    imperfect_states = np.concatenate(imperfect_states).astype(np.float32)
    imperfect_actions = np.concatenate(imperfect_actions).astype(np.float32)
    imperfect_next_states = np.concatenate(
        imperfect_next_states).astype(np.float32)
    imperfect_dones = np.concatenate(imperfect_dones).astype(np.float32)

    union_init_states = np.concatenate(
        [imperfect_init_states, expert_initial_states]).astype(np.float32)
    union_states = np.concatenate(
        [imperfect_states, expert_states]).astype(np.float32)
    union_actions = np.concatenate(
        [imperfect_actions, expert_actions]).astype(np.float32)
    union_next_states = np.concatenate(
        [imperfect_next_states, expert_next_states]).astype(np.float32)
    union_dones = np.concatenate(
        [imperfect_dones, expert_dones]).astype(np.float32)

    print('# of expert demonstraions: {}'.format(expert_states.shape[0]))
    print('# of imperfect demonstraions: {}'.format(imperfect_states.shape[0]))
    # normalize
    shift = -np.mean(imperfect_states, 0)
    scale = 1.0 / (np.std(imperfect_states, 0) + 1e-3)
    union_init_states = (union_init_states + shift) * scale
    expert_states = (expert_states + shift) * scale
    expert_next_states = (expert_next_states + shift) * scale
    union_states = (union_states + shift) * scale
    union_next_states = (union_next_states + shift) * scale

    # environment setting
    if 'ant' in env_id.lower():
        if load_hdf5_dataset:
            shift_env = np.concatenate((shift, np.zeros(84)))
            scale_env = np.concatenate((scale, np.ones(84)))
        else:
            shift_env = np.concatenate((shift, np.zeros(102)))
            scale_env = np.concatenate((scale, np.ones(102)))
    else:
        shift_env = shift
        scale_env = scale

    if load_hdf5_dataset:
        env = wrappers.create_il_env(
            env_name=env_id+'-v2', shift=shift_env, scale=scale_env, normalized_box_actions=False)
    elif xml_path:
        env = wrappers.create_il_env(env_name=env_id+'-v3', shift=shift_env,
                                     scale=scale_env, normalized_box_actions=False, xml_path=xml_path)
    elif env_robot:
        env = wrappers.create_il_env(env_name=env_id, shift=shift_env,
                                     scale=scale_env, normalized_box_actions=False, robot=env_robot)

    if config['using_absorbing']:
        # using absorbing state
        union_init_states = np.c_[union_init_states, np.zeros(
            len(union_init_states), dtype=np.float32)]
        (expert_states, expert_actions, expert_next_states,
         expert_dones) = utils.add_absorbing_states(expert_states, expert_actions, expert_next_states, expert_dones, env)
        (union_states, union_actions, union_next_states,
         union_dones) = utils.add_absorbing_states(union_states, union_actions, union_next_states, union_dones, env)
    else:
        # ignore absorbing state
        union_init_states = np.c_[union_init_states, np.zeros(
            len(union_init_states), dtype=np.float32)]
        expert_states = np.c_[expert_states, np.zeros(
            len(expert_states), dtype=np.float32)]
        expert_next_states = np.c_[expert_next_states, np.zeros(
            len(expert_next_states), dtype=np.float32)]
        union_states = np.c_[union_states, np.zeros(
            len(union_states), dtype=np.float32)]
        union_next_states = np.c_[union_next_states, np.zeros(
            len(union_next_states), dtype=np.float32)]

    observation_dim = env.observation_space.shape[0]
    if xml_path:
        if 'ant' in env_id.lower():
            if load_hdf5_dataset:
                observation_dim = 28
            else:
                observation_dim = 32
        elif 'cheetah' in env_id.lower():
            observation_dim = 24
        elif 'hopper' in env_id.lower():
            observation_dim = 14

    # Create imitator
    is_discrete_action = env.action_space.dtype == int
    action_dim = env.action_space.n if is_discrete_action else env.action_space.shape[0]

    # Create imitator
    if xml_path:
        src_env = wrappers.create_il_env(
            env_id+'-v2', shift=shift_env, scale=scale_env, normalized_box_actions=False, robot=env_robot)
    else:
        src_env = wrappers.create_il_env(
            env_name=env_id, shift=shift_env, scale=scale_env, normalized_box_actions=False, robot=config['src_env_robot'])
    if xml_path and ('ant' in env_id.lower()):
        src_obs_dim = 28
    else:
        src_obs_dim = src_env.observation_space.shape[0]

    src_imitator = demodice.DemoDICE(
        src_obs_dim,
        src_env.action_space.shape[0],
        is_discrete_action,
        config=config)
    imitator = avatar_dice.Avatar(
        observation_dim,
        action_dim,
        is_discrete_action,
        src_imitator.q_function,
        src_imitator.cost,
        src_obs_dim,
        src_env.action_space.shape[0],
        config=config)

    checkpoint_path = './checkpoint_imitator/demodice/Hopper-v2/expert-v2_1_[\'expert-v2\', \'random-v2\']_[20, 50]/transfer_torch_ver1_lr_5e-5_traj_1_20_50_adaptive_decay_pow1_seed0_l2_1e-2'
    checkpoint = os.path.join(checkpoint_path, "500000.pickle")
    imitator.load(checkpoint)
    average_returns, evaluation_timesteps = evaluate_d4rl(
        config, env_id, imitator, shift_env, scale_env)
    writer.add_scalar('Test average return', average_returns, 500000)
    print('average return: ', average_returns)


if __name__ == "__main__":

    # configurations
    args = get_parser().parse_args()
    config = vars(args)
    config['dataset_file_names'] = ast.literal_eval(
        config['dataset_file_names'])

    print("Start running")
    run(config)
