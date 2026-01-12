from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np
np.bool = np.bool_
import hydra
import torch
from tqdm import tqdm
from tqdm import trange
import wrappers
import math
import agent.demodice as demodice
import agent.avatar_dice_cross_fitting as avatar_dice
import utils.utils as utils
import time
import pickle
from torch.utils.tensorboard import SummaryWriter
import ast

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
        eval_env = wrappers.create_il_env(env_name=env_id+'-v3', shift=shift_env, scale=scale_env, normalized_box_actions=False, xml_path=config['xml_path'])
    elif env_robot:
        eval_env = wrappers.create_il_env(env_name=env_id, shift=shift_env, scale=scale_env, normalized_box_actions=False, robot=config['env_robot'])
    else:
        eval_env = wrappers.create_il_env(env_id+'-v2', shift=shift_env, scale=scale_env, normalized_box_actions=False)

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
        imperfect_dataset_names, imperfect_num_trajs = info = ast.literal_eval(config['imperfect_dataset_default_info'])
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
            start_idx = expert_num_traj if (expert_dataset_name == imperfect_dataset_name) else 0

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
                                                                                            difficulty=imperfect_dataset_name[:6], 
                                                                                            dtype=np.float32)
            elif env_robot:
                (initial_states, states, actions, next_states, dones) = utils.sample_demonstrations(env_id=env_id, 
                                                                                            num_trajectories=imperfect_num_traj, 
                                                                                            load_path=load_paths[index], 
                                                                                            max_episode_steps=500, 
                                                                                            difficulty=imperfect_dataset_name[:6], 
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


    imperfect_init_states = np.concatenate(imperfect_init_states).astype(np.float32)
    imperfect_states = np.concatenate(imperfect_states).astype(np.float32)
    imperfect_actions = np.concatenate(imperfect_actions).astype(np.float32)
    imperfect_next_states = np.concatenate(imperfect_next_states).astype(np.float32)
    imperfect_dones = np.concatenate(imperfect_dones).astype(np.float32)

    union_init_states = np.concatenate([imperfect_init_states, expert_initial_states]).astype(np.float32)
    union_states = np.concatenate([imperfect_states, expert_states]).astype(np.float32)
    union_actions = np.concatenate([imperfect_actions, expert_actions]).astype(np.float32)
    union_next_states = np.concatenate([imperfect_next_states, expert_next_states]).astype(np.float32)
    union_dones = np.concatenate([imperfect_dones, expert_dones]).astype(np.float32)

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
        env = wrappers.create_il_env(env_name=env_id+'-v2', shift=shift_env, scale=scale_env, normalized_box_actions=False)
    elif xml_path:
        env = wrappers.create_il_env(env_name=env_id+'-v3', shift=shift_env, scale=scale_env, normalized_box_actions=False, xml_path=xml_path)
    elif env_robot:
        env = wrappers.create_il_env(env_name=env_id, shift=shift_env, scale=scale_env, normalized_box_actions=False, robot=env_robot)

    if config['using_absorbing']:
        # using absorbing state
        union_init_states = np.c_[union_init_states, np.zeros(len(union_init_states), dtype=np.float32)]
        (expert_states, expert_actions, expert_next_states,
         expert_dones) = utils.add_absorbing_states(expert_states, expert_actions, expert_next_states, expert_dones, env)
        (union_states, union_actions, union_next_states,
         union_dones) = utils.add_absorbing_states(union_states, union_actions, union_next_states, union_dones, env)
    else:
        # ignore absorbing state
        union_init_states = np.c_[union_init_states, np.zeros(len(union_init_states), dtype=np.float32)]
        expert_states = np.c_[expert_states, np.zeros(len(expert_states), dtype=np.float32)]
        expert_next_states = np.c_[expert_next_states, np.zeros(len(expert_next_states), dtype=np.float32)]
        union_states = np.c_[union_states, np.zeros(len(union_states), dtype=np.float32)]
        union_next_states = np.c_[union_next_states, np.zeros(len(union_next_states), dtype=np.float32)]

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

    if xml_path:
        src_env = wrappers.create_il_env(env_id+'-v2', shift=shift_env, scale=scale_env, normalized_box_actions=False, robot=env_robot)
    else:
        src_env = wrappers.create_il_env(env_name=env_id, shift=shift_env, scale=scale_env, normalized_box_actions=False, robot=config['src_env_robot'])
    if xml_path and ('ant' in env_id.lower()):
        src_obs_dim = 28
    else:
        src_obs_dim = src_env.observation_space.shape[0]

    src_imitator = demodice.DemoDICE(
        src_obs_dim,
        src_env.action_space.shape[0],
        is_discrete_action,
        config=config)
    
    src_imitator.load(config['pretrained_model_path'])  

    imitator = avatar_dice.Avatar(
        observation_dim,
        action_dim,
        is_discrete_action,
        src_imitator.q_function,
        src_imitator.cost,
        src_obs_dim,
        src_env.action_space.shape[0],
        config=config)
    

    print("Save interval :", config['save_interval'])
    # checkpoint dir
    checkpoint_dir = f"checkpoint_imitator/{algorithm}/{env_id}/" \
                     f"{expert_dataset_name}_{expert_num_traj}_" \
                     f"{imperfect_dataset_names}_{imperfect_num_trajs}/{tb_path}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filepath = f"{checkpoint_dir}/0000"
    if config['resume'] and os.path.exists(checkpoint_filepath):
        # Load checkpoint.s
        imitator.init_dummy(observation_dim, action_dim)
        checkpoint_data = imitator.load(checkpoint_filepath)
        training_info = checkpoint_data['training_info']
        training_info['iteration'] += 1
        print(f"Checkpoint '{checkpoint_filepath}' is resumed")
    else:
        print(f"No checkpoint is found: {checkpoint_filepath}")
        training_info = {
            'iteration': 0,
            'logs': [],
        }
    print(config['save_interval'])
    total_iterations = config['total_iterations'] + 1

    # make data tensor
    union_init_states_ = torch.from_numpy(union_init_states).float().to(device)
    expert_states_ = torch.from_numpy(expert_states).float().to(device)
    expert_actions_ = torch.from_numpy(expert_actions).float().to(device)
    expert_next_states_ = torch.from_numpy(expert_next_states).float().to(device)
    union_states_ = torch.from_numpy(union_states).float().to(device)
    union_actions_ = torch.from_numpy(union_actions).float().to(device)
    union_next_states_ = torch.from_numpy(union_next_states).float().to(device)
    union_dones_ = torch.from_numpy(union_dones).float().to(device)

    # Start training
    start_time = time.time()

    info_dict = {}

    with tqdm(total=total_iterations+1, initial=training_info['iteration'], desc='',
              disable=os.environ.get("DISABLE_TQDM", False), ncols=70) as pbar:
        while training_info['iteration'] <= total_iterations:
            union_init_indices = np.random.randint(0, len(union_init_states_)/2, size=batch_size)
            expert_indices = np.random.randint(0, len(expert_states_), size=batch_size)
            union_indices_half = np.random.randint(0, len(union_states_)/2, size=batch_size//2)

            union_indices = np.random.randint(0, len(union_states_), size=batch_size//2)

            info_dict = imitator.update(
                union_init_states_,
                expert_states_[expert_indices],
                expert_actions_[expert_indices],
                expert_next_states_[expert_indices],
                union_states_,
                union_actions_,
                union_next_states_,
                union_init_indices,
                union_indices_half,
                union_indices,
                training_info['iteration'],
                int(len(union_init_states_)/2),
                int(len(union_states_)/2)
            )

            if training_info['iteration'] % config['log_interval'] == 0:
                average_returns, evaluation_timesteps = evaluate_d4rl(config, env_id, imitator, shift_env, scale_env)

                writer.add_scalar('Test average return', average_returns, training_info['iteration'])
                info_dict.update({'eval': average_returns})
                print(f'Eval: ave returns=d: {average_returns}'
                      f' ave episode length={evaluation_timesteps}'
                      f' / elapsed_time={time.time() - start_time} ({training_info["iteration"] / (time.time() - start_time)} it/sec)')
                print('=========================')
                for key, val in info_dict.items():
                    if algorithm == 'smodice':
                        print(f'{key:25}: {val.item():8.3f}')
                    else:
                        print(f'{key:25}: {val:8.3f}')
                print('=========================')

                training_info['logs'].append({'step': training_info['iteration'], 'log': info_dict})
                print(f'timestep {training_info["iteration"]} - log update...')
                print('Done!', flush=True)

            # if training_info['iteration'] % 10000 == 0:
                # writer.add_scalar('Adaptive weight/c1', imitator.c1, training_info['iteration'])
                # writer.add_scalar('Adaptive weight/c2', imitator.c2_smooth, training_info['iteration'])
                # writer.add_scalar('Time weight decay', imitator.c2_smooth**config['power_decay_weight'] / (imitator.c1**config['power_decay_weight'] + imitator.c2_smooth**config['power_decay_weight'] + 1e-6), training_info['iteration'])
                # writer.add_scalar('LMAP', info_dict['mapping_loss'], training_info['iteration'])

            training_info['iteration'] += 1
            pbar.update(1)



if __name__ == "__main__":
    from config.config import get_parser

    # configurations
    args = get_parser().parse_args()
    config = vars(args)
    config['dataset_file_names'] = ast.literal_eval(config['dataset_file_names'])

    print("Start running")
    run(config)
