from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from omegaconf import DictConfig, OmegaConf
from utils.contrastiveoi import ContrastiveInfo
from utils.discriminator import Discriminator_SA
import ast
from torch.utils.tensorboard import SummaryWriter
import pickle
import time
from utils.sac import SAC
from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
import agent.igdf as igdf
import agent.gwil as gwil
import agent.smodice as smodice
import agent.demodice as demodice
import math
import wrappers_gym as wrappers
from tqdm import trange
from tqdm import tqdm
import torch
import hydra
import os
import random
import numpy as np

np.bool = np.bool_


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_args(cfg: DictConfig):
    # cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg


def evaluate_d4rl(
    config, env_id, actor, shift_env, scale_env, num_seed=5, num_episodes=10
):
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
    env_is_gym = config["env_is_gym"]
    xml_path = config["xml_path"]
    env_robot = config["env_robot"]

    eval_env = wrappers.create_il_env(
        env_id,
        shift=shift_env,
        scale=scale_env,
        normalized_box_actions=False,
    )

    for _ in range(num_seed):
        for _ in range(num_episodes):
            state = eval_env.reset(seed=seeds)

            done = False
            length = 0
            while not done:
                if "ant" in env_id.lower():
                    state = np.concatenate((state[:27], [0.0]), -1)

                if config["algorithm"] == "smodice":
                    actions = actor.step((np.array([state])).astype(np.float32))
                    action = actions[0][0].numpy()
                elif config["algorithm"] == "gwil":
                    with utils.eval_mode(actor):
                        action = actor.act(state, sample=False)
                elif config["algorithm"] == "igdf":
                    with utils.eval_mode(actor):
                        action = actor.choose_action(state, sample=False)
                else:
                    action = actor.step(state)[0].numpy()

                next_state, reward, done, _ = eval_env.step(action)

                total_returns += reward
                total_timesteps += 1
                state = next_state
                length += 1

                if length > 1000:
                    break
        seeds += 1

    mean_score = total_returns / (num_episodes * num_seed)
    mean_timesteps = total_timesteps / (num_episodes * num_seed)

    return mean_score, mean_timesteps


def run(config, cfg):
    sac_args = get_args(cfg)

    seed = config["seed"]
    seed_torch(seed)
    np.random.seed(seed)
    random.seed(seed)

    src_only = config["src_only"]
    fixed_beta = config["fixed_beta"]

    env_id = config["env_id"]
    tb_path = config["tb_path"]
    env_is_gym = config["env_is_gym"]
    xml_path = config["xml_path"]
    env_robot = config["env_robot"]
    dataset_file_names = config["dataset_file_names"]
    load_hdf5_dataset = config["load_hdf5_dataset"]
    algorithm = config["algorithm"]
    batch_size = config["batch_size"]

    writer = SummaryWriter(tb_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # expert data info
    expert_dataset_name = config["expert_dataset_name"]
    expert_num_traj = config["expert_num_traj"]
    # imperfect data info
    imperfect_dataset_names = config["imperfect_dataset_names"]
    imperfect_num_trajs = config["imperfect_num_trajs"]
    if len(imperfect_dataset_names) == 0:
        imperfect_dataset_names, imperfect_num_trajs = info = ast.literal_eval(
            config["imperfect_dataset_default_info"]
        )
    assert len(imperfect_dataset_names) == len(imperfect_num_trajs)

    dataset_dir = config["dataset_dir"]
    if not load_hdf5_dataset:
        dataset_path = os.path.join(dataset_dir, dataset_file_names[0])
    traj_nums = []
    traj_lens = []
    if load_hdf5_dataset:
        (
            expert_initial_states,
            expert_states,
            expert_actions,
            expert_next_states,
            expert_dones,
        ) = utils.load_d4rl_data(
            dataset_dir,
            env_id + "-v2",
            expert_dataset_name,
            expert_num_traj,
            start_idx=0,
        )
    else:
        (
            expert_initial_states,
            expert_states,
            expert_actions,
            expert_next_states,
            expert_dones,
        ) = utils.sample_demonstrations(
            env_id + "-v3",
            xml_path,
            expert_num_traj,
            dataset_path,
            difficulty="expert",
            dtype=np.float32,
        )

    traj_nums.append(expert_num_traj)
    traj_lens.append(math.ceil(expert_states.shape[0] / expert_num_traj))

    # load non-expert dataset
    (
        imperfect_init_states,
        imperfect_states,
        imperfect_actions,
        imperfect_next_states,
        imperfect_dones,
    ) = ([], [], [], [], [])
    if len(imperfect_dataset_names) > 0:
        if not load_hdf5_dataset:
            index = 0
            load_paths = dataset_file_names[-2:]
            for i in range(len(load_paths)):
                if load_paths[i] is not None:
                    load_paths[i] = os.path.join(dataset_dir, load_paths[i])
                else:
                    load_paths[i] = None

        for imperfect_datatype_idx, (
            imperfect_dataset_name,
            imperfect_num_traj,
        ) in enumerate(zip(imperfect_dataset_names, imperfect_num_trajs)):
            start_idx = (
                expert_num_traj
                if (expert_dataset_name == imperfect_dataset_name)
                else 0
            )

            if load_hdf5_dataset:
                (initial_states, states, actions, next_states, dones) = (
                    utils.load_d4rl_data(
                        dataset_dir,
                        env_id + "-v2",
                        imperfect_dataset_name,
                        imperfect_num_traj,
                        start_idx=start_idx,
                    )
                )
            else:
                (initial_states, states, actions, next_states, dones) = (
                    utils.sample_demonstrations(
                        env_id + "-v3",
                        xml_path,
                        imperfect_num_traj,
                        load_paths[index],
                        difficulty=imperfect_dataset_name[:6],
                        dtype=np.float32,
                    )
                )
            if not load_hdf5_dataset:
                index += 1

            imperfect_init_states.append(initial_states)
            imperfect_states.append(states)
            imperfect_actions.append(actions)
            imperfect_next_states.append(next_states)
            imperfect_dones.append(dones)

            traj_nums.append(imperfect_num_traj)
            traj_lens.append(math.ceil(states.shape[0] / imperfect_num_traj))

    imperfect_init_states = np.concatenate(imperfect_init_states).astype(np.float32)
    imperfect_states = np.concatenate(imperfect_states).astype(np.float32)
    imperfect_actions = np.concatenate(imperfect_actions).astype(np.float32)
    imperfect_next_states = np.concatenate(imperfect_next_states).astype(np.float32)
    imperfect_dones = np.concatenate(imperfect_dones).astype(np.float32)

    union_init_states = np.concatenate(
        [imperfect_init_states, expert_initial_states]
    ).astype(np.float32)
    union_states = np.concatenate([imperfect_states, expert_states]).astype(np.float32)
    union_actions = np.concatenate([imperfect_actions, expert_actions]).astype(
        np.float32
    )
    union_next_states = np.concatenate(
        [imperfect_next_states, expert_next_states]
    ).astype(np.float32)
    union_dones = np.concatenate([imperfect_dones, expert_dones]).astype(np.float32)

    print("# of expert demonstraions: {}".format(expert_states.shape[0]))
    print("# of imperfect demonstraions: {}".format(imperfect_states.shape[0]))
    # normalize
    shift = -np.mean(imperfect_states, 0)
    scale = 1.0 / (np.std(imperfect_states, 0) + 1e-3)
    union_init_states = (union_init_states + shift) * scale
    expert_states = (expert_states + shift) * scale
    expert_next_states = (expert_next_states + shift) * scale
    union_states = (union_states + shift) * scale
    union_next_states = (union_next_states + shift) * scale

    # environment setting
    if "ant" in env_id.lower():
        shift_env = np.concatenate((shift, np.zeros(84)))
        scale_env = np.concatenate((scale, np.ones(84)))
    else:
        shift_env = shift
        scale_env = scale

    env = wrappers.create_il_env(
        env_name=env_id,
        shift=shift_env,
        scale=scale_env,
        normalized_box_actions=False,
        robot=env_robot,
    )

    if config["using_absorbing"]:
        # using absorbing state
        union_init_states = np.c_[
            union_init_states, np.zeros(len(union_init_states), dtype=np.float32)
        ]
        (expert_states, expert_actions, expert_next_states, expert_dones) = (
            utils.add_absorbing_states(
                expert_states, expert_actions, expert_next_states, expert_dones, env
            )
        )
        (union_states, union_actions, union_next_states, union_dones) = (
            utils.add_absorbing_states(
                union_states, union_actions, union_next_states, union_dones, env
            )
        )
    else:
        # ignore absorbing state
        union_init_states = np.c_[
            union_init_states, np.zeros(len(union_init_states), dtype=np.float32)
        ]
        expert_states = np.c_[
            expert_states, np.zeros(len(expert_states), dtype=np.float32)
        ]
        expert_next_states = np.c_[
            expert_next_states, np.zeros(len(expert_next_states), dtype=np.float32)
        ]
        union_states = np.c_[
            union_states, np.zeros(len(union_states), dtype=np.float32)
        ]
        print(union_states.shape)
        union_next_states = np.c_[
            union_next_states, np.zeros(len(union_next_states), dtype=np.float32)
        ]

    observation_dim = env.observation_space.shape[0]
    if xml_path:
        if "cheetah" in env_id.lower():
            observation_dim = 24
        elif "hopper" in env_id.lower():
            observation_dim = 14

    if "ant" in env_id.lower():
        observation_dim = 28

    # Create imitator
    is_discrete_action = env.action_space.dtype == int
    action_dim = env.action_space.n if is_discrete_action else env.action_space.shape[0]
    if algorithm == "demodice":
        imitator = demodice.DemoDICE(
            observation_dim, action_dim, is_discrete_action, config=config
        )
    elif algorithm == "avatar_dice":
        if xml_path:
            src_env = wrappers.create_il_env(
                env_id + "-v2",
                shift=shift_env,
                scale=scale_env,
                normalized_box_actions=False,
                robot=env_robot,
            )
        else:
            src_env = wrappers.create_il_env(
                env_name=env_id,
                shift=shift_env,
                scale=scale_env,
                normalized_box_actions=False,
                robot=config["src_env_robot"],
            )
        if "ant" in env_id.lower():
            src_obs_dim = 28
        else:
            src_obs_dim = src_env.observation_space.shape[0]

        src_imitator = demodice.DemoDICE(
            src_obs_dim,
            src_env.action_space.shape[0],
            is_discrete_action,
            config=config,
        )
        src_imitator.load(config["pretrained_model_path"])

        if src_only:
            import agent.avatar_dice_src as avatar_dice
        elif fixed_beta:
            from agent import avatar_dice_fixed_beta as avatar_dice
        else:
            from agent import avatar_dice

        imitator = avatar_dice.Avatar(
            observation_dim,
            action_dim,
            is_discrete_action,
            src_imitator.q_function,
            src_imitator.cost,
            src_obs_dim,
            src_env.action_space.shape[0],
            config=config,
        )
    elif algorithm == "smodice":
        # load src trajectory
        (
            src_expert_initial_states,
            src_expert_states,
            src_expert_actions,
            src_expert_next_states,
            src_expert_dones,
        ) = utils.load_d4rl_data(
            dataset_dir, env_id + "-v2", expert_dataset_name, 400, start_idx=0
        )

        # normalize expert dataset
        disc_cutoff = observation_dim - 1
        if src_expert_states.shape[1] < disc_cutoff:
            # Pad with zeros to reach disc_cutoff
            padding = np.zeros(
                (src_expert_states.shape[0], disc_cutoff - src_expert_states.shape[1])
            )
            src_expert_states = np.concatenate([src_expert_states, padding], axis=1)
        elif src_expert_states.shape[1] > disc_cutoff:
            # Truncate to disc_cutoff
            src_expert_states = src_expert_states[:, :disc_cutoff]
        src_expert_states = (src_expert_states + shift[:disc_cutoff]) * scale[
            :disc_cutoff
        ]
        src_expert_states = np.c_[
            src_expert_states, np.zeros(len(src_expert_states), dtype=np.float32)
        ]
        disc_cutoff += 1

        discriminator = Discriminator_SA(
            disc_cutoff, 0, hidden_dim=config["hidden_size"], device=device
        )
        dataset_expert = torch.utils.data.TensorDataset(
            torch.FloatTensor(src_expert_states)
        )
        expert_loader = torch.utils.data.DataLoader(
            dataset_expert,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        dataset_offline = torch.utils.data.TensorDataset(
            torch.FloatTensor(union_states)
        )
        offline_loader = torch.utils.data.DataLoader(
            dataset_offline,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        for i in tqdm(range(config["disc_iterations"])):
            loss = discriminator.update(expert_loader, offline_loader)

        imitator = smodice.SMODICE(
            observation_spec=observation_dim, action_spec=action_dim, config=config
        )
    elif algorithm == "gwil":
        (
            src_expert_initial_states,
            src_expert_states,
            src_expert_actions,
            src_expert_next_states,
            src_expert_dones,
        ) = utils.load_d4rl_data(
            dataset_dir, env_id + "-v2", expert_dataset_name, 1, start_idx=0
        )

        traj_expert = np.concatenate((src_expert_states, src_expert_actions), axis=1)
        src_expert_traj_len = math.ceil(src_expert_states.shape[0])
        imitator = gwil.GWIL(
            obs_dim=observation_dim,
            action_dim=action_dim,
            action_range=[
                float(env.action_space.low.min()),
                float(env.action_space.high.max()),
            ],
            config=config,
        )
        replay_buffer = ReplayBuffer(
            observation_dim,
            action_dim,
            int(config["replay_buffer_capacity"]),
            device,
            config,
        )
    elif algorithm == "igdf":
        (
            target_buffer,
            target_expert_buffer,
            source_buffer,
            source_expert_buffer,
            info,
        ) = igdf.load_data_and_train_contras(
            env_id,
            dataset_dir,
            config,
            observation_dim,
            action_dim,
            union_states,
            union_actions,
            union_next_states,
            union_dones,
            expert_states,
            expert_actions,
            expert_next_states,
            expert_dones,
        )

        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max()),
        ]
        sac_args.agent.obs_dim = observation_dim
        sac_args.agent.action_dim = action_dim
        imitator = SAC(observation_dim, action_dim, action_range, 256, sac_args, config)

        trainer = igdf.IQ_Learn(imitator)

    elif algorithm == "multisrc":
        src_imitator1 = demodice.DemoDICE(
            28,  # src_obs_dim,
            4,  # src_env.action_space.shape[0],
            False,  # is_discrete_action,
            config=config,
        )
        src_imitator2 = demodice.DemoDICE(
            28,  # src_obs_dim,
            4,  # src_env.action_space.shape[0],
            False,  # is_discrete_action,
            config=config,
        )

        src_imitator1.load(
            "/home/seanma0627/src/CDIL/pretrained_models/front_leg_ant.pickle"
        )
        src_imitator2.load(
            "/home/seanma0627/src/CDIL/pretrained_models/back_leg_ant.pickle"
        )

        import agent.avatar_dice_mul_src as avatar_dice

        print(f"=============observation dim: {observation_dim}=====================")
        print(f"=============union states: {union_states.shape}=====================")

        imitator = avatar_dice.Avatar(
            28,  # observation_dim,
            action_dim,
            is_discrete_action,
            src_imitator1.q_function,
            src_imitator2.q_function,
            28,  # src_obs_dim,
            4,  # src_env.action_space.shape[0],
            28,  # src_obs_dim,
            4,  # src_env.action_space.shape[0],
            config=config,
        )

    else:
        raise ValueError(f"{algorithm} is not supported algorithm name")

    print("Save interval :", config["save_interval"])
    # checkpoint dir
    checkpoint_dir = (
        f"checkpoint_imitator/{algorithm}/{env_id}/"
        f"{expert_dataset_name}_{expert_num_traj}_"
        f"{imperfect_dataset_names}_{imperfect_num_trajs}/{tb_path}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filepath = f"{checkpoint_dir}/0000"
    if config["resume"] and os.path.exists(checkpoint_filepath):
        # Load checkpoint.s
        imitator.init_dummy(observation_dim, action_dim)
        checkpoint_data = imitator.load(checkpoint_filepath)
        training_info = checkpoint_data["training_info"]
        training_info["iteration"] += 1
        print(f"Checkpoint '{checkpoint_filepath}' is resumed")
    else:
        print(f"No checkpoint is found: {checkpoint_filepath}")
        training_info = {
            "iteration": 0,
            "logs": [],
        }
    print(config["save_interval"])
    total_iterations = config["total_iterations"] + 1

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
    traj_num_idx = 0
    traj_len_idx = 0
    traj_idx = 0
    traj_count = 0
    traj_base_num = 0
    info_dict = {}
    with tqdm(
        total=total_iterations + 1,
        initial=training_info["iteration"],
        desc="",
        disable=os.environ.get("DISABLE_TQDM", False),
        ncols=70,
    ) as pbar:
        while training_info["iteration"] <= total_iterations:
            union_init_indices = np.random.randint(
                0, len(union_init_states), size=batch_size
            )
            expert_indices = np.random.randint(0, len(expert_states), size=batch_size)
            union_indices = np.random.randint(0, len(union_states), size=batch_size)
            if algorithm == "demodice":
                info_dict = imitator.update(
                    union_init_states_[union_init_indices],
                    expert_states_[expert_indices],
                    expert_actions_[expert_indices],
                    expert_next_states_[expert_indices],
                    union_states_[union_indices],
                    union_actions_[union_indices],
                    union_next_states_[union_indices],
                    training_info["iteration"],
                )
            elif algorithm == "avatar_dice":
                info_dict = imitator.update(
                    union_init_states_[union_init_indices],
                    expert_states_[expert_indices],
                    expert_actions_[expert_indices],
                    expert_next_states_[expert_indices],
                    union_states_,
                    union_actions_,
                    union_next_states_,
                    union_indices,
                    training_info["iteration"],
                    config["power_decay_weight"],
                    config["beta"],
                )
            elif algorithm == "smodice":
                # Get rewards
                with torch.no_grad():
                    obs_for_disc = torch.from_numpy(
                        np.array(union_states_[union_indices].cpu())
                    ).to(discriminator.device)
                    if config["state"]:
                        disc_input = obs_for_disc
                    else:
                        act_for_disc = torch.from_numpy(
                            np.array(union_actions_[union_indices].cpu())
                        ).to(discriminator.device)
                        disc_input = torch.cat([obs_for_disc, act_for_disc], axis=1)
                    reward = discriminator.predict_reward(disc_input)

                info_dict = imitator.train_step(
                    union_init_states_[union_init_indices],
                    union_states_[union_indices],
                    union_actions_[union_indices],
                    reward,
                    union_next_states_[union_indices],
                    union_dones_[union_indices],
                )
            elif algorithm == "gwil":
                if traj_idx < union_states.shape[0]:
                    traj_len = traj_lens[traj_len_idx]
                    replay_buffer.add(
                        union_states[traj_idx],
                        union_actions[traj_idx],
                        union_next_states[traj_idx],
                        union_dones[traj_idx],
                        union_dones[traj_idx],
                    )
                    traj_idx += 1
                    traj_count += 1
                    if traj_idx == (
                        traj_base_num
                        + traj_nums[traj_num_idx] * traj_lens[traj_len_idx]
                    ):
                        traj_base_num += (
                            traj_nums[traj_num_idx] * traj_lens[traj_len_idx]
                        )
                        traj_len_idx += 1
                        traj_num_idx += 1
                    if (traj_idx == union_states.shape[0]) or (traj_count == traj_len):
                        traj_count = 0
                        replay_buffer.process_trajectory(
                            traj_expert[:src_expert_traj_len],
                            src_expert_traj_len,
                        )

                if training_info["iteration"] >= config["num_seed_steps"]:
                    info_dict = imitator.update(
                        replay_buffer,
                        training_info["iteration"],
                        gw=True,
                        normalize_reward=False,
                        normalize_reward_batch=False,
                        include_external_reward=False,
                        weight_external_reward=1,
                        weight_gw_reward=1,
                    )
            elif algorithm == "igdf":
                target_batch = target_expert_buffer.sample(256 // 2)
                src_s, src_a, src_ss, done = source_expert_buffer.sample(256 // 2)

                logits, srcsa_repr, srcss_repr = info(
                    src_s, src_a, src_ss, return_repr=True
                )
                srcsa_repr = torch.linalg.norm(
                    srcsa_repr, dim=-1, keepdim=True
                )  # [128, 1]
                srcss_repr = torch.linalg.norm(
                    srcss_repr, dim=-1, keepdim=True
                )  # [128, 1]
                diagonal_elements = torch.diag(logits).reshape(-1, 1)
                src_info = diagonal_elements / (srcsa_repr * srcss_repr)  # [128, 1]

                sorted_indices = torch.argsort(src_info[:, 0])
                sorted_num = -64
                top_half_indices = sorted_indices[sorted_num:]
                src_s = src_s[top_half_indices]
                src_a = src_a[top_half_indices]
                src_ss = src_ss[top_half_indices]
                done = done[top_half_indices]
                info_temp = torch.exp(src_info[top_half_indices] * config["igdf_alpha"])
                mask = torch.ones((128 + 64, 1)).to(config["device"])
                mask[:64] = info_temp
                source_batch = [src_s, src_a, src_ss, done]
                batch = igdf.merge_batch(source_batch, target_batch)
                batch = [b.to(config["device"]) for b in batch]

                info_dict = trainer.train(batch, mask, training_info["iteration"])
            elif algorithm == "multisrc":
                info_dict = imitator.update(
                    union_init_states_[union_init_indices],
                    expert_states_[expert_indices],
                    expert_actions_[expert_indices],
                    expert_next_states_[expert_indices],
                    union_states_,
                    union_actions_,
                    union_next_states_,
                    union_indices,
                    training_info["iteration"],
                    config["power_decay_weight"],
                )

            else:
                raise ValueError(f"Undefined algorithm {algorithm}")

            if training_info["iteration"] % config["log_interval"] == 0:
                average_returns, evaluation_timesteps = evaluate_d4rl(
                    config, env_id, imitator, shift_env, scale_env
                )

                writer.add_scalar(
                    "Test average return", average_returns, training_info["iteration"]
                )
                info_dict.update({"eval": average_returns})
                print(
                    f"Eval: ave returns=d: {average_returns}"
                    f" ave episode length={evaluation_timesteps}"
                    f' / elapsed_time={time.time() - start_time} ({training_info["iteration"] / (time.time() - start_time)} it/sec)'
                )
                print("=========================")
                for key, val in info_dict.items():
                    if algorithm == "smodice":
                        print(f"{key:25}: {val.item():8.7f}")
                    else:
                        print(f"{key:25}: {val:8.7f}")
                print("=========================")

                training_info["logs"].append(
                    {"step": training_info["iteration"], "log": info_dict}
                )
                print(f'timestep {training_info["iteration"]} - log update...')
                print("Done!", flush=True)


            if (algorithm == "igdf") and (training_info["iteration"] % 10 == 0):
                for key, val in info_dict.items():
                    writer.add_scalar(
                        f"{key:25}", f"{val:8.7f}", training_info["iteration"]
                    )

            if (algorithm == "avatar_dice") and (training_info["iteration"] % 10 == 0):
                writer.add_scalar('Adaptive weight/c1',
                                  imitator.c1, training_info['iteration'])
                writer.add_scalar('Adaptive weight/c2',
                                  imitator.c2_smooth, training_info['iteration'])
                writer.add_scalar('Time weight decay', imitator.c2_smooth**config['power_decay_weight'] / (
                    imitator.c1**config['power_decay_weight'] + imitator.c2_smooth**config['power_decay_weight'] + 1e-6), training_info['iteration'])
                writer.add_scalar('LMAP', info_dict['mapping_loss'], training_info['iteration'])

            # Save checkpoint
            if (algorithm == "demodice") or (algorithm == "avatar_dice") or (algorithm == "gwil"):
                if training_info["iteration"] % config["save_interval"] == 0:
                    checkpoint_filepath = (
                        f"{checkpoint_dir}/{training_info['iteration']}.pickle"
                    )
                    # imitator.save(checkpoint_filepath, training_info)

            training_info["iteration"] += 1
            pbar.update(1)


if __name__ == "__main__":
    from config.config import get_parser

    # configurations
    args = get_parser().parse_args()
    config = vars(args)
    config["dataset_file_names"] = ast.literal_eval(config["dataset_file_names"])

    cfg = OmegaConf.load("config/iq_learn_conf/sac.yaml")

    print("Start running")
    run(config, cfg)
