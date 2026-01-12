import argparse
import ast
ENV_ID = [
    'Hopper',
    'HalfCheetah',
    'Ant',
    'Lift',
    'Door',
    'Wipe',
]

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument('--env_is_gym', default=1, type=int)
    # main
    parser.add_argument('--algorithm', default='demodice', type=str)
    parser.add_argument('--env_id', default='Hopper', type=str, choices=ENV_ID)
    parser.add_argument('--env_robot', default=None, type=str) #'Panda', 'UR5e'
    parser.add_argument('--src_env_robot', default=None, type=str) #'Panda', 'UR5e'
    parser.add_argument('--xml_path', default=None, type=str)
    parser.add_argument('--dataset_dir', default='dataset', type=str)
    parser.add_argument('--dataset_file_names', default=None)
    parser.add_argument('--src_expert_path', default=None, type=str)
    parser.add_argument('--src_random_path', default=None, type=str)
    parser.add_argument('--expert_dataset_name', default="expert-v2")
    parser.add_argument('--load_hdf5_dataset', default=1, type=int)
    parser.add_argument('--expert_num_traj', default=1, type=int)
    parser.add_argument('--imperfect_dataset_names', default=[], action='append')
    parser.add_argument('--imperfect_num_trajs', default=[], nargs='*', type=int)
    parser.add_argument('--imperfect_dataset_default_info', default=(["expert-v2", "random-v2"], [50, 500])) # source: [400, 1600] , target: [50, 200]\[10, 100], default=(["expert-v2", "random-v2"], [1, 100]), default=(["expert-v1", "cloned-v1"], [400, 1600])
    # parser.add_argument('--imperfect_dataset_default_info', type=ast.literal_eval, default=(["expert-v2", "random-v2"], [50, 500])) # source: [400, 1600] , target: [50, 200]\[10, 100], default=(["expert-v2", "random-v2"], [1, 100]), default=(["expert-v1", "cloned-v1"], [400, 1600])
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--tb_path', default=None, type=str)
    # avatar DICE
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument('--flow_model_path', default=None, type=str)
    parser.add_argument('--flow_model_action_path', default=None, type=str)
    parser.add_argument('--power_decay_weight', default=1, type=int)
    parser.add_argument('--beta', default=0.3, type=float)
    # smodice
    parser.add_argument('--hidden_sizes', default=(256, 256))
    parser.add_argument('--disc_type', default='learned', type=str)  
    parser.add_argument('--use_policy_entropy_constraint', default=False, type=boolean)
    parser.add_argument('--target_entropy', default=None, type=float)
    parser.add_argument('--f', default='kl', type=str)
    parser.add_argument('--disc_iterations', default=int(1e2), type=int)
    parser.add_argument('--v_l2_reg', default=0.0001, type=float)
    parser.add_argument('--mean_range', default=(-7.24, 7.24))
    parser.add_argument('--logstd_range', default=(-5., 2.))
    parser.add_argument('--state', default=True, type=boolean)
    # gwil
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--hidden_depth', default=2, type=int)
    parser.add_argument('--log_std_bounds', default=[-5, 2])
    parser.add_argument('--num_seed_steps', default=5000, type=int)
    parser.add_argument('--replay_buffer_capacity', default=1e7, type=int)
    # igdf
    parser.add_argument('--buffer_size', default=2000001, type=int)
    parser.add_argument('--repr_dim', default=64, type=int)
    parser.add_argument('--ensemble_size', default=1, type=int)
    parser.add_argument('--info_lr', default=3e-4, type=float)
    parser.add_argument('--info_batch_size', default=128, type=int)
    parser.add_argument('--num_update', default=7000, type=int)
    # parser.add_argument('--xi', default=0.5, type=float)
    parser.add_argument('--igdf_alpha', default=1.0, type=float)

    # optional
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--total_iterations', default=int(5e5), type=int)
    parser.add_argument('--save_interval', default=int(1e4), type=int)
    parser.add_argument('--log_interval', default=int(1e4), type=int)
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--using_absorbing', default=False, type=bool)
    parser.add_argument('--grad_reg_coeffs', default=(0.1, 1e-4))
    parser.add_argument('--use_last_layer_bias_cost', default=False, type=bool)
    parser.add_argument('--use_last_layer_bias_critic', default=False, type=bool)
    parser.add_argument('--kernel_initializer', default='he_normal', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--src_only', default=False, type=bool)
    parser.add_argument('--fixed_beta', default=False, type=bool)
    parser.add_argument('--psi', default=0.9, type=float)
    
    return parser