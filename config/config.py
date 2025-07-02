import argparse
ENV_ID = [
    'Hopper',
    'HalfCheetah',
    'Ant',
    'Lift',
    'Door',
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
    parser.add_argument('--expert_dataset_name', default="expert-v2")
    parser.add_argument('--load_hdf5_dataset', default=1, type=int)
    parser.add_argument('--expert_num_traj', default=1, type=int)
    parser.add_argument('--imperfect_dataset_names', default=[], action='append')
    parser.add_argument('--imperfect_num_trajs', default=[], action='append', type=int)
    parser.add_argument('--imperfect_dataset_default_info', default=(["expert-v2", "random-v2"], [1, 50])) # source: [400, 1600] , target: [50, 200]\[10, 100], default=(["expert-v2", "random-v2"], [1, 100]), default=(["expert-v1", "cloned-v1"], [400, 1600])
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--tb_path', default=None, type=str)
    # avatar DICE
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument('--flow_model_path', default=None, type=str)
    parser.add_argument('--flow_model_action_path', default=None, type=str)
    # optional
    parser.add_argument('--total_iterations', default=int(5e5), type=int)
    parser.add_argument('--save_interval', default=int(1e5), type=int)
    parser.add_argument('--log_interval', default=int(1e4), type=int)
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--using_absorbing', default=False, type=bool)
    parser.add_argument('--grad_reg_coeffs', default=(0.1, 1e-4))
    parser.add_argument('--use_last_layer_bias_cost', default=False, type=bool)
    parser.add_argument('--use_last_layer_bias_critic', default=False, type=bool)
    parser.add_argument('--kernel_initializer', default='he_normal', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--power_decay_weight', default=1, type=int)
    return parser