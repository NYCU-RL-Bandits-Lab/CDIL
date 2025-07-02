#!/bin/bash

# # demodice in source domain(mujoco)
# python train_il.py \
#   --env_is_gym=1 \
#   --algorithm=demodice \
#   --env_id=Hopper \
#   --load_hdf5_dataset=1 \
#   --tb_path=tensorboard/test0 \
#   --log_interval=10

# # demodice in target domain(mujoco)
# python train_il.py \
#   --env_is_gym=0 \
#   --algorithm=demodice \
#   --env_id=Hopper \
#   --xml_path=./env/hopper_target.xml \
#   --dataset_file_names="['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']" \
#   --load_hdf5_dataset=0 \
#   --tb_path=tensorboard/test0 \
#   --log_interval=10


# # avtar_dice in target domain(mujoco)
# python train_il.py \
#   --env_is_gym=0 \
#   --algorithm=avatar_dice \
#   --env_id=Hopper \
#   --xml_path=./env/hopper_target.xml \
#   --dataset_file_names="['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']" \
#   --load_hdf5_dataset=0 \
#   --tb_path=tensorboard/test0 \
#   --log_interval=10 \
#   --pretrained_model_path=./pretrained_models/hopper.pickle \
#   --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \
#   --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \


# # demodice in target domain(robosuite)
# python train_il.py \
#   --env_is_gym=0 \
#   --algorithm=demodice \
#   --env_id=Lift \
#   --env_robot=Panda \
#   --dataset_file_names="['Lift_Panda_5_191.6623830954895_500.0.npz', 'Lift_Panda_400_193.8453640401841_500.0.npz', 'Lift_Panda_random_1600.npz']" \
#   --load_hdf5_dataset=0 \
#   --tb_path=tensorboard/test0 \
#   --log_interval=10000 \




# # avtar_dice in target domain(robosuite)
# python train_il.py \
#   --env_is_gym=0 \
#   --algorithm=avatar_dice \
#   --env_id=Lift \
#   --env_robot=UR5e \
#   --src_env_robot=Panda \
#   --dataset_file_names="['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_100.npz']" \
#   --load_hdf5_dataset=0 \
#   --tb_path=tensorboard/test0 \
#   --log_interval=10000 \
#   --pretrained_model_path=./pretrained_models/lift.pickle \
#   --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \
#   --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \