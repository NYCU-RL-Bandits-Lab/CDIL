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


# smodice in target domain(mujoco)
# python train_il.py \
#   --env_is_gym=0 \
#   --algorithm=smodice \
#   --env_id=Ant \
#   --xml_path=./env/ant_target.xml \
#   --dataset_file_names="['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']" \
#   --src_expert_path=Door_Panda_400_206.56235546643248_500.0.npz \
#   --load_hdf5_dataset=0 \
#   --tb_path=tensorboard/smodice_ant_ver1_seed2 \
#   --log_interval=10000 \
#   --disc_type=learned \
#   --actor_lr=1e-4 \
#   --seed=2 \


# # smodice in target domain(robosuite)
# python train_il.py \
#   --env_is_gym=0 \
#   --algorithm=smodice \
#   --env_id=Wipe \
#   --env_robot=UR5e \
#   --src_env_robot=Panda \
#   --dataset_file_names="['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', 'Wipe_UR5e_random_100.npz']" \
#   --src_expert_path=Wipe_Panda_400_108.31298115605115_496.2325.npz \
#   --load_hdf5_dataset=0 \
#   --tb_path=tensorboard/smodice_wipe_ver1_seed2 \
#   --log_interval=10000 \
#   --actor_lr=1e-4 \
#   --seed=2 \


# datasets
# "['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']"
# "['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']"
# "['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']"
# "['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_100.npz']"
# "['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_100.npz']"
# "['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', 'Wipe_UR5e_random_100.npz']"


