export CUDA_VISIBLE_DEVICES=1 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=gwil \
  --env_id=Hopper \
  --xml_path=./env/hopper_target.xml \
  --dataset_file_names="['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [20,50])' \
  --actor_lr 5e-5 \
  --tb_path=gwil_tfboard/gwil_hopper_seed0 \
  --seed=0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=gwil \
  --env_id=Wipe \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', 'Wipe_UR5e_random_50.npz']" \
  --src_expert_path=Wipe_Panda_400_108.31298115605115_496.2325.npz \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2", "random-v2"], [10, 50])' \
  --critic_lr=1e-4 \
  --tb_path=gwil_tfboard/gwil_wipe_seed0 \
  --seed=0


CUDA_VISIBLE_DEVICES=1 && \
cd ~/src/CDIL && conda activate cdil && \
python test_model.py \
  --env_is_gym=0 \
  --algorithm=gwil \
  --env_id=Ant \
  --xml_path=./env/ant_target.xml \
  --dataset_file_names="['target_ant_400_5718.754833834492_987.1325.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_500.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 10 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [400,400])' \
  --seed=0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=gwil \
  --env_id=HalfCheetah \
  --xml_path=./env/cheetah_target.xml \
  --dataset_file_names="['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=gwil_tfboard/gwil_cheetah_seed0 \
  --seed=0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=gwil \
  --env_id=Door \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_100.npz']" \
  --src_expert_path=Door_Panda_400_206.56235546643248_500.0.npz \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2", "random-v2"], [10, 50])' \
  --tb_path=gwil_tfboard/gwil_door_seed0 \
  --seed=0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=gwil \
  --env_id=Lift \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_100.npz']" \
  --src_expert_path=Lift_Panda_400_193.8453640401841_500.0.npz \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2", "random-v2"], [10, 50])' \
  --tb_path=gwil_tfboard/gwil_lift_seed0 \
  --seed=0
