python train_il.py \
  --env_is_gym=0 \
  --algorithm=demodice \
  --env_id=Hopper \
  --xml_path=./env/hopper_target.xml \
  --dataset_file_names="['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [20,50])' \
  --actor_lr 5e-5 \
  --tb_path=demodice_ablation/hopper_set1_seed0 \
  --seed 0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=demodice \
  --env_id=Wipe \
  --env_robot=UR5e \
  --dataset_file_names="['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', 'Wipe_UR5e_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [10,50])' \
  --critic_lr 1e-4 \
  --tb_path=demodice_ablation/wipe_set1_seed0 \
  --seed 0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=demodice \
  --env_id=Ant \
  --xml_path=./env/ant_target.xml \
  --dataset_file_names="['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=demodice_ablation/ant_set1_seed0 \
  --seed 0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=demodice \
  --env_id=HalfCheetah \
  --xml_path=./env/cheetah_target.xml \
  --dataset_file_names="['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=demodice_ablation/cheetah_set1_seed0 \
  --seed 0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=demodice \
  --env_id=Door \
  --env_robot=UR5e \
  --dataset_file_names="['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_100.npz']" \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=demodice_ablation/door_set1_seed0 \
  --seed 0



python train_il.py \
  --env_is_gym=0 \
  --algorithm=demodice \
  --env_id=Lift \
  --env_robot=UR5e \
  --dataset_file_names="['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_100.npz']" \
  --log_interval=10000 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=demodice_ablation/lift_set1_seed0 \
  --seed 0
