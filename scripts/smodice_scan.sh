export CONDA_VISIBLE_DEVICES=0 && \
cd ~/src/CDIL && conda activate cdil && python train_il.py \
  --env_is_gym=0 \
  --algorithm=smodice \
  --env_id=Door \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_500.npz']" \
  --src_expert_path=Door_Panda_400_206.56235546643248_500.0.npz \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --disc_type=learned \
  --disc_iterations 10 \
  --expert_num_traj 10 \
  --imperfect_dataset_default_info='(["expert-v2", "random-v2"], [400,400])' \
  --tb_path=smodice_tfboard/smodice_door_seed0 \
  --seed=0


export CUDA_VISIBLE_DEVICES=0 && \
conda activate cdil && cd ~/src/CDIL && python train_il.py \
  --env_is_gym=0 \
  --algorithm=smodice \
  --env_id=Lift \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_100.npz']" \
  --src_expert_path=Lift_Panda_400_193.8453640401841_500.0.npz \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --disc_type=learned \
  --disc_iterations 10 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2", "random-v2"], [1, 100])' \
  --tb_path=tfboard/smodice/lift_set1_seed0 \
  --seed=0
