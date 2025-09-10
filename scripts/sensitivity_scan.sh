### running commands ###

zcat -qf /var/log/apt/history.log* \
  | grep -E '^(Commandline: apt(-get)? install)' \
  | grep -v -- '--reinstall' \
  | grep -oP 'install\s+\K.*' \
  | tr ' ' '\n' \
  | sort -u \
  | paste -sd ' ' -

--no-install-recommends -y apt-utils autoconf automake build-essential ca-certificates curl gfortran git gnupg htop ibverbs-providers ibverbs-utils iputils-ping jq less libatlas-base-dev libbz2-dev libc-ares2 libedit-dev libegl-mesa0 libegl1-mesa-dev libglfw3 libglib2.0-0 libglx-mesa0 libgoogle-glog-dev libhdf5-dev libhwloc15 libibumad-dev libibumad3 libibverbs-dev libibverbs1 libleveldb-dev liblmdb-dev libncurses6 libncursesw6 libnl-3-dev libnl-route-3-200 libnl-route-3-dev libnss-ldap libnuma-dev libnuma1 libosmesa6 libosmesa6-dev libpam-ldap libpmi2-0-dev libpng-dev libprotobuf-dev librdmacm-dev librdmacm1 libre2-dev libsnappy-dev libsndfile1 libtcmalloc-minimal4 libtool nano nasm neovim ninja-build numactl nvtop openssh-client openssh-server patch patchelf pkg-config protobuf-compiler python-is-python3 python3 python3-dev python3-venv python3.12-dev rapidjson-dev ripgrep screen sox sudo supervisor tmux tree unzip vim wget



hopper
1.
export CUDA_VISIBLE_DEVICES=0 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Hopper \
  --xml_path=./env/hopper_target.xml \
  --dataset_file_names="['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/hopper.pickle \
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [20,50])' \
  --actor_lr 5e-5 \
  --tb_path=sensitivity_scan_tfboard/hopper_set1_seed0 \
  --seed 0
  # --imperfect_num_trajs 20 50 \

2.
export CUDA_VISIBLE_DEVICES=0 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Hopper \
  --xml_path=./env/hopper_target.xml \
  --dataset_file_names="['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/hopper.pickle \
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \
  --expert_num_traj 5 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [20,50])' \
  --actor_lr 5e-5 \
  --tb_path=sensitivity_scan_tfboard/hopper_set2_seed0 \
  --seed 0
  # --imperfect_num_trajs 20 50 \

3.
export CUDA_VISIBLE_DEVICES=0 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Hopper \
  --xml_path=./env/hopper_target.xml \
  --dataset_file_names="['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/hopper.pickle \
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [50,100])' \
  --actor_lr 5e-5 \
  --tb_path=sensitivity_scan_tfboard/hopper_set3_seed0 \
  --seed 0
  # --imperfect_num_trajs 50 100 \

###############################

wipe
1.
export CUDA_VISIBLE_DEVICES=1 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Wipe \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', 'Wipe_UR5e_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/wipe.pickle \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [10,50])' \
  --critic_lr 1e-4 \
  --tb_path=sensitivity_scan_tfboard/wipe_set1_seed0 \
  --seed 0
  # --imperfect_num_trajs 10 50 1>wipe_set1.log 2>&1

2.
export CUDA_VISIBLE_DEVICES=1 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Wipe \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', 'Wipe_UR5e_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/wipe.pickle \
  --expert_num_traj 5 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [10,50])' \
  --critic_lr 1e-4 \
  --tb_path=sensitivity_scan_tfboard/wipe_set2_seed0 \
  --seed 0
  # --imperfect_num_trajs 10 50 1>wipe_set2.log 2>&1

3.
export CUDA_VISIBLE_DEVICES=6 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Wipe \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', None]" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/wipe.pickle \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [50,100])' \
  --critic_lr 1e-4 \
  --tb_path=sensitivity_scan_tfboard/wipe_set3_seed0 \
  --seed 0
  # --imperfect_num_trajs 50 100 1>wipe_set3.log 2>&1

###############################

ant
1.
export CUDA_VISIBLE_DEVICES=2 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Ant \
  --xml_path=./env/ant_target.xml \
  --dataset_file_names="['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/ant.pickle \
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=sensitivity_scan_tfboard/ant_set1_seed0 \
  --seed 0
  # --imperfect_num_trajs 1 100 1>ant_set1.log 2>&1

2.
export CUDA_VISIBLE_DEVICES=2 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Ant \
  --xml_path=./env/ant_target.xml \
  --dataset_file_names="['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/ant.pickle \
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \
  --expert_num_traj 5 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=sensitivity_scan_tfboard/ant_set2_seed0 \
  --seed 0
  # --imperfect_num_trajs 1 100 1>ant_set2.log 2>&1

3.
export CUDA_VISIBLE_DEVICES=2 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Ant \
  --xml_path=./env/ant_target.xml \
  --dataset_file_names="['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_500.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/ant.pickle \
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [5,500])' \
  --tb_path=sensitivity_scan_tfboard/ant_set3_seed0 \
  --seed 0
  # --imperfect_num_trajs 5 500 1>ant_set3.log 2>&1


###############################

cheetah
1.
export CUDA_VISIBLE_DEVICES=3 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=HalfCheetah \
  --xml_path=./env/cheetah_target.xml \
  --dataset_file_names="['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/cheetah.pickle \
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=sensitivity_scan_tfboard/cheetah_set1_seed0 \
  --seed 0
  # --imperfect_num_trajs 1 100 1>cheetah_set1.log 2>&1

2.
export CUDA_VISIBLE_DEVICES=3 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=HalfCheetah \
  --xml_path=./env/cheetah_target.xml \
  --dataset_file_names="['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/cheetah.pickle \
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \
  --expert_num_traj 5 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --tb_path=sensitivity_scan_tfboard/cheetah_set2_seed0 \
  --seed 0
  # --imperfect_num_trajs 1 100 1>cheetah_set2.log 2>&1

3.
export CUDA_VISIBLE_DEVICES=7 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=HalfCheetah \
  --xml_path=./env/cheetah_target.xml \
  --dataset_file_names="['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_500.npz']" \
  --load_hdf5_dataset=0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/cheetah.pickle \
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [5,500])' \
  --tb_path=sensitivity_scan_tfboard/cheetah_set3_seed0 \
  --seed 0
  # --imperfect_num_trajs 5 500 1>cheetah_set3.log 2>&1

###############################

door
1.
export CUDA_VISIBLE_DEVICES=4 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Door \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --tb_path=sensitivity_scan_tfboard/door_set1_seed0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/door.pickle \
  --flow_model_path=./flow_model/door/model/door_flow_seed1.pt \
  --flow_model_action_path=./flow_model/door/model_action/door_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --seed 0
  # --imperfect_num_trajs 1 100 1>door_set1.log 2>&1

2.
export CUDA_VISIBLE_DEVICES=4 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Door \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --tb_path=sensitivity_scan_tfboard/door_set2_seed0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/door.pickle \
  --flow_model_path=./flow_model/door/model/door_flow_seed1.pt \
  --flow_model_action_path=./flow_model/door/model_action/door_flow_seed1.pt \
  --expert_num_traj 5 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --seed 0
  # --imperfect_num_trajs 1 100 1>door_set2.log 2>&1

3.
export CUDA_VISIBLE_DEVICES=4 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Door \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_500.npz']" \
  --load_hdf5_dataset=0 \
  --tb_path=sensitivity_scan_tfboard/door_set3_seed0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/door.pickle \
  --flow_model_path=./flow_model/door/model/door_flow_seed1.pt \
  --flow_model_action_path=./flow_model/door/model_action/door_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [5,500])' \
  --seed 0
  # --imperfect_num_trajs 5 500 1>door_set3.log 2>&1

###############################

lift
1.
export CUDA_VISIBLE_DEVICES=5 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Lift \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --tb_path=sensitivity_scan_tfboard/lift_set1_seed0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/lift.pickle \
  --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \
  --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --seed 0
  # --imperfect_num_trajs 1 100 1>lift_set1.log 2>&1

2.
export CUDA_VISIBLE_DEVICES=5 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Lift \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_100.npz']" \
  --load_hdf5_dataset=0 \
  --tb_path=sensitivity_scan_tfboard/lift_set2_seed0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/lift.pickle \
  --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \
  --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \
  --expert_num_traj 5 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [1,100])' \
  --seed 0
  # --imperfect_num_trajs 1 100 1>lift_set2.log 2>&1

3.
export CUDA_VISIBLE_DEVICES=5 && \
cd ~/src/CDIL && conda activate cdil && \
python train_il.py \
  --env_is_gym=0 \
  --algorithm=avatar_dice \
  --env_id=Lift \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names="['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_500.npz']" \
  --load_hdf5_dataset=0 \
  --tb_path=sensitivity_scan_tfboard/lift_set3_seed0 \
  --log_interval=10000 \
  --pretrained_model_path=./pretrained_models/lift.pickle \
  --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \
  --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info '(["expert-v2","random-v2"], [5,500])' \
  --seed 0
  # --imperfect_num_trajs 5 500 1>lift_set3.log 2>&1
