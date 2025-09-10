#!/usr/bin/env bash
set -euo pipefail

SESSION="0"
ATTACH=1
SEED=""

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --session|-s) SESSION="$2"; shift 2 ;;
    --no-attach) ATTACH=0; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [[ -z "${SEED:-}" ]]; then
  echo "Usage: $0 --seed <int> [--session <name>] [--no-attach]" >&2
  exit 2
fi

# --- helpers ---
ensure_conda_snippet='
# Load conda (best-effort)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then . "$HOME/miniconda3/etc/profile.d/conda.sh"; fi
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then . "$HOME/miniforge3/etc/profile.d/conda.sh"; fi
if command -v conda >/dev/null 2>&1; then
  conda activate cdil
else
  echo "[warn] conda not found; hoping python resolves to your env"
fi
'

mkcmd() {
  # $1: out file
  # $2: CUDA_VISIBLE_DEVICES
  # $3..: body (python command lines WITHOUT last two switches)
  local out="$1" gpu="$2"
  shift 2
  cat >"$out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${gpu}
cd \$HOME/src/CDIL
${ensure_conda_snippet}

python train_il.py \\
$* \\
  --tb_path=\${TB_PATH} \\
  --seed \${SEED}
EOF
  chmod +x "$out"
}

new_win_run() {
  # $1 name, $2 script path, $3 tb_path
  local name="$1" file="$2" tb="$3"
  # Wait 60s after the terminal is created before launching the experiment.
  tmux new-window -t "${SESSION}:" -n "$name" "sleep 10; SEED=${SEED} TB_PATH=${tb} bash '$file'"
}

# --- start tmux session w/ monitor window (0) ---
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n "mon"    # create session & window 0
else
  # if session exists, ensure window 0 exists/named
  tmux rename-window -t "${SESSION}:0" "mon" 2>/dev/null || true
fi

# split vertically and run monitors
tmux select-window -t "${SESSION}:0"
tmux split-window -t "${SESSION}:0" -v
tmux select-pane -t "${SESSION}:0.0"
sleep 10
tmux send-keys -t "${SESSION}:0.0" "htop -u \"$USER\"" C-m
tmux select-pane -t "${SESSION}:0.1"
sleep 10
tmux send-keys -t "${SESSION}:0.1" "nvtop" C-m
tmux select-layout -t "${SESSION}:0" even-vertical

# --- create per-exp temp scripts ---
TMPDIR="\${TMPDIR:-/tmp}"
# Hopper
H1="\$TMPDIR/exp_h1_seed${SEED}.sh"
mkcmd "\$H1" 0 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/hopper.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [20,50])' \\
  --actor_lr 5e-5"
H2="\$TMPDIR/exp_h2_seed${SEED}.sh"
mkcmd "\$H2" 0 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/hopper.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --expert_num_traj 5 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [20,50])' \\
  --actor_lr 5e-5"
H3="\$TMPDIR/exp_h3_seed${SEED}.sh"
mkcmd "\$H3" 0 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/hopper.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [50,100])' \\
  --actor_lr 5e-5"

# Wipe
W1="\$TMPDIR/exp_w1_seed${SEED}.sh"
mkcmd "\$W1" 1 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Wipe \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/wipe.pickle \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4"
W2="\$TMPDIR/exp_w2_seed${SEED}.sh"
mkcmd "\$W2" 1 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Wipe \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/wipe.pickle \\
  --expert_num_traj 5 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4"
W3="\$TMPDIR/exp_w3_seed${SEED}.sh"
mkcmd "\$W3" 6 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Wipe \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/wipe.pickle \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [50,100])' \\
  --critic_lr 1e-4"

# Ant
A1="\$TMPDIR/exp_a1_seed${SEED}.sh"
mkcmd "\$A1" 2 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Ant \\
  --xml_path=./env/ant_target.xml \\
  --dataset_file_names=\"['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/ant.pickle \\
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"
A2="\$TMPDIR/exp_a2_seed${SEED}.sh"
mkcmd "\$A2" 2 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Ant \\
  --xml_path=./env/ant_target.xml \\
  --dataset_file_names=\"['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/ant.pickle \\
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \\
  --expert_num_traj 5 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"
A3="\$TMPDIR/exp_a3_seed${SEED}.sh"
mkcmd "\$A3" 2 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Ant \\
  --xml_path=./env/ant_target.xml \\
  --dataset_file_names=\"['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/ant.pickle \\
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,500])'"

# HalfCheetah
C1="\$TMPDIR/exp_c1_seed${SEED}.sh"
mkcmd "\$C1" 3 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=HalfCheetah \\
  --xml_path=./env/cheetah_target.xml \\
  --dataset_file_names=\"['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/cheetah.pickle \\
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"
C2="\$TMPDIR/exp_c2_seed${SEED}.sh"
mkcmd "\$C2" 3 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=HalfCheetah \\
  --xml_path=./env/cheetah_target.xml \\
  --dataset_file_names=\"['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/cheetah.pickle \\
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \\
  --expert_num_traj 5 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"
C3="\$TMPDIR/exp_c3_seed${SEED}.sh"
mkcmd "\$C3" 7 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=HalfCheetah \\
  --xml_path=./env/cheetah_target.xml \\
  --dataset_file_names=\"['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/cheetah.pickle \\
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,500])'"

# Door
D1="\$TMPDIR/exp_d1_seed${SEED}.sh"
mkcmd "\$D1" 4 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Door \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --tb_path=sensitivity_scan_tfboard/door_setX_will_be_overwritten \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/door.pickle \\
  --flow_model_path=./flow_model/door/model/door_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/door/model_action/door_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"
# (tb_path above is overwritten via TB_PATH env; kept to preserve structure)
D2="\$TMPDIR/exp_d2_seed${SEED}.sh"
mkcmd "\$D2" 4 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Door \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --tb_path=sensitivity_scan_tfboard/door_setX_will_be_overwritten \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/door.pickle \\
  --flow_model_path=./flow_model/door/model/door_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/door/model_action/door_flow_seed1.pt \\
  --expert_num_traj 5 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"
D3="\$TMPDIR/exp_d3_seed${SEED}.sh"
mkcmd "\$D3" 4 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Door \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --tb_path=sensitivity_scan_tfboard/door_setX_will_be_overwritten \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/door.pickle \\
  --flow_model_path=./flow_model/door/model/door_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/door/model_action/door_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,500])'"

# Lift
L1="\$TMPDIR/exp_l1_seed${SEED}.sh"
mkcmd "\$L1" 5 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Lift \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --tb_path=sensitivity_scan_tfboard/lift_setX_will_be_overwritten \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/lift.pickle \\
  --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"
L2="\$TMPDIR/exp_l2_seed${SEED}.sh"
mkcmd "\$L2" 5 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Lift \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --tb_path=sensitivity_scan_tfboard/lift_setX_will_be_overwritten \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/lift.pickle \\
  --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \\
  --expert_num_traj 5 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"
L3="\$TMPDIR/exp_l3_seed${SEED}.sh"
mkcmd "\$L3" 5 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Lift \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', None]\" \\
  --load_hdf5_dataset=0 \\
  --tb_path=sensitivity_scan_tfboard/lift_setX_will_be_overwritten \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/lift.pickle \\
  --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,500])'"

# --- open windows and run ---
new_win_run "h1" "\$H1" "sensitivity_scan_tfboard/hopper_set1_seed${SEED}"
new_win_run "h2" "\$H2" "sensitivity_scan_tfboard/hopper_set2_seed${SEED}"
new_win_run "h3" "\$H3" "sensitivity_scan_tfboard/hopper_set3_seed${SEED}"

new_win_run "w1" "\$W1" "sensitivity_scan_tfboard/wipe_set1_seed${SEED}"
new_win_run "w2" "\$W2" "sensitivity_scan_tfboard/wipe_set2_seed${SEED}"
new_win_run "w3" "\$W3" "sensitivity_scan_tfboard/wipe_set3_seed${SEED}"

new_win_run "a1" "\$A1" "sensitivity_scan_tfboard/ant_set1_seed${SEED}"
new_win_run "a2" "\$A2" "sensitivity_scan_tfboard/ant_set2_seed${SEED}"
new_win_run "a3" "\$A3" "sensitivity_scan_tfboard/ant_set3_seed${SEED}"

new_win_run "c1" "\$C1" "sensitivity_scan_tfboard/cheetah_set1_seed${SEED}"
new_win_run "c2" "\$C2" "sensitivity_scan_tfboard/cheetah_set2_seed${SEED}"
new_win_run "c3" "\$C3" "sensitivity_scan_tfboard/cheetah_set3_seed${SEED}"

new_win_run "d1" "\$D1" "sensitivity_scan_tfboard/door_set1_seed${SEED}"
new_win_run "d2" "\$D2" "sensitivity_scan_tfboard/door_set2_seed${SEED}"
new_win_run "d3" "\$D3" "sensitivity_scan_tfboard/door_set3_seed${SEED}"

new_win_run "l1" "\$L1" "sensitivity_scan_tfboard/lift_set1_seed${SEED}"
new_win_run "l2" "\$L2" "sensitivity_scan_tfboard/lift_set2_seed${SEED}"
new_win_run "l3" "\$L3" "sensitivity_scan_tfboard/lift_set3_seed${SEED}"

# focus back to monitor
tmux select-window -t "${SESSION}:0"

# attach/switch
if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
