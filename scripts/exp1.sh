#!/usr/bin/env bash
# =============================================================================
# Experiment 1: Main Evaluation - AdaptDICE (Default Setting)
# Paper Reference: Table 1, Figure D.1 (training curves)
# Description: AdaptDICE on all 6 environments with Default dataset configuration
#   - MuJoCo: Hopper, Ant, HalfCheetah
#   - Robosuite: BlockLifting, DoorOpening, TableWiping
# =============================================================================
set -euo pipefail

SESSION="exp1_adaptdice"
ATTACH=0
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
  local name="$1" file="$2" tb="$3"
  tmux new-window -t "${SESSION}:" -n "$name" "sleep 10; SEED=${SEED} TB_PATH=${tb} bash '$file'"
  tmux set-option -w remain-on-exit on
  tmux set-option -g mouse on
}

# --- start tmux session ---
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
fi

# --- create per-exp temp scripts ---
TMPDIR="${TMPDIR:-/tmp/exp1}"
mkdir -p "$TMPDIR"

# Hopper (Default: 1 expert, 10 expert + 50 random in sub-optimal)
H="$TMPDIR/hopper_seed${SEED}.sh"
mkcmd "$H" 0 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/hopper.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4"

# Ant (Default: 1 expert, 1 expert + 100 random in sub-optimal)
A="$TMPDIR/ant_seed${SEED}.sh"
mkcmd "$A" 1 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Ant \\
  --xml_path=./env/ant_target.xml \\
  --dataset_file_names=\"['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/ant.pickle \\
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

# HalfCheetah (Default: 1 expert, 1 expert + 100 random in sub-optimal)
C="$TMPDIR/cheetah_seed${SEED}.sh"
mkcmd "$C" 2 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=HalfCheetah \\
  --xml_path=./env/cheetah_target.xml \\
  --dataset_file_names=\"['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/cheetah.pickle \\
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

# BlockLifting (Default: 1 expert, 1 expert + 100 random in sub-optimal)
L="$TMPDIR/lift_seed${SEED}.sh"
mkcmd "$L" 3 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Lift \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/lift.pickle \\
  --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

# DoorOpening (Default: 1 expert, 1 expert + 100 random in sub-optimal)
D="$TMPDIR/door_seed${SEED}.sh"
mkcmd "$D" 4 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Door \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/door.pickle \\
  --flow_model_path=./flow_model/door/model/door_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/door/model_action/door_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

# TableWiping (Default: 1 expert, 10 expert + 50 random in sub-optimal)
W="$TMPDIR/wipe_seed${SEED}.sh"
mkcmd "$W" 5 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Wipe \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', 'Wipe_UR5e_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/wipe.pickle \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4"

# --- open windows and run ---
new_win_run "hopper" "$H" "tfboard/exp1_adaptdice/hopper_seed${SEED}"
new_win_run "ant" "$A" "tfboard/exp1_adaptdice/ant_seed${SEED}"
new_win_run "cheetah" "$C" "tfboard/exp1_adaptdice/cheetah_seed${SEED}"
new_win_run "lift" "$L" "tfboard/exp1_adaptdice/lift_seed${SEED}"
new_win_run "door" "$D" "tfboard/exp1_adaptdice/door_seed${SEED}"
new_win_run "wipe" "$W" "tfboard/exp1_adaptdice/wipe_seed${SEED}"

# cleanup
tmux kill-window -t 0 2>/dev/null || true
tmux select-window -t "${SESSION}:1"

if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
