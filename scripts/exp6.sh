#!/usr/bin/env bash
# Experiment 6: AdaptDICE Dataset Sensitivity (Expert Rich / Sub-Optimal Rich) - Table 3, Figure D.3
set -euo pipefail

SESSION="exp6_sensitivity"
ATTACH=0
SEED=""
MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --session|-s) SESSION="$2"; shift 2 ;;
    --no-attach) ATTACH=0; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [[ -z "${SEED:-}" || -z "${MODE:-}" ]]; then
  echo "Usage: $0 --seed <int> --mode <expert_rich|suboptimal_rich> [--session <name>]" >&2
  exit 2
fi

ensure_conda_snippet='
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
}

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
fi

TMPDIR="${TMPDIR:-/tmp/exp6}"
mkdir -p "$TMPDIR"

if [[ "$MODE" == "expert_rich" ]]; then
  HOPPER_EXPERT=5
  HOPPER_IMPERFECT="[10,50]"
  OTHER_EXPERT=5
  OTHER_IMPERFECT="[1,100]"
  MODE_TAG="expert_rich"
elif [[ "$MODE" == "suboptimal_rich" ]]; then
  HOPPER_EXPERT=1
  HOPPER_IMPERFECT="[50,250]"
  OTHER_EXPERT=1
  OTHER_IMPERFECT="[5,500]"
  MODE_TAG="suboptimal_rich"
else
  echo "Invalid mode: $MODE. Use 'expert_rich' or 'suboptimal_rich'" >&2
  exit 2
fi

H="$TMPDIR/hopper_${MODE_TAG}_seed${SEED}.sh"
mkcmd "$H" 0 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_250.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/hopper.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --expert_num_traj ${HOPPER_EXPERT} \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], ${HOPPER_IMPERFECT})' \\
  --critic_lr 1e-4"

A="$TMPDIR/ant_${MODE_TAG}_seed${SEED}.sh"
mkcmd "$A" 1 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Ant \\
  --xml_path=./env/ant_target.xml \\
  --dataset_file_names=\"['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/ant.pickle \\
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \\
  --expert_num_traj ${OTHER_EXPERT} \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], ${OTHER_IMPERFECT})'"

C="$TMPDIR/cheetah_${MODE_TAG}_seed${SEED}.sh"
mkcmd "$C" 2 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=HalfCheetah \\
  --xml_path=./env/cheetah_target.xml \\
  --dataset_file_names=\"['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/cheetah.pickle \\
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \\
  --expert_num_traj ${OTHER_EXPERT} \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], ${OTHER_IMPERFECT})'"

L="$TMPDIR/lift_${MODE_TAG}_seed${SEED}.sh"
mkcmd "$L" 3 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Lift \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Lift_UR5e_5_233.24838385195758_500.0.npz', 'Lift_UR5e_400_203.1412076563412_500.0.npz', 'Lift_UR5e_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/lift.pickle \\
  --flow_model_path=./flow_model/lift/model/lift_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/lift/model_action/lift_flow_seed1.pt \\
  --expert_num_traj ${OTHER_EXPERT} \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], ${OTHER_IMPERFECT})'"

D="$TMPDIR/door_${MODE_TAG}_seed${SEED}.sh"
mkcmd "$D" 4 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Door \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Door_UR5e_5_218.3826844253832_500.0.npz', 'Door_UR5e_400_212.19304507548483_500.0.npz', 'Door_UR5e_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/door.pickle \\
  --flow_model_path=./flow_model/door/model/door_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/door/model_action/door_flow_seed1.pt \\
  --expert_num_traj ${OTHER_EXPERT} \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], ${OTHER_IMPERFECT})'"

W="$TMPDIR/wipe_${MODE_TAG}_seed${SEED}.sh"
mkcmd "$W" 5 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Wipe \\
  --env_robot=UR5e \\
  --src_env_robot=Panda \\
  --dataset_file_names=\"['Wipe_UR5e_5_101.32457334196555_500.0.npz', 'Wipe_UR5e_400_100.22506601753761_500.0.npz', 'Wipe_UR5e_random_250.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/wipe.pickle \\
  --expert_num_traj ${HOPPER_EXPERT} \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], ${HOPPER_IMPERFECT})' \\
  --critic_lr 1e-4"

new_win_run "hopper" "$H" "tfboard/exp6_sensitivity/hopper_${MODE_TAG}_seed${SEED}"
new_win_run "ant" "$A" "tfboard/exp6_sensitivity/ant_${MODE_TAG}_seed${SEED}"
new_win_run "cheetah" "$C" "tfboard/exp6_sensitivity/cheetah_${MODE_TAG}_seed${SEED}"
new_win_run "lift" "$L" "tfboard/exp6_sensitivity/lift_${MODE_TAG}_seed${SEED}"
new_win_run "door" "$D" "tfboard/exp6_sensitivity/door_${MODE_TAG}_seed${SEED}"
new_win_run "wipe" "$W" "tfboard/exp6_sensitivity/wipe_${MODE_TAG}_seed${SEED}"

tmux kill-window -t 0 2>/dev/null || true
tmux select-window -t "${SESSION}:1"

if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
