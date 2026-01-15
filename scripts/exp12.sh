#!/usr/bin/env bash
# Experiment 12: AdaptDICE vs GWIL with single source expert trajectory - Figure D.10
set -euo pipefail

SESSION="exp12_single"
ATTACH=0
SEED=""

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

TMPDIR="${TMPDIR:-/tmp/exp12}"
mkdir -p "$TMPDIR"

H_A="$TMPDIR/hopper_avatar_single_seed${SEED}.sh"
mkcmd "$H_A" 0 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=pretrained_models/new_models/hopper_source_1.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4"

H_G="$TMPDIR/hopper_gwil_single_seed${SEED}.sh"
mkcmd "$H_G" 1 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4"

A_A="$TMPDIR/ant_avatar_single_seed${SEED}.sh"
mkcmd "$A_A" 2 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Ant \\
  --xml_path=./env/ant_target.xml \\
  --dataset_file_names=\"['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=pretrained_models/new_models/ant_source_1.pickle \\
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

A_G="$TMPDIR/ant_gwil_single_seed${SEED}.sh"
mkcmd "$A_G" 3 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Ant \\
  --xml_path=./env/ant_target.xml \\
  --dataset_file_names=\"['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

C_A="$TMPDIR/cheetah_avatar_single_seed${SEED}.sh"
mkcmd "$C_A" 4 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=HalfCheetah \\
  --xml_path=./env/cheetah_target.xml \\
  --dataset_file_names=\"['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=pretrained_models/new_models/cheetah_source_1.pickle \\
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

C_G="$TMPDIR/cheetah_gwil_single_seed${SEED}.sh"
mkcmd "$C_G" 5 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=HalfCheetah \\
  --xml_path=./env/cheetah_target.xml \\
  --dataset_file_names=\"['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

new_win_run "h_avatar" "$H_A" "tfboard/exp12_single/hopper_avatar_seed${SEED}"
new_win_run "h_gwil" "$H_G" "tfboard/exp12_single/hopper_gwil_seed${SEED}"
new_win_run "a_avatar" "$A_A" "tfboard/exp12_single/ant_avatar_seed${SEED}"
new_win_run "a_gwil" "$A_G" "tfboard/exp12_single/ant_gwil_seed${SEED}"
new_win_run "c_avatar" "$C_A" "tfboard/exp12_single/cheetah_avatar_seed${SEED}"
new_win_run "c_gwil" "$C_G" "tfboard/exp12_single/cheetah_gwil_seed${SEED}"

tmux kill-window -t 0 2>/dev/null || true
tmux select-window -t "${SESSION}:1"

if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
