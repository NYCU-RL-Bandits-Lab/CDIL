#!/usr/bin/env bash
# Experiment 11: Source Data Scaling (1, 100, 400 expert trajectories) - Figure D.11
set -euo pipefail

SESSION="exp11_source"
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

TMPDIR="${TMPDIR:-/tmp/exp11}"
mkdir -p "$TMPDIR"

H1="$TMPDIR/hopper_src1_seed${SEED}.sh"
mkcmd "$H1" 0 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=pretrained_models/new_models/hopper_source_1.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --critic_lr 1e-4 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])'"

H100="$TMPDIR/hopper_src100_seed${SEED}.sh"
mkcmd "$H100" 1 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=pretrained_models/new_models/hopper_source_100.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --critic_lr 1e-4 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])'"

H400="$TMPDIR/hopper_src400_seed${SEED}.sh"
mkcmd "$H400" 2 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/hopper.pickle \\
  --flow_model_path=./flow_model/hopper/model/hopper_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/hopper/model_action/hopper_flow_seed1.pt \\
  --critic_lr 1e-4 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])'"

A1="$TMPDIR/ant_src1_seed${SEED}.sh"
mkcmd "$A1" 3 \
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

A100="$TMPDIR/ant_src100_seed${SEED}.sh"
mkcmd "$A100" 4 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=Ant \\
  --xml_path=./env/ant_target.xml \\
  --dataset_file_names=\"['target_ant_5_4920.728662537093_848.0.npz', 'target_ant_400_5718.754833834492_987.1325.npz', 'target_Ant-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=pretrained_models/new_models/ant_source_100.pickle \\
  --flow_model_path=./flow_model/ant/model/ant_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/ant/model_action/ant_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

A400="$TMPDIR/ant_src400_seed${SEED}.sh"
mkcmd "$A400" 5 \
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

new_win_run "h_src1" "$H1" "tfboard/exp11_source/hopper_src1_seed${SEED}"
new_win_run "h_src100" "$H100" "tfboard/exp11_source/hopper_src100_seed${SEED}"
new_win_run "h_src400" "$H400" "tfboard/exp11_source/hopper_src400_seed${SEED}"
new_win_run "a_src1" "$A1" "tfboard/exp11_source/ant_src1_seed${SEED}"
new_win_run "a_src100" "$A100" "tfboard/exp11_source/ant_src100_seed${SEED}"
new_win_run "a_src400" "$A400" "tfboard/exp11_source/ant_src400_seed${SEED}"

tmux kill-window -t 0 2>/dev/null || true
tmux select-window -t "${SESSION}:1"

if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
