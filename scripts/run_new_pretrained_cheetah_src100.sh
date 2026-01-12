#!/usr/bin/env bash
set -euo pipefail

SESSION="avatar"
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
  # $1: out file
  # $2: CUDA_VISIBLE_DEVICES
  # $3..: body (python command lines WITHOUT last two switches)
  local out="$1" gpu="$2"
  shift 2
  cat >"$out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${gpu}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin:/usr/lib/nvidia
# export MUJOCO_GL=egl

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
  tmux new-window -t "${SESSION}:" -n "$name" "sleep 10; SEED=${SEED} TB_PATH=${tb} bash '$file'"
  tmux set-option -w remain-on-exit on
  tmux set-option -g mouse on
}

# --- start tmux session w/ monitor window (0) ---
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" # -n "mon"    # create session & window 0
# else
#   # if session exists, ensure window 0 exists/named
#   tmux rename-window -t "${SESSION}:0" "mon" 2>/dev/null || true
fi

# split vertically and run monitors
# tmux select-window -t "${SESSION}:0"
# tmux split-window -t "${SESSION}:0" -v
# tmux select-pane -t "${SESSION}:0.0"
# sleep 10
# tmux send-keys -t "${SESSION}:0.0" "htop -u \"$USER\"" C-m
# tmux select-pane -t "${SESSION}:0.1"
# sleep 10
# tmux send-keys -t "${SESSION}:0.1" "nvtop" C-m
# tmux select-layout -t "${SESSION}:0" even-vertical
# tmux set-window-option -g remain-on-exit on

# --- create per-exp temp scripts ---
TMPDIR="${TMPDIR:-/tmp/avatar}"
mkdir -p "$TMPDIR"

C2="$TMPDIR/exp_c2_seed${SEED}.sh"
mkcmd "$C2" 3 \
  "--env_is_gym=0 \\
  --algorithm=avatar_dice \\
  --env_id=HalfCheetah \\
  --xml_path=./env/cheetah_target.xml \\
  --dataset_file_names=\"['target_cheetah_5_14411.743096437334_1000.0.npz', 'target_cheetah_400_14932.046536124828_1000.0.npz', 'target_HalfCheetah-v3_random_100.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=pretrained_models/new_models/cheetah_source_100.pickle \\
  --flow_model_path=./flow_model/cheetah/model/cheetah_flow_seed1.pt \\
  --flow_model_action_path=./flow_model/cheetah/model_action/cheetah_flow_seed1.pt \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])'"

new_win_run "c2" "$C2" "tfboard/avatardice/new_pretrained/cheetah_seed${SEED}_source100"

# focus back to monitor
# tmux select-window -t "${SESSION}:0"
tmux kill-window -t 0
tmux select-window -t "${SESSION}:1"
# rm /home/seanma0627/src/CDIL/\$*

# attach/switch
if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
