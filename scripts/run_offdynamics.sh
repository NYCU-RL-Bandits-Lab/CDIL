#!/usr/bin/env bash
set -euo pipefail

SESSION="offdyn"
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

python train_il_offdynamics.py \\
$* \\
  --tb_path=\${TB_PATH} \\
  --seed \${SEED}
EOF
  chmod +x "$out"
}

new_win_run() {
  # $1 name, $2 script path, $3 tb_path
  local name="$1" file="$2" tb="$3"
  tmux new-window -t "${SESSION}:" -n "$name" "sleep 2; SEED=${SEED} TB_PATH=${tb} bash '$file'"
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

H1="$TMPDIR/offdyn_h1_${SEED}.sh"
mkcmd "$H1" 0 \
  "--algorithm=avatar_dice \\
  --env_id=Hopper \\
  --dataset_file_names=\"['hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_216_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/hopper.pickle \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4"
H2="$TMPDIR/offdyn_h2_${SEED}.sh"
mkcmd "$H2" 1 \
  "--algorithm=smodice \\
  --env_id=Hopper \\
  --dataset_file_names=\"['hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_216_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,50])' \\
  --log_interval 10000 \\
  --critic_lr 1e-4 \\
  --disc_type learned"
H3="$TMPDIR/offdyn_h3_${SEED}.sh"
mkcmd "$H3" 2 \
  "--algorithm=gwil \\
  --env_id=Hopper \\
  --dataset_file_names=\"['hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_216_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,50])' \\
  --log_interval 10000 \\
  --critic_lr 1e-4"
H4="$TMPDIR/offdyn_h4_${SEED}.sh"
mkcmd "$H4" 3 \
  "--algorithm=igdf \\
  --env_id=Hopper \\
  --dataset_file_names=\"['hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_216_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,50])' \\
  --log_interval 10000 \\
  --critic_lr 1e-4 \\
  --batch_size=256"

A1="$TMPDIR/offdyn_a1_${SEED}.sh"
mkcmd "$A1" 4 \
  "--algorithm=avatar_dice \\
  --env_id=Ant \\
  --dataset_file_names=\"['ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_100_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --pretrained_model_path=./pretrained_models/ant.pickle \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])' \\
  --critic_lr 1e-4"
A2="$TMPDIR/offdyn_a2_${SEED}.sh"
mkcmd "$A2" 5 \
  "--algorithm=smodice \\
  --env_id=Ant \\
  --dataset_file_names=\"['ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_100_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])' \\
  --log_interval 10000 \\
  --critic_lr 1e-4 \\
  --disc_type learned"
A3="$TMPDIR/offdyn_a3_${SEED}.sh"
mkcmd "$A3" 6 \
  "--algorithm=gwil \\
  --env_id=Ant \\
  --dataset_file_names=\"['ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_100_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])' \\
  --log_interval 10000 \\
  --critic_lr 1e-4"
A4="$TMPDIR/offdyn_a4_${SEED}.sh"
mkcmd "$A4" 7 \
  "--algorithm=igdf \\
  --env_id=Ant \\
  --dataset_file_names=\"['ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_100_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])' \\
  --log_interval 10000 \\
  --critic_lr 1e-4 \\
  --batch_size=256"

new_win_run "h1" "$H1" "tfboard/offdyn/hopper_avatar_seed${SEED}"
new_win_run "h2" "$H2" "tfboard/offdyn/hopper_smodice_seed${SEED}"
new_win_run "h3" "$H3" "tfboard/offdyn/hopper_gwil_seed${SEED}"
new_win_run "h4" "$H4" "tfboard/offdyn/hopper_igdf_seed${SEED}"
new_win_run "a1" "$A1" "tfboard/offdyn/ant_avatar_seed${SEED}"
new_win_run "a2" "$A2" "tfboard/offdyn/ant_smodice_seed${SEED}"
new_win_run "a3" "$A3" "tfboard/offdyn/ant_gwil_seed${SEED}"
new_win_run "a4" "$A4" "tfboard/offdyn/ant_igdf_seed${SEED}"

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
