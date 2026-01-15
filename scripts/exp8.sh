#!/usr/bin/env bash
# Experiment 8: Off-Dynamics Experiments - Figure D.12
set -euo pipefail

SESSION="exp8_offdyn"
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

python train_il_offdynamics.py \\
$* \\
  --tb_path=\${TB_PATH} \\
  --seed \${SEED}
EOF
  chmod +x "$out"
}

new_win_run() {
  local name="$1" file="$2" tb="$3"
  tmux new-window -t "${SESSION}:" -n "$name" "sleep 2; SEED=${SEED} TB_PATH=${tb} bash '$file'"
  tmux set-option -w remain-on-exit on
}

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
fi

TMPDIR="${TMPDIR:-/tmp/exp8}"
mkdir -p "$TMPDIR"

H1="$TMPDIR/hopper_avatar_${SEED}.sh"
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

H2="$TMPDIR/hopper_smodice_${SEED}.sh"
mkcmd "$H2" 1 \
  "--algorithm=smodice \\
  --env_id=Hopper \\
  --dataset_file_names=\"['hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_216_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,50])' \\
  --critic_lr 1e-4 \\
  --disc_type learned"

H3="$TMPDIR/hopper_gwil_${SEED}.sh"
mkcmd "$H3" 2 \
  "--algorithm=gwil \\
  --env_id=Hopper \\
  --dataset_file_names=\"['hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_216_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,50])' \\
  --critic_lr 1e-4"

H4="$TMPDIR/hopper_igdf_${SEED}.sh"
mkcmd "$H4" 3 \
  "--algorithm=igdf \\
  --env_id=Hopper \\
  --dataset_file_names=\"['hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_4_4357.64990234375_1249.75.npz', 'hopper_gravity2_216_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [5,50])' \\
  --critic_lr 1e-4 \\
  --batch_size=256"

A1="$TMPDIR/ant_avatar_${SEED}.sh"
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

A2="$TMPDIR/ant_smodice_${SEED}.sh"
mkcmd "$A2" 5 \
  "--algorithm=smodice \\
  --env_id=Ant \\
  --dataset_file_names=\"['ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_100_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])' \\
  --critic_lr 1e-4 \\
  --disc_type learned"

A3="$TMPDIR/ant_gwil_${SEED}.sh"
mkcmd "$A3" 6 \
  "--algorithm=gwil \\
  --env_id=Ant \\
  --dataset_file_names=\"['ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_100_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])' \\
  --critic_lr 1e-4"

A4="$TMPDIR/ant_igdf_${SEED}.sh"
mkcmd "$A4" 7 \
  "--algorithm=igdf \\
  --env_id=Ant \\
  --dataset_file_names=\"['ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_4_9965.572265625_1249.75.npz', 'ant_friction0.1_100_random.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [1,100])' \\
  --critic_lr 1e-4 \\
  --batch_size=256"

new_win_run "h_avatar" "$H1" "tfboard/exp8_offdyn/hopper_avatar_seed${SEED}"
new_win_run "h_smodice" "$H2" "tfboard/exp8_offdyn/hopper_smodice_seed${SEED}"
new_win_run "h_gwil" "$H3" "tfboard/exp8_offdyn/hopper_gwil_seed${SEED}"
new_win_run "h_igdf" "$H4" "tfboard/exp8_offdyn/hopper_igdf_seed${SEED}"
new_win_run "a_avatar" "$A1" "tfboard/exp8_offdyn/ant_avatar_seed${SEED}"
new_win_run "a_smodice" "$A2" "tfboard/exp8_offdyn/ant_smodice_seed${SEED}"
new_win_run "a_gwil" "$A3" "tfboard/exp8_offdyn/ant_gwil_seed${SEED}"
new_win_run "a_igdf" "$A4" "tfboard/exp8_offdyn/ant_igdf_seed${SEED}"

tmux kill-window -t 0 2>/dev/null || true
tmux select-window -t "${SESSION}:1"

if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
