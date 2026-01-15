#!/usr/bin/env bash
set -euo pipefail

SESSION="gwil"
ATTACH=0

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session|-s) SESSION="$2"; shift 2 ;;
    --no-attach) ATTACH=0; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

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
  --tb_path=\${TB_PATH}
EOF
  chmod +x "$out"
}

new_win_run() {
  # $1 name, $2 script path, $3 tb_path
  local name="$1" file="$2" tb="$3"
  # Wait 60s after the terminal is created before launching the experiment.
  tmux new-window -t "${SESSION}:" -n "$name" "sleep 10; TB_PATH=${tb} bash '$file'"
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
tmux set-window-option -g remain-on-exit on

# --- create per-exp temp scripts ---
TMPDIR="${TMPDIR:-/tmp/gwil}"
mkdir -p "$TMPDIR"

H10="$TMPDIR/exp_h1_seed0.sh"
mkcmd "$H10" 0 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4 \\
  --seed 0"
H11="$TMPDIR/exp_h1_seed1.sh"
mkcmd "$H11" 1 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4 \\
  --seed 1"
H12="$TMPDIR/exp_h1_seed2.sh"
mkcmd "$H12" 2 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4 \\
  --seed 2"
H13="$TMPDIR/exp_h1_seed3.sh"
mkcmd "$H13" 3 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4 \\
  --seed 3"
H14="$TMPDIR/exp_h1_seed4.sh"
mkcmd "$H14" 6 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 1 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [10,50])' \\
  --critic_lr 1e-4 \\
  --seed 4"
# -----------------------------------------------------------
H20="$TMPDIR/exp_h2_seed0.sh"
mkcmd "$H20" 0 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_400_3075.603734081881_997.5925.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 10 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [400,400])' \\
  --critic_lr 1e-4 \\
  --seed 0"
H21="$TMPDIR/exp_h2_seed1.sh"
mkcmd "$H21" 1 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_400_3075.603734081881_997.5925.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 10 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [400,400])' \\
  --critic_lr 1e-4 \\
  --seed 1"
H22="$TMPDIR/exp_h2_seed2.sh"
mkcmd "$H22" 2 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_400_3075.603734081881_997.5925.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 10 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [400,400])' \\
  --critic_lr 1e-4 \\
  --seed 2"
H23="$TMPDIR/exp_h2_seed3.sh"
mkcmd "$H23" 3 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_400_3075.603734081881_997.5925.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 10 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [400,400])' \\
  --critic_lr 1e-4 \\
  --seed 3"
H24="$TMPDIR/exp_h2_seed4.sh"
mkcmd "$H24" 4 \
  "--env_is_gym=0 \\
  --algorithm=gwil \\
  --env_id=Hopper \\
  --xml_path=./env/hopper_target.xml \\
  --dataset_file_names=\"['target_hopper_400_3075.603734081881_997.5925.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_500.npz']\" \\
  --load_hdf5_dataset=0 \\
  --log_interval=10000 \\
  --expert_num_traj 10 \\
  --imperfect_dataset_default_info '([\"expert-v2\",\"random-v2\"], [400,400])' \\
  --critic_lr 1e-4 \\
  --seed 4"


# --- open windows and run ---
# new_win_run "h10" "$H10" "tfboard/gwil/hopper_set1_seed0"
# new_win_run "h11" "$H11" "tfboard/gwil/hopper_set1_seed1"
# new_win_run "h12" "$H12" "tfboard/gwil/hopper_set1_seed2"
# new_win_run "h13" "$H13" "tfboard/gwil/hopper_set1_seed3"
# new_win_run "h14" "$H14" "tfboard/gwil/hopper_set1_seed4"
new_win_run "h20" "$H20" "tfboard/gwil/hopper_set2_seed0"
new_win_run "h21" "$H21" "tfboard/gwil/hopper_set2_seed1"
new_win_run "h22" "$H22" "tfboard/gwil/hopper_set2_seed2"
# new_win_run "h23" "$H23" "tfboard/gwil/hopper_set2_seed3"
# new_win_run "h24" "$H24" "tfboard/gwil/hopper_set2_seed4"

# focus back to monitor
# tmux select-window -t "${SESSION}:0"
# tmux kill-window -t 0
# tmux select-window -t "${SESSION}:1"
# rm /home/seanma0627/src/CDIL/\$*

# attach/switch
if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
