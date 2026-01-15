#!/usr/bin/env bash
set -euo pipefail

ENV_GROUP=""
SESSION="0"
ATTACH=0
SLEEP_BETWEEN=10
ROOT_DIR="$HOME/src/CDIL"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)        ENV_GROUP="$2"; shift 2 ;;
    --session|-s) SESSION="$2";  shift 2 ;;
    --no-attach)  ATTACH=0;      shift    ;;
    --sleep)      SLEEP_BETWEEN="${2:-1.5}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${ENV_GROUP:-}" || ! "$ENV_GROUP" =~ ^(mujoco|robosuite)$ ]]; then
  echo "Usage: $0 --env {mujoco|robosuite} [--session NAME] [--no-attach] [--sleep SEC]" >&2
  exit 2
fi

# Helper: conda activation (best-effort)
ensure_conda_snippet='
# Load conda if present; otherwise continue with system python
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then . "$HOME/miniconda3/etc/profile.d/conda.sh"; fi
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then . "$HOME/miniforge3/etc/profile.d/conda.sh"; fi
if command -v conda >/dev/null 2>&1; then
  conda activate cdil
else
  echo "[warn] conda not found; proceeding without explicit activation"
fi
'

# tmux monitor window (htop + nvtop)
start_monitor() {
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "mon"
  else
    tmux rename-window -t "${SESSION}:0" "mon" 2>/dev/null || true
  fi

  tmux select-window -t "${SESSION}:0"
  # top/bottom panes: split-window -v creates a top and a bottom pane
  tmux send-keys -t "${SESSION}:0.0" "htop -u \"$USER\"" C-m
  tmux split-window -t "${SESSION}:0" -v
  tmux send-keys -t "${SESSION}:0.1" "nvtop" C-m
  tmux select-layout -t "${SESSION}:0" even-vertical
}

# Small factory to create one-off run scripts
mk_run_script() {
  # $1: out path
  # $2: ARGS block (multi-line)
  local out="$1"
  local args_block="$2"
  mkdir -p "$(dirname "$out")"

  # Define local variables we need
  local RL_ROOT_DIR="${ROOT_DIR}"
  local RL_CONDA_SNIPPET="${ensure_conda_snippet}"
  local RL_ARGS_BLOCK="${args_block}"

  cat >"$out" <<SH
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU}"
cd "${RL_ROOT_DIR}"
${RL_CONDA_SNIPPET}
python train_il.py \\
  --env_is_gym=0 \\
  --algorithm=demodice \\
${RL_ARGS_BLOCK} \\
  --tb_path="\${TB_PATH}" \\
  --seed "\${SEED}"
SH

  chmod +x "$out"
}


# -------------------------------
# Per-env argument blocks (exactly what you provided, but tb_path & seed injected later)
# -------------------------------
args_for_env() {
  case "$1" in
    Hopper) cat <<'EOS'
  --env_id=Hopper \
  --xml_path=./env/hopper_target.xml \
  --dataset_file_names='["target_hopper_5_3091.0554050953983_1000.0.npz", "target_hopper_400_3075.603734081881_997.5925.npz", "target_Hopper-v3_random_50.npz"]' \
  --load_hdf5_dataset=0 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2","random-v2"], [20,50])' \
  --log_interval=10000 \
  --actor_lr=5e-5
EOS
    ;;
    Ant) cat <<'EOS'
  --env_id=Ant \
  --xml_path=./env/ant_target.xml \
  --dataset_file_names='["target_ant_5_4920.728662537093_848.0.npz", "target_ant_400_5718.754833834492_987.1325.npz", "target_Ant-v3_random_100.npz"]' \
  --load_hdf5_dataset=0 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2","random-v2"], [1,100])' \
  --log_interval=10000
EOS
    ;;
    HalfCheetah) cat <<'EOS'
  --env_id=HalfCheetah \
  --xml_path=./env/cheetah_target.xml \
  --dataset_file_names='["target_cheetah_5_14411.743096437334_1000.0.npz", "target_cheetah_400_14932.046536124828_1000.0.npz", "target_HalfCheetah-v3_random_100.npz"]' \
  --load_hdf5_dataset=0 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2","random-v2"], [1,100])' \
  --log_interval=10000
EOS
    ;;
    Wipe) cat <<'EOS'
  --env_id=Wipe \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names='["Wipe_UR5e_5_101.32457334196555_500.0.npz", "Wipe_UR5e_400_100.22506601753761_500.0.npz", "Wipe_UR5e_random_50.npz"]' \
  --load_hdf5_dataset=0 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2","random-v2"], [10,50])' \
  --log_interval=10000 \
  --critic_lr=1e-4
EOS
    ;;
    Door) cat <<'EOS'
  --env_id=Door \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names='["Door_UR5e_5_218.3826844253832_500.0.npz", "Door_UR5e_400_212.19304507548483_500.0.npz", "Door_UR5e_random_100.npz"]' \
  --load_hdf5_dataset=0 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2","random-v2"], [1,100])' \
  --log_interval=10000
EOS
    ;;
    Lift) cat <<'EOS'
  --env_id=Lift \
  --env_robot=UR5e \
  --src_env_robot=Panda \
  --dataset_file_names='["Lift_UR5e_5_233.24838385195758_500.0.npz", "Lift_UR5e_400_203.1412076563412_500.0.npz", "Lift_UR5e_random_100.npz"]' \
  --load_hdf5_dataset=0 \
  --expert_num_traj 1 \
  --imperfect_dataset_default_info='(["expert-v2","random-v2"], [1,100])' \
  --log_interval=10000
EOS
    ;;
  esac
}

# Single-letter for window names
env_letter() {
  case "$1" in
    Hopper) echo h ;;
    Ant) echo a ;;
    HalfCheetah) echo c ;;
    Wipe) echo w ;;
    Door) echo d ;;
    Lift) echo l ;;
  esac
}

# GPU mapping per your spec:
# card1: exp1 s0/1 | card2: exp2 s0/1 | card3: exp3 s0/1
# card4: exp1 s2/3 | card5: exp2 s2/3 | card6: exp3 s2/3
# card7: exp1 s4 & exp2 s4 | card8: exp3 s4
gpu_for() {
  # $1 exp_idx in {0,1,2}, $2 seed in {0..4}
  local idx="$1" seed="$2"
  case "$seed" in
    0|1) echo $((0 + idx)) ;;
    2|3) echo $((3 + idx)) ;;
    4)
      if   [[ $idx -eq 0 ]]; then echo 6
      elif [[ $idx -eq 1 ]]; then echo 6
      else                         echo 7
      fi
      ;;
  esac
}

# The three experiments for each group
if [[ "$ENV_GROUP" == "mujoco" ]]; then
  EXP_LIST=("Hopper" "Ant" "HalfCheetah")
else
  EXP_LIST=("Wipe" "Door" "Lift")
fi

# -------------------------------
# Build + launch
# -------------------------------
# start_monitor
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
  tmux set-option -g remain-on-exit on
fi

TMPDIR="${TMPDIR:-/tmp}/demodice_runs_${SESSION}"
mkdir -p "$TMPDIR"

# Prepare 15 runs (3 envs x seeds 0..4)
for i in "${!EXP_LIST[@]}"; do
  ENV_NAME="${EXP_LIST[$i]}"
  LETTER="$(env_letter "$ENV_NAME")"
  ARGS_BLOCK="$(args_for_env "$ENV_NAME")"

  for SEED in 0 1 2 3 4; do
    GPU="$(gpu_for "$i" "$SEED")"
    # TensorBoard path per-seed
    lower=$(echo "$ENV_NAME" | tr '[:upper:]' '[:lower:]')
    TB_PATH="tfboard/ablation/demodice/demodice_${lower}_seed${SEED}"

    # One run file per launch
    RUN_FILE="${TMPDIR}/${LETTER}${SEED}.sh"
    mk_run_script "$RUN_FILE" "$ARGS_BLOCK" 0

    # Create window and start the job
    WIN_NAME="${LETTER}${SEED}"
    tmux new-window -t "${SESSION}:" -n "$WIN_NAME" \
      "sleep ${SLEEP_BETWEEN}; GPU=${GPU} SEED=${SEED} TB_PATH='${TB_PATH}' bash '${RUN_FILE}'"
    # sleep "$SLEEP_BETWEEN"
  done
done

# Focus back to monitor
# tmux select-window -t "${SESSION}:0"
tmux kill-window -t 0
tmux select-window -t "${SESSION}:1"

# Attach or switch
if [[ $ATTACH -eq 1 ]]; then
  if [[ -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION"
  else
    tmux switch-client -t "$SESSION"
  fi
fi
