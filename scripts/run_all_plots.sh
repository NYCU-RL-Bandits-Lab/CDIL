#!/usr/bin/env bash
set -euo pipefail

# Get script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

experiments=( ant_set1 ant_set2 ant_set3 cheetah_set1 cheetah_set2 cheetah_set3 \
               door_set1 door_set2 door_set3 hopper_set1 hopper_set2 hopper_set3 \
               lift_set1 lift_set2 lift_set3 wipe_set1 wipe_set2 wipe_set3 )

seeds=(seed0 seed1 seed2 seed3 seed4)

tag="Test average return"
title_suffix="Set"
out_dir_base="${PROJECT_ROOT}/plots"

mkdir -p "$out_dir_base"

for exp in "${experiments[@]}"; do
  # Build list of all seed directories for this experiment
  seed_dirs=()
  for seed in "${seeds[@]}"; do
    seed_dirs+=("${PROJECT_ROOT}/sensitivity_scan_tfboard/${exp}_${seed}")
  done

  outf="${out_dir_base}/${exp}_mean_band.png"

  echo "Generating plot for ${exp} (all seeds)..."

  python "${SCRIPT_DIR}/tb_plot_5seeds.py" \
    "${seed_dirs[@]}" \
    --tag "$tag" \
    --title "${exp^} ${title_suffix}" \
    --out "$outf"
done

