#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  echo 'Usage: bash run_export.sh smoke|full' >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HKTEX_REPO="${HKTEX_REPO:-/root/hktex-work/hktex}"
HKTEX_MESH=/root/hktex-work/data/SMPL_texture_samples/Textured_Mesh_samples/SMPL_male.obj
HKTEX_DATA_ROOT="${HKTEX_DATA_ROOT:-/root/hktex-work/data/offline_multiview}"
HKTEX_LOG_ROOT="${HKTEX_LOG_ROOT:-/root/hktex-work/test-logs}"
HKTEX_SIZE="${HKTEX_SIZE:-256}"
HKTEX_GPU="${HKTEX_GPU:-0}"
if [[ "$MODE" == smoke ]]; then
  DEFAULT_SPP=16
else
  DEFAULT_SPP=128
fi
HKTEX_SPP="${HKTEX_SPP:-$DEFAULT_SPP}"
HKTEX_STAMP="$(date +%Y%m%d-%H%M%S)-$$"
HKTEX_OUTPUT="$HKTEX_DATA_ROOT/smpl-male-$MODE-$HKTEX_STAMP"
HKTEX_LOG="$HKTEX_LOG_ROOT/offline-$MODE-$HKTEX_STAMP.log"
mkdir -p "$HKTEX_DATA_ROOT" "$HKTEX_LOG_ROOT"
: "${CONDA_PREFIX:?Activate the hktex conda environment first}"
if [[ ! -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  echo 'Missing Conda libstdc++.so.6; check the environment before exporting.' >&2
  exit 1
fi
export MPLBACKEND=Agg
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
{
  echo "[OUTPUT] $HKTEX_OUTPUT"
  echo "[LOG] $HKTEX_LOG"
  python -u "$SCRIPT_DIR/offline_data.py" export \
    --repo "$HKTEX_REPO" --mesh "$HKTEX_MESH" --output "$HKTEX_OUTPUT" \
    --preset "$MODE" --size "$HKTEX_SIZE" --spp "$HKTEX_SPP" --gpu "$HKTEX_GPU"
  python -u "$SCRIPT_DIR/offline_data.py" verify \
    --repo "$HKTEX_REPO" --dataset "$HKTEX_OUTPUT" --gpu "$HKTEX_GPU"
  echo "[SUCCESS] $HKTEX_OUTPUT"
  echo '[NEXT] Inspect contact_sheet.png, validation.json and replay_validation.json.'
} 2>&1 | tee "$HKTEX_LOG"
