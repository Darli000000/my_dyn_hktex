#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-check}"
case "$MODE" in
  check|smoke|baseline|adc|full) ;;
  *) echo 'Usage: bash run_training.sh check|smoke|baseline|adc|full [dataset-directory]' >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Positional argument, not a generic HKTEX_MESH environment variable.
DATASET="${2:-/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340}"
REPO='/root/hktex-work/hktex'
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
OUT="$REPO/outputs/image-training/$MODE-$STAMP"
LOG="/root/hktex-work/test-logs/image-$MODE-$STAMP.log"
mkdir -p /root/hktex-work/test-logs
: "${CONDA_PREFIX:?Activate the hktex conda environment first}"
export MPLBACKEND=Agg
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
if [[ "$MODE" == check ]]; then
  ARGS=(check)
else
  ARGS=(train --profile "$MODE")
  if [[ "$MODE" == smoke ]]; then ARGS+=(--observer-interval 10); fi
fi
{
  echo "[DATASET] $DATASET"
  echo "[OUTPUT] $OUT"
  echo "[LOG] $LOG"
  python -u "$SCRIPT_DIR/train_images.py" "${ARGS[@]}" \
    --repo "$REPO" --dataset "$DATASET" --output "$OUT" --expected-size 2048 --gpu 0
} 2>&1 | tee "$LOG"
