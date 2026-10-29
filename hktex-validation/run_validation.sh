#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${HKTEX_REPO:-/root/hktex-work/hktex}"
RUN="${HKTEX_RUN:-$REPO/outputs/image-training/full-reference-20261025-061226}"
DATASET="${HKTEX_DATASET:-/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340}"
PROFILE="${1:-quick}"
if [[ "$PROFILE" != quick && "$PROFILE" != confirm ]]; then
  echo 'Usage: bash run_validation.sh [quick|confirm] [extra validate.py arguments]' >&2
  exit 1
fi
if [[ $# -gt 0 ]]; then shift; fi
[[ -f "$RUN/checkpoint.pt" ]] || { echo "Missing checkpoint: $RUN/checkpoint.pt; set HKTEX_RUN to its run directory." >&2; exit 1; }
[[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]] || {
  echo 'Activate the hktex environment; its lib/libstdc++.so.6 is required.' >&2; exit 1;
}
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
OUT="$REPO/outputs/pipeline-validation/$PROFILE-$STAMP"
mkdir -p "$REPO/outputs/pipeline-validation" /root/hktex-work/test-logs
LOG="/root/hktex-work/test-logs/pipeline-validation-$PROFILE-$STAMP.log"
ARGS=(--spp 64 --repeats 4 --grad-spp 64 --grad-repeats 3)
if [[ "$PROFILE" == confirm ]]; then
  ARGS=(--spp 256 --repeats 8 --grad-spp 256 --grad-repeats 6)
fi
echo "[INPUT RUN] $RUN"
echo "[OUTPUT] $OUT"
echo "[LOG] $LOG"
env MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python "$HERE/validate.py" --repo "$REPO" --run "$RUN" --dataset "$DATASET" \
  --output "$OUT" "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"
