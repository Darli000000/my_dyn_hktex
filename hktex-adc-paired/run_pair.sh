#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:?Usage: bash run_pair.sh all|pilot|finish ABSOLUTE_PAIR_DIRECTORY}"
PAIR="${2:?Specify pair directory}"
REPO="${HKTEX_REPO:-/root/hktex-work/hktex}"
DATASET="${HKTEX_DATASET:-/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340}"
[[ "$PAIR" = /* ]] || { echo 'Use an absolute pair directory' >&2;exit 2; }
: "${CONDA_PREFIX:?Activate hktex first}"
[[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]] || { echo 'Conda libstdc++ missing' >&2;exit 2; }
case "$MODE" in
  all|pilot)
    [[ ! -e "$PAIR" ]] || { echo 'all/pilot requires a NEW pair directory' >&2;exit 2; }
    ENDS=(2000)
    if [[ "$MODE" = all ]]; then ENDS=(2000 5000);fi
    mkdir -p "$PAIR/logs"
    ;;
  finish)
    ENDS=(5000)
    [[ ! -e "$PAIR/s05000" ]] || { echo 's05000 already exists; follow README recovery instructions' >&2;exit 2; }
    python "$HERE/compare_pair.py" --pair "$PAIR" --until 2000 --require-active
    ;;
  *) echo 'Choose all, pilot or finish' >&2;exit 2;;
esac
for END in "${ENDS[@]}"; do
printf -v STAGE 's%05d' "$END"
for ARM in linear loggamma; do
  EXTRA=()
  if [[ "$END" = 5000 ]]; then EXTRA=(--resume "$PAIR/s02000/$ARM/checkpoint.pt");fi
  env MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
    LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
    python "$HERE/run_arm.py" --arm "$ARM" --pair "$PAIR" --until "$END" \
    --repo "$REPO" --dataset "$DATASET" --output "$PAIR/$STAGE/$ARM" "${EXTRA[@]}" \
    2>&1 | tee "$PAIR/logs/$STAGE-$ARM.log"
done
python "$HERE/compare_pair.py" --pair "$PAIR" --until "$END"
done
if [[ "$MODE" = all ]]; then
  bash "$HERE/render_pair.sh" "$PAIR" 5000
  echo "[DONE] Both arms completed; comparison: $PAIR/comparison_05000.json"
fi
