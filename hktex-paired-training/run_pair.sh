#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:?Usage: bash run_pair.sh smoke|1000|5000 PAIR_DIRECTORY}"
PAIR="${2:?Specify the same absolute pair directory at each stage}"
REPO="${HKTEX_REPO:-/root/hktex-work/hktex}"
DATASET="${HKTEX_DATASET:-/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340}"
case "$STAGE" in
  smoke) UNTIL=50; PREV='' ;;
  1000) UNTIL=1000; PREV=s00050 ;;
  5000) UNTIL=5000; PREV=s01000 ;;
  *) echo 'Stage must be smoke, 1000 or 5000' >&2;exit 2 ;;
esac
printf -v LABEL 's%05d' "$UNTIL"
[[ "$PAIR" = /* ]] || { echo 'PAIR_DIRECTORY must be absolute' >&2;exit 2; }
: "${CONDA_PREFIX:?Activate hktex first}"
[[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]] || { echo 'Missing conda libstdc++' >&2;exit 2; }
if [[ -n "$PREV" ]]; then
  # Ensure the previous two arms both completed before starting the next stage.
  python "$HERE/compare_pair.py" --pair "$PAIR" --stage "$PREV"
fi
mkdir -p "$PAIR/logs" "$PAIR/$LABEL"
for ARM in linear loggamma; do
  OUT="$PAIR/$LABEL/$ARM"
  [[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT; do not overwrite or blindly rerun the pair." >&2;exit 2; }
  EXTRA=()
  if [[ -n "$PREV" ]]; then EXTRA=(--resume "$PAIR/$PREV/$ARM/checkpoint.pt");fi
  echo "[PAIR] $ARM -> $OUT"
  env MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
    LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
    python "$HERE/run_arm_resume_fix.py" --arm "$ARM" --pair "$PAIR" --until "$UNTIL" \
      --repo "$REPO" --dataset "$DATASET" --output "$OUT" "${EXTRA[@]}" \
    2>&1 | tee "$PAIR/logs/$LABEL-$ARM.log"
done
python "$HERE/compare_pair.py" --pair "$PAIR" --stage "$LABEL"
