#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAIR="${1:?Usage: bash run_smoke.sh NEW_ABSOLUTE_PAIR_DIRECTORY}"
REPO="${HKTEX_REPO:-/root/hktex-work/hktex}"
DATASET="${HKTEX_DATASET:-/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340}"
[[ "$PAIR" = /* && ! -e "$PAIR" ]] || { echo 'Use a NEW absolute pair directory' >&2;exit 2; }
: "${CONDA_PREFIX:?Activate hktex first}"
[[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]] || { echo 'Conda libstdc++ missing' >&2;exit 2; }
mkdir -p "$PAIR/logs"
for END in 250 450; do
  printf -v STAGE 's%05d' "$END"
  for ARM in linear loggamma; do
    EXTRA=()
    if [[ "$END" = 450 ]]; then EXTRA=(--resume "$PAIR/s00250/$ARM/checkpoint.pt");fi
    env MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
      LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
      python "$HERE/run_arm.py" --arm "$ARM" --pair "$PAIR" --until "$END" \
      --repo "$REPO" --dataset "$DATASET" --output "$PAIR/$STAGE/$ARM" "${EXTRA[@]}" \
      2>&1 | tee "$PAIR/logs/$STAGE-$ARM.log"
  done
done
python "$HERE/compare_smoke.py" --pair "$PAIR"
