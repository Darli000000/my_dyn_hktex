#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UPDATES="${1:?Usage: bash render_pair.sh 50|1000|5000 PAIR_DIRECTORY [size] [final|all]}"
PAIR="${2:?Specify pair directory}"
SIZE="${3:-512}"
SNAPSHOTS="${4:-final}"
case "$SNAPSHOTS" in final|all) ;; *) exit 2;; esac
case "$UPDATES" in 50|1000|5000) ;; *) exit 2;; esac
case "$SIZE" in 256|512|1024|2048) ;; *) echo 'size must divide 2048' >&2;exit 2;; esac
printf -v STAGE 's%05d' "$UPDATES"
REPO="${HKTEX_REPO:-/root/hktex-work/hktex}"
: "${CONDA_PREFIX:?Activate hktex first}"
python "$HERE/compare_pair.py" --pair "$PAIR" --stage "$STAGE"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
for ARM in linear loggamma; do
  RUN="$PAIR/$STAGE/$ARM"
  OUT="$RUN/native-$SIZE-$STAMP"
  env MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
    LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
    python "$REPO/hktex-image-training-observed/render_native_images.py" \
      --repo "$REPO" --recording "$RUN/observer" --snapshots "$SNAPSHOTS" --output "$OUT" \
      --views view_000,view_002 --size "$SIZE" --spp 128 --tile-size 32 \
      --point-batch 512 --seed 21931 --no-denoise \
    2>&1 | tee "$PAIR/logs/$STAGE-$ARM-render-$STAMP.log"
  echo "[VIEWER] $OUT/index.html"
done
