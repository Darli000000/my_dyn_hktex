#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAIR="${1:?Usage: bash render_smoke.sh PAIR_DIRECTORY}"
REPO="${HKTEX_REPO:-/root/hktex-work/hktex}"
: "${CONDA_PREFIX:?Activate hktex first}"
python "$HERE/compare_smoke.py" --pair "$PAIR"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
for ARM in linear loggamma; do
  RUN="$PAIR/s00450/$ARM";OUT="$RUN/native-512-$STAMP"
  env MPLBACKEND=Agg LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
    python "$REPO/hktex-image-training-observed/render_native_images.py" \
    --repo "$REPO" --recording "$RUN/observer" --snapshots final --output "$OUT" \
    --views view_000,view_002 --size 512 --spp 128 --tile-size 32 --point-batch 512 --seed 21931 --no-denoise
  echo "[VIEWER] $OUT/index.html"
done
