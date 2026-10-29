#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAIR="${1:?Usage: bash render_pair.sh PAIR_DIRECTORY 2000|5000}"
END="${2:?Specify 2000 or 5000}"
[[ "$END" = 2000 || "$END" = 5000 ]] || { echo 'Choose 2000 or 5000' >&2;exit 2; }
REPO="${HKTEX_REPO:-/root/hktex-work/hktex}"
: "${CONDA_PREFIX:?Activate hktex first}"
python "$HERE/compare_pair.py" --pair "$PAIR" --until "$END"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
printf -v STAGE 's%05d' "$END"
for ARM in linear loggamma; do
  RUN="$PAIR/$STAGE/$ARM";OUT="$RUN/native-1024-$STAMP"
  env MPLBACKEND=Agg LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
    python "$REPO/hktex-image-training-observed/render_native_images.py" \
    --repo "$REPO" --recording "$RUN/observer" --snapshots final --output "$OUT" \
    --views view_000,view_002 --size 1024 --spp 256 --tile-size 32 --point-batch 512 --seed 21931 --no-denoise
  echo "[VIEWER] $OUT/index.html"
done
