#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/root/hktex-work/hktex}"
MESH="${MESH:-/root/hktex-work/data/SMPL_texture_samples/Textured_Mesh_samples/SMPL_male.obj}"
DATASET="${DATASET:-/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340}"
IMAGE_RUN="${IMAGE_RUN:-/root/hktex-work/hktex/outputs/image-training/adc-full-pair-20261027-024845/s05000/linear}"
GPU="${GPU:-0}"
UPDATES="${UPDATES:-5000}"
SIZE="${SIZE:-1024}"
SPP="${SPP:-72}"
VIEWS="${VIEWS:-view_000,view_002,view_005}"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
OUT="${OUT:-$REPO/outputs/original-routes-$STAMP}"
UV_GEO_SCALE="${UV_GEO_SCALE:-1}"
: "${CONDA_PREFIX:?Activate the hktex environment first}"
test -f "$MESH"
test -f "$DATASET/manifest.json"
test -f "$IMAGE_RUN/checkpoint.pt"
test ! -e "$OUT"
export MPLBACKEND=Agg
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
mkdir -p "$OUT/logs"
printf '[OUTPUT] %s\n' "$OUT"
python "$TASK_DIR/test_protocol.py" --repo "$REPO" 2>&1 | tee "$OUT/logs/preflight.log"
for ROUTE in uv ray; do
  python -u "$TASK_DIR/train_route.py" --repo "$REPO" --mesh "$MESH" --dataset "$DATASET" \
    --route "$ROUTE" --output "$OUT/$ROUTE" --updates "$UPDATES" --record-every 250 \
    --uv-geo-scale "$UV_GEO_SCALE" --gpu "$GPU" 2>&1 | tee "$OUT/logs/train-$ROUTE.log"
done
# Both trainings are saved before any potentially expensive rendering starts.
for ROUTE in uv ray; do
  python -u "$REPO/hktex-native-viewer/render_native.py" --repo "$REPO" \
    --recording "$OUT/$ROUTE/observer" --snapshots final --size "$SIZE" --spp "$SPP" \
    --frames 4 --tile-size 32 --point-batch 512 --no-denoise --gpu "$GPU" \
    --output "$OUT/$ROUTE/author-native" 2>&1 | tee "$OUT/logs/author-native-$ROUTE.log"
  python -u "$TASK_DIR/evaluate.py" --repo "$REPO" --run "$OUT/$ROUTE" --dataset "$DATASET" \
    --output "$OUT/eval-$ROUTE" --size "$SIZE" --spp "$SP P" --views "$VIEWS" --gpu "$GPU" \
    2>&1 | tee "$OUT/logs/eval-$ROUTE.log"
done
python -u "$TASK_DIR/evaluate.py" --repo "$REPO" --run "$IMAGE_RUN" --dataset "$DATASET" \
  --output "$OUT/eval-image-linear" --size "$SIZE" --spp "$SPP" --views "$VIEWS" --gpu "$GPU" \
  2>&1 | tee "$OUT/logs/eval-image-linear.log"
python "$TASK_DIR/compare.py" --evaluations "author-uv=$OUT/eval-uv" "author-ray=$OUT/eval-ray" \
  "image-linear=$OUT/eval-image-linear" --output "$OUT/comparison"
printf '[DONE] %s/comparison/comparison_raw.png\n' "$OUT"
printf '[METRICS] %s/comparison/comparison.json\n' "$OUT"
