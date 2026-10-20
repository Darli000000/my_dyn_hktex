#!/usr/bin/env bash
# Run in the activated hktex environment. First argument is a position-LR multiplier.
# Always scales the original YAML, never a previously scaled configuration.
set -euo pipefail

: "${CONDA_PREFIX:?Activate the hktex Conda environment first}"
HKTEX_REPO=/root/hktex-work/hktex
HKTEX_MESH=/root/hktex-work/data/download/handpainted_dwarven_head.glb
HKTEX_RUNNER="$HKTEX_REPO/hktex_checked_run_v3.py"
HKTEX_SCALE="${1:-0.1}"
HKTEX_STAMP="$(date +%Y%m%d-%H%M%S)-$$"
HKTEX_CONFIG="$HKTEX_REPO/configs/dwarf-position-test-$HKTEX_STAMP.yaml"
HKTEX_LOG="/root/hktex-work/test-logs/dwarf-position-test-$HKTEX_STAMP.log"

test -f "$HKTEX_MESH"
test -f "$CONDA_PREFIX/lib/libstdc++.so.6"
HKTEX_VERSION="$(python "$HKTEX_RUNNER" --version)"
if [[ "$HKTEX_VERSION" != "HKTex checked runner 3.0.0" ]]; then
    echo "Unexpected diagnostic runner: $HKTEX_VERSION" >&2
    exit 1
fi

cd "$HKTEX_REPO"
mkdir -p /root/hktex-work/test-logs

python - "$HKTEX_SCALE" "$HKTEX_CONFIG" <<'PY'
import math
import sys
from omegaconf import OmegaConf

scale = float(sys.argv[1])
if not math.isfinite(scale) or not 0 < scale <= 1:
    raise SystemExit("Position learning-rate multiplier must be in (0, 1].")
cfg = OmegaConf.load("configs/texture_hktex_knn.yaml")
opts = [opt for opt in cfg.trainer.optimizers if opt.name == "GeodesicOpt"]
if len(opts) != 1:
    raise SystemExit("Expected exactly one GeodesicOpt entry; configuration was not saved.")
opt = opts[0]
old_lr = float(opt.args.lr)
opt.args.lr = old_lr * scale
OmegaConf.save(cfg, sys.argv[2])
print(f"Position learning rate: {old_lr} -> {opt.args.lr}", flush=True)
print(f"Momentum unchanged: {opt.args.get('momentum', 0)}", flush=True)
print(f"Generated config: {sys.argv[2]}", flush=True)
PY

echo "Log: $HKTEX_LOG"
env MPLBACKEND=Agg \
    CUDA_LAUNCH_BLOCKING=1 \
    LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
    python -u "$HKTEX_RUNNER" \
    --repo "$HKTEX_REPO" \
    --config "$HKTEX_CONFIG" \
    --gpu 0 \
    --check-tracer \
    name=dwarf-position-test \
    "tag=lr${HKTEX_SCALE}_${HKTEX_STAMP}" \
    "exp_root_dir=$HKTEX_REPO/outputs" \
    "data.mesh_path=$HKTEX_MESH" \
    data.batch_size=512 \
    data.num_workers=0 \
    data.use_importance_sampling=False \
    trainer.model.n_sources=256 \
    trainer.data_initialisation_random_ratio=0.5 \
    'trainer.density_controllers=[]' \
    optim.iters=50 \
    renderer.n_rotating_frames=2 \
    renderer.camera_config.img_width=128 \
    renderer.camera_config.img_height=128 \
    renderer.camera_config.sample_count=4 \
    renderer.ground_plane_config.activated=False \
    2>&1 | tee "$HKTEX_LOG"
