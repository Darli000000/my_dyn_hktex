#!/usr/bin/env bash
# Launch a fresh 5000-step validation run; do not resume the earlier smoke model.
set -euo pipefail
: "${CONDA_PREFIX:?Activate the hktex environment first}"
HKTEX_REPO=/root/hktex-work/hktex
HKTEX_MESH=/root/hktex-work/data/objaverse/dc5a0c0c439942d58d038a10d7064c91/model.glb
HKTEX_RUNNER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hktex_checked_run_v4.py"
HKTEX_STAMP="$(date +%Y%m%d-%H%M%S)-$$"
HKTEX_CONFIG="$HKTEX_REPO/configs/apple-full-$HKTEX_STAMP.yaml"
HKTEX_LOG="/root/hktex-work/test-logs/apple-full-$HKTEX_STAMP.log"
test -f "$HKTEX_MESH"
test -f "$CONDA_PREFIX/lib/libstdc++.so.6"
HKTEX_VERSION="$(python "$HKTEX_RUNNER" --version)"
[[ "$HKTEX_VERSION" == 'HKTex checked runner 4.0.0' ]] || { echo "Unexpected runner: $HKTEX_VERSION" >&2; exit 1; }
cd "$HKTEX_REPO"
mkdir -p /root/hktex-work/test-logs
python - "$HKTEX_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/texture_hktex_knn.yaml')
opts = [opt for opt in cfg.trainer.optimizers if opt.name == 'GeodesicOpt']
assert len(opts) == 1, 'Expected exactly one GeodesicOpt'
old_lr = float(opts[0].args.lr)
opts[0].args.lr = old_lr * 0.1
cfg.data.batch_size = 4096
cfg.data.num_workers = 0
cfg.data.use_importance_sampling = True
cfg.data.importance_sampling_pool_size = 1000000
cfg.data.importance_sampling_warmup_steps = 250
cfg.trainer.model.n_sources = 1000
cfg.trainer.data_initialisation_random_ratio = 0.5
controllers = [x.density_controller_type for x in cfg.trainer.density_controllers]
assert controllers == ['density_controllers.importance_pruning', 'density_controllers.error_based_densification'], controllers
cfg.optim.iters = 5000
cfg.optim.save_logs = True
cfg.optim.save_model = True
OmegaConf.save(cfg, sys.argv[1])
print(f'Position LR: {old_lr} -> {opts[0].args.lr}; momentum: {opts[0].args.momentum}')
print('Enabled controllers:', controllers)
print('Fresh full run: 5000 iterations; 1000 initial kernels; batch 4096; IS enabled.')
print('Generated config:', sys.argv[1])
PY
echo "Log: $HKTEX_LOG"
env MPLBACKEND=Agg CUDA_LAUNCH_BLOCKING=1 \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u "$HKTEX_RUNNER" \
  --repo "$HKTEX_REPO" \
  --config "$HKTEX_CONFIG" \
  --rendering_config "$HKTEX_REPO/configs/render.yaml" \
  --gpu 0 --check-tracer --forward-inner-texture --save-previews \
  name=apple-full \
  "tag=init05_lr10_5000steps_$HKTEX_STAMP" \
  "exp_root_dir=$HKTEX_REPO/outputs" \
  "data.mesh_path=$HKTEX_MESH" \
  renderer.n_rotating_frames=60 \
  renderer.camera_config.img_width=256 \
  renderer.camera_config.img_height=256 \
  renderer.camera_config.sample_count=8 \
  renderer.ground_plane_config.activated=False \
  2>&1 | tee "$HKTEX_LOG"
