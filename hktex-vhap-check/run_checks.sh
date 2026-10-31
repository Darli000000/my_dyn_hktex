#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXPORT="${EXPORT:-/root/hktex-work/data/vhap-export/018_EMO-1-shout+laugh-20261029_epoch30}"
VHAP_ROOT="${VHAP_ROOT:-/root/hktex-work/VHAP}"
FRAME="${FRAME:-0}"
GPU="${GPU:-0}"
RUN="${RUN:-$REPO/outputs/vhap-stage12-$(date +%Y%m%d-%H%M%S)-$$}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/outputs/vhap-check-cache}"
VHAP_ENV="${VHAP_ENV:-VHAP}"
HKTEX_ENV="${HKTEX_ENV:-hktex}"
if [[ -e "$RUN" ]]; then
  echo "Choose a new RUN directory: $RUN" >&2
  exit 1
fi
mkdir -p "$RUN"
exec > >(tee "$RUN/run.log") 2>&1
trap 'code=$?; if (( code != 0 )); then echo "[FAILED] exit=$code; keep $RUN/run.log and any report.json"; fi' EXIT
echo "[OUTPUT] $RUN"
echo "[INPUT] $EXPORT / frame $FRAME"
# Resolve the frame mesh from metadata instead of assuming its filename.
MESH="$(conda run -n "$VHAP_ENV" python -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); rows=[r for r in json.loads((p/"transforms.json").read_text())["frames"] if r["timestep_index"]==int(sys.argv[2])]; paths={r["mesh_path"] for r in rows}; assert len(paths)==1, "missing/ambiguous timestep"; print((p/paths.pop()).resolve())' "$EXPORT" "$FRAME")"
conda run --no-capture-output -n "$VHAP_ENV" python "$SCRIPT_DIR/export_regions.py" \
  --vhap "$VHAP_ROOT" --mesh "$MESH" --output "$RUN/regions.npz"
conda run --no-capture-output -n "$HKTEX_ENV" python "$SCRIPT_DIR/prepare.py" \
  --export "$EXPORT" --frame "$FRAME" --regions "$RUN/regions.npz" --output "$RUN/dataset"
conda run --no-capture-output -n "$HKTEX_ENV" python "$SCRIPT_DIR/check_gpu.py" \
  --repo "$REPO" --dataset "$RUN/dataset" --output "$RUN/gpu" --cache-root "$CACHE_ROOT" --gpu "$GPU"
echo "[DONE] Review $RUN/gpu/report.json and $RUN/dataset/roi_contact_sheet.jpg"
echo "These checks do not start full training or repair the mesh."
