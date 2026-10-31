#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
EXPORT="${EXPORT:-/root/hktex-work/data/vhap-export/018_EMO-1-shout+laugh-20261029_epoch30}"
SOURCE_REGIONS="${SOURCE_REGIONS:-$REPO/outputs/vhap-stage12-20261031-012229-414642/regions.npz}"
VHAP_ROOT="${VHAP_ROOT:-/root/hktex-work/VHAP}"
HKTEX_ENV="${HKTEX_ENV:-hktex}"
VHAP_ENV="${VHAP_ENV:-VHAP}"
GPU="${GPU:-0}"
FRAME="${FRAME:-0}"
STEPS="${STEPS:-300}"
RUN="${RUN:-$REPO/outputs/real-face-eyes-$(date +%Y%m%d-%H%M%S)-$$}"
if [[ -e "$RUN" ]]; then echo "Choose a new RUN directory: $RUN" >&2; exit 1; fi
mkdir -p "$RUN"
exec > >(tee "$RUN/run.log") 2>&1
trap 'code=$?; if (( code != 0 )); then echo "[STOPPED] exit=$code; inspect $RUN/run.log and fit/preflight.json or fit/report.json"; fi' EXIT
echo "[OUTPUT] $RUN"
MESH="$(conda run -n "$HKTEX_ENV" python -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); rows=[r for r in json.loads((p/"transforms.json").read_text())["frames"] if r["timestep_index"]==int(sys.argv[2])]; paths={r["mesh_path"] for r in rows}; assert len(paths)==1, "missing/ambiguous timestep"; print((p/paths.pop()).resolve())' "$EXPORT" "$FRAME")"
if [[ -f "$SOURCE_REGIONS" ]]; then
  conda run --no-capture-output -n "$HKTEX_ENV" python "$HERE/regions.py" --mesh "$MESH" --source-regions "$SOURCE_REGIONS" --output "$RUN/regions.npz"
else
  conda run --no-capture-output -n "$VHAP_ENV" python "$HERE/regions.py" --mesh "$MESH" --vhap "$VHAP_ROOT" --output "$RUN/regions.npz"
fi
conda run --no-capture-output -n "$HKTEX_ENV" python "$HERE/prepare.py" \
  --export "$EXPORT" --frame "$FRAME" --regions "$RUN/regions.npz" --output "$RUN/dataset"
conda run --no-capture-output -n "$HKTEX_ENV" python "$HERE/pipeline.py" \
  --repo "$REPO" --dataset "$RUN/dataset" --output "$RUN/fit" --gpu "$GPU" --steps "$STEPS"
echo "[DONE] $RUN/fit/report.json"
