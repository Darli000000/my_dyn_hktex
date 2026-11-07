#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
pair="${PAIR:-$repo/outputs/real-face-full-pair-20261103-045946-594464}"
old="${OLD_PAIR:-$repo/outputs/real-face-adc-pair-20261031-034956-419984}"
src="${VHAP_RUN:-/root/hktex-work/VHAP/output/018_EMO-1-shout+laugh_DS4/2026-10-29_10-58-21}"
obj="${OBJ:-$src/eval_30/mesh/frame_00000.obj}"
if [[ ! -f "$obj" && -z "${OBJ:-}" && -d "$src" ]]; then
  mapfile -t matches < <(find "$src" -path '*/eval_30/mesh/frame_00000.obj')
  if [[ "${#matches[@]}" == 1 ]]; then obj="${matches[0]}"; fi
fi
[[ -f "$obj" ]] || { echo "Set OBJ to the full eval_30/mesh/frame_00000.obj path"; exit 1; }
out="${OUT:-$pair/uv-review-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir "$out"
args=(--repo "$repo" --gpu "${GPU:-0}" --out "$out")
{
  echo "[OUTPUT] $out"
  if [[ "${SKIP_OLD:-0}" != 1 ]]; then python "$here/review.py" "${args[@]}" --pair "$old" --old; fi
  python "$here/review.py" "${args[@]}" --pair "$pair" --obj "$obj"
  echo "[DONE] $out"
} 2>&1 | tee "$out/run.log"
