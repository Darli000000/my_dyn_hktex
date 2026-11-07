#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
SOURCE_DATASET="${SOURCE_DATASET:-$REPO/outputs/real-face-eyes-20261031-024447-417163/dataset}"
CONFIG="${CONFIG:-$HERE/config.json}"
HKTEX_ENV="${HKTEX_ENV:-hktex}"
GPU="${GPU:-0}"
RUN="${RUN:-$REPO/outputs/real-face-full-pair-$(date +%Y%m%d-%H%M%S)-$$}"
if [[ -e "$RUN" ]]; then echo "RUN must be new: $RUN" >&2; exit 1; fi
mkdir -p "$RUN"
trap 'code=$?; if ((code != 0)); then echo "[STOPPED] exit=$code; inspect $RUN/run.log and the stage report.json"; fi' EXIT
run_python() { conda run --no-capture-output -n "$HKTEX_ENV" python "$@"; }
run_all() {
  printf '[OUTPUT] %s\n' "$RUN"
  if [[ -n "${PREPARED_DATASET:-}" ]]; then
    TRAIN_DATASET="$PREPARED_DATASET"
  else
    run_python "$HERE/apply_masks.py" --source "$SOURCE_DATASET" --output "$RUN/dataset"
    TRAIN_DATASET="$RUN/dataset"
  fi
  run_python "$HERE/experiment.py" setup --repo "$REPO" --pair "$RUN" --dataset "$TRAIN_DATASET" --config "$CONFIG" --gpu "$GPU"
  run_python "$HERE/experiment.py" smoke --repo "$REPO" --pair "$RUN" --gpu "$GPU"
  run_python "$HERE/experiment.py" train --arm baseline --repo "$REPO" --pair "$RUN" --gpu "$GPU"
  run_python "$HERE/experiment.py" train --arm adc --repo "$REPO" --pair "$RUN" --gpu "$GPU"
  run_python "$HERE/experiment.py" compare --repo "$REPO" --pair "$RUN"
  printf '[DONE] %s/comparison.json\n' "$RUN"
}
run_all 2>&1 | tee "$RUN/run.log"
