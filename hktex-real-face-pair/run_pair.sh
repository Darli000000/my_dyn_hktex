#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
DATASET="${DATASET:-$REPO/outputs/real-face-eyes-20261031-024447-417163/dataset}"
CONFIG="${CONFIG:-$HERE/config.json}"
HKTEX_ENV="${HKTEX_ENV:-hktex}"
GPU="${GPU:-0}"
RUN="${RUN:-$REPO/outputs/real-face-adc-pair-$(date +%Y%m%d-%H%M%S)-$$}"
if [[ -e "$RUN" ]]; then echo "RUN must be new: $RUN" >&2; exit 1; fi
if [[ ! -f "$DATASET/manifest.json" ]]; then echo "Missing prepared dataset: $DATASET" >&2; exit 1; fi
mkdir -p "$RUN"
trap 'code=$?; if ((code != 0)); then echo "[STOPPED] exit=$code; inspect stage report.json and $RUN/run.log"; fi' EXIT
run_all() {
export HKTEX_STABLE_KNN=1
printf '[OUTPUT] %s\n' "$RUN"
run_python() { conda run --no-capture-output -n "$HKTEX_ENV" python "$@"; }
run_python "$HERE/experiment.py" setup --repo "$REPO" --pair "$RUN" --dataset "$DATASET" --config "$CONFIG" --gpu "$GPU"
run_python "$HERE/experiment.py" smoke --repo "$REPO" --pair "$RUN" --gpu "$GPU"
run_python "$HERE/experiment.py" train --arm baseline --repo "$REPO" --pair "$RUN" --gpu "$GPU"
run_python "$HERE/experiment.py" train --arm adc --repo "$REPO" --pair "$RUN" --gpu "$GPU"
run_python "$HERE/experiment.py" compare --repo "$REPO" --pair "$RUN"
printf '[DONE] %s/comparison.json\n' "$RUN"
}
run_all 2>&1 | tee "$RUN/run.log"
