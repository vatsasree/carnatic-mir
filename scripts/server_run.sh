#!/usr/bin/env bash
# Run on the GPU server, inside the container, from the repo root:
#   bash scripts/server_run.sh                       # default: urls_new.txt -> results/h100
#   URLS=other.txt OUT=results/run2 bash scripts/server_run.sh
# Settings (env): OVERLAP (4 = model default, best), FP16 (0 = full precision), WORKERS (tracks in parallel).
set -euo pipefail
cd "$(dirname "$0")/.."
URLS=${URLS:-urls_new.txt}
OUT=${OUT:-results/h100}
WORKERS=${WORKERS:-4}

echo "== 1/4 system packages"
if ! command -v ffmpeg >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
fi
ffmpeg -version | head -1

echo "== 2/4 python packages (torch from the container is kept)"
if python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install -q --root-user-action=ignore -r requirements-colab.txt
else                                   # some vLLM images ship uv instead of pip
    uv pip install --system -q -r requirements-colab.txt
fi
python3 scripts/shard.py join          # rebuild any sharded files (e.g. model weights) pushed via git
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible: start the container with --gpus all"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
import essentia, torchcrepe, audio_separator, parselmouth  # noqa: F401  (fail early if an install broke)
print("imports ok")
PY

echo "== 3/4 analysis ($URLS -> $OUT, $WORKERS workers)"
mkdir -p "$OUT"
export CARNATIC_MIR_OUTPUTS="$OUT"
export CARNATIC_MIR_MODELS=${CARNATIC_MIR_MODELS:-models}
export CARNATIC_MIR_SEP_OVERLAP=${OVERLAP:-4}
export CARNATIC_MIR_SEP_FP16=${FP16:-0}
python3 -m carnatic_mir.cli --urls "$URLS" --workers "$WORKERS" 2>&1 | tee -a "$OUT/run.log" || true
grep -h "timings:\|failed\|Error" "$OUT"/logs/worker*.log 2>/dev/null | tail -40 || true

echo "== 4/4 pack for git"
python3 scripts/pack_results.py "$OUT"
echo "done. Now: git add $OUT && git commit -m 'H100 results' && git push"
