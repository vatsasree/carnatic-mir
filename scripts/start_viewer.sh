#!/usr/bin/env bash
# Start the viewer (http://127.0.0.1:8765), optionally with a public Cloudflare URL.
#   ./scripts/start_viewer.sh            # local only
#   ./scripts/start_viewer.sh --public   # also open a trycloudflare.com quick tunnel
# Ctrl-C stops everything. Env: PORT (default 8765), CONDA_ENV (default carnatic-mir).
# Note: the public URL is open to anyone who has it, including upload/process endpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8765}"
CONDA_ENV="${CONDA_ENV:-carnatic-mir}"
PUBLIC=0
[[ "${1:-}" == "--public" ]] && PUBLIC=1
mkdir -p logs

# Python from the conda env if available, else whatever python is on PATH.
PY="$(conda run -n "$CONDA_ENV" which python 2>/dev/null || command -v python3)"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT already in use; stop the old viewer first (lsof -iTCP:$PORT)"; exit 1
fi

PIDS=()
cleanup() { echo; echo "stopping..."; kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

PORT="$PORT" "$PY" -m app.server > logs/viewer.log 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break
  kill -0 "${PIDS[0]}" 2>/dev/null || { echo "viewer failed to start:"; tail -20 logs/viewer.log; exit 1; }
  sleep 1
done
echo "viewer:  http://127.0.0.1:$PORT   (compare: /compare, log: logs/viewer.log)"

if [[ $PUBLIC == 1 ]]; then
  command -v cloudflared >/dev/null || { echo "install cloudflared: brew install cloudflared"; exit 1; }
  cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > logs/cloudflared.log 2>&1 &
  PIDS+=($!)
  URL=""
  for _ in $(seq 1 30); do
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' logs/cloudflared.log | head -1 || true)"
    [[ -n "$URL" ]] && break; sleep 1
  done
  [[ -n "$URL" ]] && echo "public:  $URL   (may take ~10 s to resolve)" || { echo "tunnel failed:"; tail -20 logs/cloudflared.log; exit 1; }
fi

echo "Ctrl-C to stop."
wait
