#!/usr/bin/env bash
# Start the listening-study app (http://127.0.0.1:8770), optionally with a public Cloudflare URL for a pilot.
#   ./scripts/start_study.sh            # local only
#   ./scripts/start_study.sh --public   # also open a trycloudflare.com quick tunnel (pilot only: URL changes each run)
# Participants need a link key:  python -m experiment.links add pilot   -> append ?k=<key> to the URL.
# Ctrl-C stops everything. Env: PORT (default 8770), CONDA_ENV (default carnatic-mir), STUDY_ADMIN_TOKEN (optional).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8770}"
CONDA_ENV="${CONDA_ENV:-carnatic-mir}"
PUBLIC=0
[[ "${1:-}" == "--public" ]] && PUBLIC=1
mkdir -p logs

PY="$(conda run -n "$CONDA_ENV" which python 2>/dev/null || command -v python3)"
PATH="$(dirname "$PY"):$PATH"     # ffmpeg etc. from the same env

[[ -n "$(ls study/snippets/*.mp3 2>/dev/null)" ]] || { echo "no snippets yet: python -m experiment.snippets suggest && python -m experiment.snippets cut"; exit 1; }
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT already in use; stop the old server first (lsof -iTCP:$PORT)"; exit 1
fi

PIDS=()
cleanup() { echo; echo "stopping..."; kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

PORT="$PORT" "$PY" -m experiment.server > logs/study.log 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break
  kill -0 "${PIDS[0]}" 2>/dev/null || { echo "study server failed to start:"; tail -20 logs/study.log; exit 1; }
  sleep 1
done
BASE="http://127.0.0.1:$PORT"

if [[ $PUBLIC == 1 ]]; then
  command -v cloudflared >/dev/null || { echo "install cloudflared: brew install cloudflared"; exit 1; }
  cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > logs/study_tunnel.log 2>&1 &
  PIDS+=($!)
  URL=""
  for _ in $(seq 1 30); do
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' logs/study_tunnel.log | head -1 || true)"
    [[ -n "$URL" ]] && break; sleep 1
  done
  [[ -n "$URL" ]] || { echo "tunnel failed:"; tail -20 logs/study_tunnel.log; exit 1; }
  BASE="$URL"
fi

echo "study server running (log: logs/study.log). Participant links:"
"$PY" -m experiment.links list | sed "s#: /#: $BASE/#"
echo "export data any time:  python -m experiment.export   -> study/data/export/"
echo "Ctrl-C to stop."
wait
