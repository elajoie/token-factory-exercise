#!/usr/bin/env bash
# Block until the vLLM server answers /health (default: 15 min timeout).
PORT=${PORT:-8000}
for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null; then echo "server ready"; exit 0; fi
  sleep 5
done
echo "server did not become ready" >&2; exit 1
