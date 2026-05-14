#!/bin/sh
# Entrypoint for the ollama container.
# Starts `ollama serve` in the background, waits for the API to come up,
# pulls the chat and embedding models (idempotent), then re-attaches to
# the serve process. The compose healthcheck verifies both models are
# listed before declaring the service healthy.

set -e

CHAT_MODEL="${CHAT_MODEL:-qwen2.5:3b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

ollama serve &
SERVE_PID=$!

# Wait for the local API to be reachable.
until ollama list >/dev/null 2>&1; do
    echo "[entrypoint] waiting for ollama serve..."
    sleep 1
done

echo "[entrypoint] ensuring chat model: $CHAT_MODEL"
ollama pull "$CHAT_MODEL"

echo "[entrypoint] ensuring embedding model: $EMBED_MODEL"
ollama pull "$EMBED_MODEL"

echo "[entrypoint] ollama ready"

wait "$SERVE_PID"
