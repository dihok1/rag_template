#!/bin/bash
# Deployment script for server

set -e

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Use KNOWLEDGE_BASE_PATH from .env or default ./kb
KB_PATH="${KNOWLEDGE_BASE_PATH:-./kb}"
if [ ! -d "$KB_PATH" ]; then
    echo "Error: knowledge base directory not found: $KB_PATH"
    echo "Create the kb directory and add .md/.txt files, or set KNOWLEDGE_BASE_PATH in .env"
    exit 1
fi

echo "Building Docker image..."
docker build -t rag-template-bot .

echo "Building index from: $KB_PATH"
docker run --rm --env-file .env \
    -v "$(realpath "$KB_PATH"):/app/kb:ro" \
    -v "$(pwd)/data/index:/app/data/index" \
    -e KNOWLEDGE_BASE_PATH=/app/kb \
    rag-template-bot python -m app.rag.index_builder

echo "Starting bot with docker-compose..."
docker-compose up -d

echo "Bot deployed! Check logs with: docker-compose logs -f"
