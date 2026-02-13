#!/bin/bash
# Build index script for local or Docker use

set -e

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "Building index from: ${KNOWLEDGE_BASE_PATH:-./kb}"
echo "Saving to: ${INDEX_PATH:-./data/index}"

python -m app.rag.index_builder
