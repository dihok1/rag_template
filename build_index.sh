#!/bin/bash
# Build index script for local or Docker use

set -e

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "Building index from: ${KNOWLEDGE_BASE_PATH}"
echo "Saving to: ${INDEX_PATH}"

python -m app.rag.index_builder
