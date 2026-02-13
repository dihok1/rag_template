#!/bin/bash
# Запуск деплоя на удалённом сервере: git pull + ./deploy.sh
# Использование:
#   export DEPLOY_HOST=user@your-server.com
#   export DEPLOY_PATH=/path/to/rag_template
#   ./deploy-remote.sh
# или:
#   DEPLOY_HOST=user@server DEPLOY_PATH=/opt/rag_template ./deploy-remote.sh

set -e

if [ -z "$DEPLOY_HOST" ] || [ -z "$DEPLOY_PATH" ]; then
    echo "Укажи переменные: DEPLOY_HOST и DEPLOY_PATH"
    echo "  DEPLOY_HOST=user@hostname  (SSH-цель)"
    echo "  DEPLOY_PATH=/path/to/rag_template  (каталог проекта на сервере)"
    echo ""
    echo "Пример:"
    echo "  DEPLOY_HOST=root@123.45.67.89 DEPLOY_PATH=/opt/rag_template ./deploy-remote.sh"
    exit 1
fi

echo "Деплой на $DEPLOY_HOST:$DEPLOY_PATH"
ssh "$DEPLOY_HOST" "cd $DEPLOY_PATH && git pull origin main && ./deploy.sh"
