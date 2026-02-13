# RAG Telegram Bot Template

Шаблон Telegram-бота с RAG: отвечает на вопросы по **вашей** базе знаний. Положите документы в папку `kb`, настройте `.env` и запустите бота — локально или через Docker.

**Стек:** Telegram (aiogram), RAG (FAISS, эмбеддинги, опционально BM25 / reranker / query expansion), LLM по контексту (OpenAI-совместимый API).

Подробная архитектура: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Требования

- Python 3.11+
- Токен Telegram-бота ([@BotFather](https://t.me/BotFather))
- API-ключ OpenAI-совместимого API (Polza, OpenAI, OpenRouter и т.п.)

## Быстрый старт

1. Клонируйте репозиторий:
   ```bash
   git clone git@github.com:dihok1/rag_template.git
   cd rag_template
   ```

2. Настройте окружение:
   ```bash
   cp .env.example .env
   ```
   Заполните в `.env`: `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY` (и при необходимости `OPENAI_API_BASE`).

3. Добавьте свои документы: положите файлы **.md** или **.txt** в папку **`kb/`** (или укажите другой путь в `KNOWLEDGE_BASE_PATH`). В шаблоне уже есть пример `kb/example.md` — можно сразу тестировать.

4. Постройте индекс:
   ```bash
   ./build_index.sh
   # или
   make index
   ```

5. Запустите бота:
   ```bash
   python -m app.main
   # или
   make run
   ```

## Добавление своих документов

- **Форматы:** `.md`, `.txt` (рекурсивно по подпапкам).
- **Расположение:** по умолчанию каталог **`kb/`** в корне проекта. Можно задать свой путь в `.env`: `KNOWLEDGE_BASE_PATH=/path/to/your/docs`.
- **После добавления или изменения файлов** обязательно пересоберите индекс: `make index` или `./build_index.sh` (при Docker — `make index-docker`, затем перезапуск контейнера при необходимости).

## Деплой (Docker)

1. Скопируйте `.env.example` в `.env` и заполните токены.
2. Документы должны лежать в `./kb` (или путь из `KNOWLEDGE_BASE_PATH` в `.env`).
3. Запустите деплой:
   ```bash
   ./deploy.sh
   ```
   Скрипт соберёт образ, построит индекс из `kb` и запустит бота через docker-compose.

4. Логи: `docker-compose logs -f`

### Обновление на сервере

С локальной машины (подставьте свой хост и путь):

```bash
export DEPLOY_HOST=user@your-server.com
export DEPLOY_PATH=/path/to/rag_template
./deploy-remote.sh
```

Скрипт выполнит на сервере `git pull origin main` и `./deploy.sh`.

### Ручной деплой (Docker)

```bash
docker build -t rag-template-bot .
docker run --rm --env-file .env \
  -v "$(pwd)/kb:/app/kb:ro" \
  -v "$(pwd)/data/index:/app/data/index" \
  -e KNOWLEDGE_BASE_PATH=/app/kb \
  rag-template-bot python -m app.rag.index_builder
docker-compose up -d
```

## Конфигурация (.env)

Основные переменные:

- `TELEGRAM_BOT_TOKEN` — токен бота
- `OPENAI_API_KEY`, `OPENAI_API_BASE` — API для эмбеддингов и LLM
- `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL` — модели
- `KNOWLEDGE_BASE_PATH` — папка с документами (по умолчанию `./kb`)
- `INDEX_PATH` — каталог индекса (по умолчанию `./data/index`)
- `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_RELEVANCE_SCORE`, `RATE_LIMIT_PER_MINUTE`

Опционально: **свой системный промпт** для LLM — переменная `RAG_SYSTEM_PROMPT` (если пусто, используется встроенный универсальный промпт).

### Расширенный RAG (опционально)

- `HYBRID_SEARCH_ENABLED` — гибридный поиск (BM25 + векторный + RRF)
- `RERANKER_ENABLED` — переранжирование cross-encoder (локально или API)
- `QUERY_EXPANSION_ENABLED` — переформулировка запроса (multi-query)

Подробнее см. комментарии в `.env.example` и [ARCHITECTURE.md](ARCHITECTURE.md).

## Проверка качества поиска

Посмотреть, какие чанки подтягиваются по запросу:

```bash
python -m app.rag.check_retrieval "ваш вопрос"
```

Подбор порога релевантности (распределение score по тестовым запросам):

```bash
python -m app.rag.evaluate_relevance
```

После смены `CHUNK_SIZE`, `CHUNK_OVERLAP` или модели эмбеддингов нужно пересобрать индекс. Изменение `TOP_K` или `MIN_RELEVANCE_SCORE` — только в `.env` и перезапуск бота.

## Makefile

```bash
make build          # Собрать Docker-образ
make index          # Построить индекс локально
make index-docker   # Построить индекс в Docker
make run            # Запустить бота локально
make deploy         # Полный деплой (build + index + docker-compose up)
make stop           # Остановить контейнеры
make logs           # Просмотр логов
make clean          # Очистить индекс и остановить контейнеры
```

## Структура проекта

```
rag_template/
├── app/
│   ├── main.py              # Точка входа бота
│   ├── config.py            # Конфигурация из .env
│   └── rag/
│       ├── index_builder.py  # Индексация (очистка, чанки, эмбеддинги)
│       ├── text_cleaning.py # Нормализация текста
│       ├── retriever.py     # Поиск (векторный / гибрид, RRF, reranker)
│       ├── rrf.py           # Reciprocal Rank Fusion
│       ├── reranker.py      # Cross-encoder reranker
│       ├── query_expansion.py
│       ├── check_retrieval.py
│       ├── evaluate_relevance.py
│       └── llm.py           # Генерация ответа
├── kb/                      # База знаний: ваши .md и .txt
│   ├── README.md
│   └── example.md           # Пример документа
├── data/index/              # FAISS-индекс и метаданные (собирается локально)
├── Dockerfile
├── docker-compose.yml
├── deploy.sh
├── build_index.sh
├── deploy-remote.sh
├── Makefile
├── requirements.txt
└── .env.example              # Скопируйте в .env
```

Бот отвечает на языке пользователя, опираясь только на проиндексированные документы из `kb`.
