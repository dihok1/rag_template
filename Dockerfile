FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1

# Default: run bot. Override to run index builder, e.g.:
# docker run --env-file .env -v $(pwd)/kb:/app/kb:ro -v $(pwd)/data/index:/app/data/index rag-template-bot python -m app.rag.index_builder
CMD ["python", "-m", "app.main"]
