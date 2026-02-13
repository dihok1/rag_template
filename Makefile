.PHONY: build index run deploy stop logs clean

build:
	docker build -t rag-template-bot .

index:
	python -m app.rag.index_builder

index-docker:
	docker run --rm --env-file .env \
		-v "$$(pwd)/kb:/app/kb:ro" \
		-v "$$(pwd)/data/index:/app/data/index" \
		-e KNOWLEDGE_BASE_PATH=/app/kb \
		rag-template-bot python -m app.rag.index_builder

run:
	python -m app.main

deploy:
	./deploy.sh

stop:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	rm -rf data/index/*.faiss data/index/*.json
