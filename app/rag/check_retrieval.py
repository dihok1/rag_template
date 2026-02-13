"""
Проверка качества поиска: по запросу выводит топ-K чанков с оценкой и источником.
Запуск: python -m app.rag.check_retrieval "твой вопрос"
       python -m app.rag.check_retrieval   # интерактивно, по одному запросу на строку
"""
import sys

from app.config import TOP_K
from app.rag.retriever import RAGRetriever


def main() -> None:
    retriever = RAGRetriever()
    retriever.load()

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        print("Введите запрос (пустая строка — выход):")
        queries = []
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            queries.append(line.strip())

    if not queries:
        print("Нет запросов.")
        return

    for query in queries:
        print(f"\n{'='*60}\nЗапрос: {query}\n{'='*60}")
        results = retriever.search(query, top_k=TOP_K)
        if not results:
            print("Ничего не найдено.")
            continue
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            path = r.get("source_path", "?")
            text = (r.get("text") or "")[:400]
            if len((r.get("text") or "")) > 400:
                text += "..."
            print(f"\n--- Чанк {i} (score={score:.4f}, источник: {path}) ---\n{text}\n")


if __name__ == "__main__":
    main()
