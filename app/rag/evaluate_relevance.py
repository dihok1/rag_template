"""
Оценка релевантности выдачи: запуск тестовых запросов и вывод распределения score.
Помогает подобрать MIN_RELEVANCE_SCORE.
Запуск: python -m app.rag.evaluate_relevance
"""
import statistics
import sys

from app.config import MIN_RELEVANCE_SCORE, TOP_K
from app.rag.retriever import RAGRetriever

# Примеры запросов для проверки (можно расширить)
SAMPLE_QUERIES = [
    "как продвигать мобильное приложение без бюджета",
    "какие каналы трафика использовали",
    "стратегия контента для TikTok",
    "как монетизировать SaaS",
    "истории успеха с нуля",
]


def main() -> None:
    retriever = RAGRetriever()
    retriever.load()

    all_scores = []
    print(f"TOP_K={TOP_K}, MIN_RELEVANCE_SCORE={MIN_RELEVANCE_SCORE}")
    print("=" * 60)

    for q in SAMPLE_QUERIES:
        results = retriever.search(q, top_k=TOP_K, min_score=0.0)  # без фильтра — смотрим все score
        scores = [r["score"] for r in results]
        all_scores.extend(scores)
        mean_s = statistics.mean(scores) if scores else 0
        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 0
        n_above = sum(1 for s in scores if s >= MIN_RELEVANCE_SCORE)
        print(f"Q: {q[:50]}...")
        print(f"  scores: min={min_s:.4f} max={max_s:.4f} mean={mean_s:.4f}  above_threshold={n_above}/{len(scores)}")
        print()

    if all_scores:
        print("=" * 60)
        print(f"Overall: min={min(all_scores):.4f} max={max(all_scores):.4f} mean={statistics.mean(all_scores):.4f}")
        print("Рекомендация: установите MIN_RELEVANCE_SCORE между min и mean, например 0.35–0.45, чтобы отсечь слабые совпадения.")
    else:
        print("Нет результатов. Проверьте индекс и запросы.")
        sys.exit(1)


if __name__ == "__main__":
    main()
