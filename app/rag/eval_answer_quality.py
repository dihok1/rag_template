"""
Оценка качества выдачи: запрос -> retrieval -> ответ.
Печатает запрос, топ чанки (score, источник), итоговый ответ.
Запуск: python -m app.rag.eval_answer_quality
"""
from app.config import TOP_K
from app.rag.llm import generate_answer
from app.rag.retriever import RAGRetriever

SAMPLE_QUERIES = [
    "как продвигать мобильное приложение без бюджета",
    "какие каналы трафика работают для стартапа",
    "как монетизировать SaaS",
    "советы по найму первых сотрудников",
    "как привлечь первых пользователей",
]


def main() -> None:
    retriever = RAGRetriever()
    retriever.load()

    for query in SAMPLE_QUERIES:
        print("\n" + "=" * 70)
        print("ЗАПРОС:", query)
        print("=" * 70)
        contexts = retriever.search(query, top_k=TOP_K)
        if not contexts:
            print("Чанков не найдено.\n")
            continue
        print("\n--- Топ чанки (релевантность) ---")
        for i, c in enumerate(contexts, 1):
            score = c.get("score", 0)
            path = c.get("source_path", "?")
            text = (c.get("text") or "")[:280]
            if len((c.get("text") or "")) > 280:
                text += "..."
            print(f"  {i}. score={score:.3f} | {path}")
            print(f"     {text}\n")
        try:
            answer = generate_answer(query, contexts)
            print("--- ОТВЕТ БОТА ---")
            print(answer)
        except Exception as e:
            print("--- ОШИБКА ГЕНЕРАЦИИ ---", e)
        print()


if __name__ == "__main__":
    main()
