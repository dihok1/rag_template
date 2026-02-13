"""Query expansion: multi-query reformulations for better retrieval."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL

EXPAND_SYSTEM = """Ты помогаешь переформулировать поисковые запросы. Выводи только переформулировки вопроса, по одной на строку, без нумерации и пояснений. Сохраняй смысл и язык вопроса."""

EXPAND_USER = """Переформулируй следующий вопрос {n} разными способами (разные формулировки, тот же смысл). Каждая переформулировка — одна строка.

Вопрос: {query}

Переформулировки (по одной на строку):"""


def expand_query_multi(query: str, num_variants: int = 3) -> list[str]:
    """
    Return [original_query, reformulation_2, ...] so that total length is num_variants.
    Uses LLM to generate (num_variants - 1) reformulations.
    """
    query = (query or "").strip()
    if not query:
        return [query]
    if num_variants <= 1:
        return [query]
    n_extra = num_variants - 1
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPAND_SYSTEM),
        ("human", EXPAND_USER),
    ])
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        max_tokens=256,
    )
    chain = prompt | llm
    msg = chain.invoke({"query": query, "n": n_extra})
    text = (msg.content or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:n_extra]
    result = [query]
    for ln in lines:
        if ln and ln not in result:
            result.append(ln)
    return result
