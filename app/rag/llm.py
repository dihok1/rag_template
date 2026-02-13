"""RAG answer generation: LangChain ChatOpenAI + structured prompt template."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL, RAG_SYSTEM_PROMPT

_DEFAULT_SYSTEM_PROMPT = """Ты ассистент, отвечающий только на основе приведённого контекста из базы знаний.
Отвечай ТОЛЬКО на основе контекста ниже. Если в контексте нет информации для ответа — так и скажи.
Отвечай на языке пользователя.

Строго соблюдай структуру ответа:

1. **Вводная строка** — одна строка или короткий абзац: общий вывод по запросу.

2. **Разделы с подзаголовками** — используй заголовки ### Название раздела (при необходимости уточнение в скобках). Группируй материал по смыслу.

3. **Пункты с цитированием** — каждый пункт: краткое название/тезис, описание, в конце номера источников в квадратных скобках: [01], [14] или [14, 22]. Допускаются вложенные подпункты.

4. **Блок «Контекст источников»** — в конце ответа обязательно добавь разделитель --- и блок:
---
Контекст источников:
   [01] Краткое название источника: одно предложение — о чём документ.
   [02] …
(только для тех номеров [NN], которые ты реально использовал в ответе; для каждого — узнаваемое имя и одна строка контекста.)

В контексте ниже каждому источнику присвоен номер [01], [02], [03] и т.д. Используй в ответе только эти номера для ссылок. Не придумывай источники."""

SYSTEM_PROMPT = (RAG_SYSTEM_PROMPT.strip() or _DEFAULT_SYSTEM_PROMPT)

HUMAN_TEMPLATE = """Контекст из базы знаний (источники помечены номерами [01], [02], …):

{context}

Вопрос пользователя: {query}

Ответь по шаблону: вводная строка, разделы ### с пунктами и ссылками [NN], в конце блок «Контекст источников»."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


def _numbered_context(contexts: list[dict]) -> tuple[str, dict[str, str]]:
    """
    Build context string with [01], [02] prefixes per source_path and return
    mapping source_path -> number for reference.
    """
    seen: dict[str, str] = {}
    parts: list[str] = []
    num = 1
    for c in contexts:
        path = c.get("source_path") or "?"
        if path not in seen:
            seen[path] = f"{num:02d}"
            num += 1
        nn = seen[path]
        text = (c.get("text") or "").strip()
        parts.append(f"[{nn}]\n{text}")
    return "\n\n".join(parts), seen


def generate_answer(query: str, contexts: list[dict]) -> str:
    """Build numbered context and return LLM answer in the structured template format."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    if not contexts:
        return "По запросу ничего не найдено в базе знаний."

    context_block, _ = _numbered_context(contexts)

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        max_tokens=2048,
    )
    chain = PROMPT | llm
    msg = chain.invoke({"context": context_block, "query": query})
    return (msg.content or "").strip()
