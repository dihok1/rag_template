"""RAG Telegram bot over custom knowledge base."""
import asyncio
import logging
import time
from collections import defaultdict

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message

from app.config import RATE_LIMIT_PER_MINUTE, TELEGRAM_BOT_TOKEN
from app.rag.llm import generate_answer
from app.rag.retriever import RAGRetriever
from app.utils.telegram_format import markdown_to_telegram_html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limit: chat_id -> list of timestamps in last minute
_rate: defaultdict[int, list[float]] = defaultdict(list)


def _check_rate_limit(chat_id: int) -> bool:
    now = time.monotonic()
    window = 60.0  # 1 minute
    _rate[chat_id] = [t for t in _rate[chat_id] if now - t < window]
    if len(_rate[chat_id]) >= RATE_LIMIT_PER_MINUTE:
        return False
    _rate[chat_id].append(now)
    return True


async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я бот с базой знаний. Добавь документы в папку kb, построй индекс и задавай вопросы — "
        "я отвечу на основе твоих документов."
    )


async def on_text(message: Message, retriever: RAGRetriever) -> None:
    if not message.text or not message.text.strip():
        return
    chat_id = message.chat.id
    if not _check_rate_limit(chat_id):
        await message.answer("Слишком много запросов. Подожди минуту.")
        return
    query = message.text.strip()
    try:
        contexts = retriever.search(query)
        if not contexts:
            await message.answer("По твоему запросу ничего не найдено в базе знаний.")
            return
        answer = generate_answer(query, contexts)
        answer_html = markdown_to_telegram_html(answer)
        try:
            await message.answer(answer_html, parse_mode="HTML")
        except Exception:
            await message.answer(answer)
    except Exception as e:
        logger.exception("RAG error")
        await message.answer(f"Ошибка при ответе: {e!s}")


async def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")

    retriever = RAGRetriever()
    retriever.load()

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, CommandStart())

    async def handle_text(message: Message) -> None:
        await on_text(message, retriever)

    dp.message.register(handle_text, F.text)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
