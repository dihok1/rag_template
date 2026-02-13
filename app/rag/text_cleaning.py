"""
Очистка и нормализация текста перед индексацией и поиском.
Улучшает качество чанков и эмбеддингов.
"""
import re
from pathlib import Path
from unicodedata import normalize as u_normalize


# Файлы/пути, которые не индексируем (macOS resource forks и т.п.)
SKIP_PATH_PATTERNS = (
    re.compile(r"^\._"),           # ._filename
    re.compile(r"/\._"),          # path/._file
    re.compile(r"\.DS_Store$"),
)


def should_skip_path(path: Path | str) -> bool:
    """Возвращает True, если файл нужно пропустить при индексации."""
    name = path.name if isinstance(path, Path) else str(path)
    for pat in SKIP_PATH_PATTERNS:
        if pat.search(name):
            return True
    return False


def clean_text(raw: str) -> str:
    """
    Нормализация текста: пробелы, управляющие символы, Unicode.
    Не удаляет смысловой контент (ссылки, пунктуацию оставляем).
    """
    if not raw or not isinstance(raw, str):
        return ""

    # Нормализация Unicode (NFC — каноническая композиция)
    text = u_normalize("NFC", raw)

    # Удаление управляющих и непечатаемых символов (кроме \n, \t)
    text = "".join(
        c for c in text
        if c in "\n\t\r" or (c.isprintable() or c.isspace())
    )

    # Замена переносов и табов на единообразные
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    # Схлопывание множественных пробелов в один (внутри строки)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Схлопывание множественных пустых строк в максимум две (параграф)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_for_embedding(text: str) -> str:
    """
    Дополнительная нормализация перед отправкой в embedding API:
    убрать лишние переносы, оставить один пробел между словами.
    Используется для запроса; для индексации можно использовать clean_text.
    """
    t = clean_text(text)
    return re.sub(r"\s+", " ", t).strip()
