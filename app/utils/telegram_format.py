"""Convert LLM Markdown output to Telegram HTML for correct display."""
import html


def markdown_to_telegram_html(text: str) -> str:
    """
    Convert common Markdown (**bold**, *italic*, `code`) to Telegram HTML.
    Escapes <, >, & for HTML safety, then applies tags.
    """
    if not text or not text.strip():
        return text
    # Сначала экранируем HTML-символы, чтобы не сломать разметку
    out = html.escape(text)
    # **bold** -> <b>bold</b> (обрабатываем первым, т.к. содержит *)
    parts = out.split("**")
    for i in range(1, len(parts), 2):
        parts[i] = f"<b>{parts[i]}</b>"
    out = "".join(parts)
    # *italic* -> <i>italic</i>
    parts = out.split("*")
    for i in range(1, len(parts), 2):
        parts[i] = f"<i>{parts[i]}</i>"
    out = "".join(parts)
    # `code` -> <code>code</code>
    parts = out.split("`")
    for i in range(1, len(parts), 2):
        parts[i] = f"<code>{parts[i]}</code>"
    out = "".join(parts)
    return out
