"""Check operational acknowledgements against actual successful tool calls."""
import re

_EXECUTION = re.compile(r"\b(?:поставь|запусти|добавь|создай|допиши|исправь|сделай|открой|закрой|переключи|напечатай|отмени|запомни|сохрани|поручи|передай|попроси|найди|посмотри|проверь|включи|выключи|напомни)\b", re.I)


def needs_execution(text: str) -> bool:
    question = re.search(r"^\W*(?:как|почему|зачем|что\s+(?:значит|будет)|можно\s+ли)\b", text, re.I)
    # Text composition is an answer, not an external action.
    composition = re.search(r"\b(?:сделай|составь|создай|добавь)\s+(?:мне\s+)?(?:кратк\w*\s+)?(?:резюме|список|стих|рассказ|пример|перевод|текст|объяснение)\b", text, re.I)
    return not question and not composition and bool(_EXECUTION.search(text))


def explains_nonexecution(text: str) -> bool:
    """Allow clarifying questions and honest limits instead of a generic refusal."""
    return bool(text.rstrip().endswith("?") or re.search(r"\b(?:не\s+(?:могу|умею|выполнено|запущен)|нет\s+(?:доступа|инструмента)|недоступн|уточни)", text, re.I))


def unsupported_claim(text: str, tools: set[str]) -> bool:
    checks = [
        (r"(?:поруч(?:ил[аи]?|аю|у)|переда(?:л[аи]?|ю|м)).{0,50}(?:кодекс|клод|агент)|(?:кодекс|клод|агент).{0,35}(?:работает|запущен)|работаю\s+над\s+этим|задач[ау].{0,20}(?:в\s+фон|запущен)", {"ask_coder"}),
        (r"постав(?:ил[аи]?|лю).{0,25}таймер|таймер.{0,15}(?:поставлен|запущен)", {"set_timer"}),
        (r"запомнил[аи]?|сохранил[аи]?\s+(?:факт|предпочтени)", {"remember"}),
        (r"открыл[аи]?\s+(?:браузер|сайт|программ|окно)", {"open_app", "open_url", "browser_search"}),
        (r"закрыл[аи]?\s+(?:окно|программ)", {"close_window"}),
        (r"напечатал[аи]?\b", {"type_text"}),
    ]
    return any(re.search(pattern, text, re.I) and not (allowed & tools) for pattern, allowed in checks)
