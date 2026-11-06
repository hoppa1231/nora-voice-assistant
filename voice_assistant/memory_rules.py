"""Small, auditable rules: only the user's own declarative words enter auto-memory."""
import re
from .memory import fingerprint


def fact_key(text: str) -> str:
    text = text.strip().strip('«»"').lower()
    if re.match(r"(?:что\s+)?(?:меня\s+зовут|мо[её]\s+имя)\b", text):
        return "user_name"
    if re.match(r"(?:что\s+)?(?:моя\s+(?:учебная\s+)?группа|я\s+учусь\s+в\s+группе)\b", text):
        return "study_group"
    preference = re.fullmatch(r"(?:я\s+(?:люблю|не\s+люблю)|мне\s+(?:нравится|не\s+нравится))\s+(.+)", text)
    if preference:
        return "preference:" + fingerprint(preference[1])
    return ""


def automatic_fact(text: str) -> tuple[str, str, bool] | None:
    text = text.strip().rstrip(".!…")
    if len(text) > 220 or any(c in text for c in '?\n"«»'):
        return None
    if re.search(r"\b(?:если|например|допустим|возможно|наверное|сегодня|сейчас|вчера|завтра|не\s+запоминай|не\s+сохраняй|пароль|токен|секрет|паспорт|диагноз|болезнь|адрес|телефон)\b|https?://|\b\d{7,}\b", text, re.I):
        return None
    if re.search(r"парол|паспорт|токен|секрет|диагноз|болезн|здоров|религи|политич|сексу|банков|кредит|api.?key|token", text, re.I):
        return None
    if re.search(r"\b(?:скажи|сделай|открой|закрой|удали|забудь|запомни|включи|выключи|запусти)\b", text, re.I):
        return None
    key = fact_key(text)
    if key in {"user_name", "study_group"}:
        # A stable identity/group statement, not an instruction or quoted text.
        if re.search(r"\b(?:скажи|запомни|удали|забудь|исправь|больше|теперь|раньше)\b", text, re.I):
            return None
        if key == "user_name" and not re.fullmatch(r"(?:меня\s+зовут|мо[её]\s+имя\s*[-—:]?)\s+[а-яёa-z-]+(?:\s+[а-яёa-z-]+){0,3}", text, re.I):
            return None
        if key == "user_name" and re.search(r"\b(?:а|и|или|не|как|что|это)\b", text, re.I):
            return None
        if key == "study_group" and not re.fullmatch(r"(?:моя\s+(?:учебная\s+)?группа\s*[-—:]?|я\s+учусь\s+в\s+группе)\s+[\w./-]+", text, re.I):
            return None
        return text, key, True
    if re.fullmatch(r"(?:я\s+(?:предпочитаю|люблю|не\s+люблю)|мне\s+нравится)\s+[^;]+", text, re.I):
        return text, key, False
    return None


def explicit_memory_request(text: str) -> bool:
    return bool(re.match(r"^\W*(?:пожалуйста\W+)?(?:запомни|запоминай|запиши\s+в\s+память|сохрани\s+в\s+память)\b", text, re.I))
