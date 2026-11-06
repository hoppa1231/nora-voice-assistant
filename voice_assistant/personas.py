"""Named assistant profiles; shared capabilities, separate identity and memory."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    wake_word: str
    male: bool = False

    def stock(self, text: str) -> str:
        """Inflect only known response prefixes, never quoted user facts or labels."""
        if not self.male:
            return text
        forms = {"Поставила": "Поставил", "Отменила": "Отменил", "Запомнила": "Запомнил", "Забыла": "Забыл", "Спросила": "Спросил", "Поручила": "Поручил", "Откатила": "Откатил"}
        text = re.sub(r"^(Поставила|Отменила|Запомнила|Забыла|Спросила|Поручила|Откатила)\b", lambda m: forms[m[0]], text)
        if text == "Ладно, вернулась в обычный режим.":
            return "Вернулся в обычный режим."
        return text


NORA = Persona("nora", "Нора", "нора")
JARVIS = Persona("jarvis", "Джарвиз", "джарвиз", male=True)
PROFILES = {p.key: p for p in (NORA, JARVIS)}


def get_persona(profile: str) -> Persona:
    try:
        return PROFILES[profile.lower()]
    except KeyError:
        raise ValueError(f"Неизвестный профиль {profile!r}; доступны nora и jarvis") from None
