"""«Хочу говорить дольше» / «не перебивай»: a long-listening mode, decided by code.

For up to ten minutes Nora needs no name and does not answer at the first pause:
the phrases are transcribed as they come and glued together, and the whole text
is answered only after a long silence. The microphone is not muted between
phrases, so nothing said is lost.
"""

from __future__ import annotations

import re
import time

WINDOW_S = 600.0  # the mode lasts this long
PAUSE_S = 3.5  # silence that means «I've finished», instead of the usual 0.6 s

STARTED = "Ну давай, выкладывай: слушаю до десяти минут и перебивать не буду. Замолчишь на несколько секунд — отвечу."
STOPPED = "Ладно, вернулась в обычный режим."

_ON = re.compile(
    r"(?:говорить|поговорить|рассказывать|рассказать|подумать|слушать|слушай)\s+(?:меня\s+)?(?:подольше|дольше|долго|до\s+конца)"
    r"|(?:подольше|дольше|долго)\s+(?:говорить|поговорить|рассказывать|слушать|слушай)"
    r"|дай\s+(?:мне\s+)?(?:договорить|подумать|высказаться)"
    r"|не\s+(?:будем\s+)?(?:перебивай|перебивать|торопи|перебрасываться)"
    r"|без\s+перебиваний"
    r"|(?:долг\w+|длинн\w+)\s+(?:режим|слушани\w+)",
    re.IGNORECASE,
)
_OFF = re.compile(
    r"(?:хватит|перестань|прекрати|больше\s+не\s+надо)\s+(?:меня\s+)?(?:долго\s+)?слушать"
    r"|обычн\w+\s+режим"
    r"|можешь\s+(?:снова\s+)?перебивать"
    r"|(?:выключи|отключи|останови)\s+(?:долг\w+|длинн\w+)\s+(?:режим|слушание)",
    re.IGNORECASE,
)
# Words that do not make a request on their own: «Нора, я хочу говорить подольше, пожалуйста».
_FILLER = {
    "нора", "джарвиз", "джарвис", "jarvis", "эй", "я", "мне", "меня", "хочу", "хотела", "хотел", "бы", "пожалуйста", "давай", "ну", "и", "а",
    "сейчас", "можно", "можешь", "ты", "так", "что", "чтобы", "очень", "буду", "немного", "только",
}


def parse_command(text: str) -> str | None:
    """'on', 'off' or None if the phrase does not mention the mode."""
    if _OFF.search(text):
        return "off"
    if _ON.search(text):
        return "on"
    return None


def only_command(text: str) -> bool:
    """The phrase is just the switch, with no other request to answer."""
    rest = (_OFF.sub(" ", text) if _OFF.search(text) else _ON.sub(" ", text)).lower()
    return not [w for w in re.findall(r"[a-zа-яё]+", rest) if w not in _FILLER]


class LongListening:
    def __init__(self) -> None:
        self.until = 0.0
        self._parts: list[str] = []
        self._last = 0.0

    @property
    def active(self) -> bool:
        return time.monotonic() < self.until

    def start(self, window_s: float = WINDOW_S) -> None:
        self.until = max(self.until, time.monotonic() + window_s)

    def stop(self) -> None:
        self.until = 0.0

    def clear(self) -> None:
        self._parts.clear()

    @property
    def collecting(self) -> bool:
        return bool(self._parts)

    def add(self, text: str, now: float | None = None) -> None:
        self._parts.append(text.strip())
        self._last = time.monotonic() if now is None else now

    def take(self, speaking: bool, now: float | None = None) -> str | None:
        """The whole text once the speaker has been silent for PAUSE_S, else None."""
        now = time.monotonic() if now is None else now
        if not self._parts or speaking or now - self._last < PAUSE_S:
            return None
        text, self._parts = " ".join(self._parts), []
        return text
