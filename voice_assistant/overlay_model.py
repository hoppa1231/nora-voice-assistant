"""Pure layout/persistence helpers, also loaded by the standalone GTK process."""
import json
import os
import tempfile
from pathlib import Path

TASK_LABELS = {"awaiting_confirmation": "Жду подтверждения", "preparing": "Запускаю", "running": "Выполняется", "checking": "Проверяю", "done": "Готово", "failed": "Ошибка", "cancelled": "Отменено"}


def clamp_position(x: int, y: int, width: int, height: int, screen_width: int, screen_height: int) -> tuple[int, int]:
    return max(0, min(int(x), max(0, screen_width - width))), max(0, min(int(y), max(0, screen_height - height)))


def load_position(path: str) -> tuple[int, int] | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text())
        if type(value["x"]) is int and type(value["y"]) is int:
            return value["x"], value["y"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def save_position(path: str, x: int, y: int) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, prefix=".indicator-", delete=False) as file:
        temporary = file.name
        json.dump({"x": int(x), "y": int(y)}, file)
    try:
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
