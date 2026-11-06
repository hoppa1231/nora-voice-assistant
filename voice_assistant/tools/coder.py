"""Hand work to a coding agent (Claude Code or Codex CLI) and speak the result when it is done.

Questions run read-only right away. Changes need a spoken «да» first; changes to the
assistant itself are wrapped in git commits (snapshot before, result after) so any
upgrade can be rolled back with undo_coder_change. New code takes effect after
restart_assistant, which refuses to restart while the tests fail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import ToolContext, ToolError, ToolRegistry
from ..personas import NORA, Persona
from ..skills import build_task

log = logging.getLogger("va.coder")

TIMEOUT_S = 20 * 60
COMMIT_PREFIX = "coder: "
AGENT_NAMES = {"claude": "Клод", "codex": "Кодекс"}

READ_TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]
EDIT_TOOLS = [
    "Read", "Edit", "Write", "Glob", "Grep", "WebSearch", "WebFetch",
    "Bash(uv run pytest:*)", "Bash(uv run python -m compileall:*)", "Bash(uv sync:*)",
    "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)", "Bash(ls:*)",
]

BRIEF = """Тебя вызвал голосовой ассистент {assistant_name} по голосовой просьбе пользователя. Уточнить ничего нельзя: \
работай самостоятельно и доведи задачу до конца.
Проект: {project}.{self_note}
Режим: {mode_note}
В самом конце напиши строку «ИТОГ: …» — одно-два предложения по-русски, их озвучат голосом: \
без markdown, кода, путей и ссылок."""

SELF_NOTE = """
Это код голосового ассистента {assistant_name} (Python 3.12, uv). Устройство описано в README.md. Инструменты — модули в \
voice_assistant/tools/: функция register(reg, ctx) регистрирует их через @reg.tool(описание, свойства, \
required, final) и подключается в build_tools в tools/__init__.py; final=True — результат сразу \
озвучивается. Пиши код в стиле соседнего, добавляй тесты в tests/ и прогоняй `uv run pytest -q`. \
Не трогай data/, models/ и .env. Не коммить: коммитами управляет {assistant_name}."""

MODE_NOTES = {
    "question": "только ответ на вопрос, ничего не меняй.",
    "change": "можно менять файлы проекта. Перед ИТОГ убедись, что тесты проходят.",
}


@dataclass
class Job:
    agent: str
    task: str
    project: Path
    mode: str
    started: float
    log_path: Path
    process: asyncio.subprocess.Process | None = None
    result: str | None = None  # full final answer once done
    summary: str | None = None  # the spoken part
    task_id: str = ""


def short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: text.rfind(" ", 0, limit)].rstrip(",.;:") + "…"


def summarize(answer: str) -> str:
    match = re.search(r"ИТОГ:\s*(.+)", answer, re.DOTALL)
    text = match.group(1) if match else answer
    text = re.sub(r"`[^`]*`|\*\*|__|#+\s|https?://\S+", "", text)
    text = " ".join(text.split())
    if len(text) > 300:
        cut = text[:300]
        text = cut[: max(cut.rfind(". "), cut.rfind("! "), 100) + 1]
    return text.strip()


def command(agent: str, task: str, project: Path, mode: str, brief: str, out_file: Path) -> list[str]:
    if agent == "claude":
        cmd = ["claude", "-p", task, "--output-format", "json", "--append-system-prompt", brief]
        if mode == "change":
            cmd += ["--permission-mode", "acceptEdits", "--allowedTools", *EDIT_TOOLS]
        else:
            cmd += ["--allowedTools", *READ_TOOLS, "--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit"]
        return cmd
    sandbox = "workspace-write" if mode == "change" else "read-only"
    return [
        "codex", "exec", "--cd", str(project), "--sandbox", sandbox, "--skip-git-repo-check", "--ephemeral",
        "--model", os.getenv("CODEX_MODEL", "gpt-6-luna"),
        "-c", "model_reasoning_effort=" + json.dumps(os.getenv("CODEX_REASONING_EFFORT", "low")),
        "-o", str(out_file), f"{brief}\n\nЗадача: {task}",
    ]


def parse_answer(agent: str, stdout: str, out_file: Path) -> str:
    if agent == "claude":
        for line in reversed(stdout.strip().splitlines()):
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if isinstance(data, dict) and "result" in data:
                return str(data["result"])
        return stdout.strip()[-2000:]
    try:
        return out_file.read_text().strip()
    except OSError:
        return stdout.strip()[-2000:]


async def git(project: Path, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git", *args, cwd=project, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        raise ToolError(f"git {args[0]}: {out.decode().strip()[-300:]}")
    return out.decode().strip()


async def commit_all(project: Path, message: str, persona: Persona = NORA) -> bool:
    """Commit everything; False if there was nothing to commit."""
    if not (project / ".git").exists():
        await git(project, "init", "-q")
    await git(project, "add", "-A")
    if not await git(project, "status", "--porcelain"):
        return False
    await git(project, "-c", f"user.name={persona.name}", "-c", f"user.email={persona.key}@localhost", "commit", "-q", "-m", message)
    return True


async def run_tests(project: Path) -> tuple[bool, str]:
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "pytest", "-q", cwd=project, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False, "тесты не закончились за 5 минут"
    lines = out.decode().strip().splitlines()
    return proc.returncode == 0, lines[-1] if lines else ""


class Coder:
    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._logs = ctx.project_dir / "data" / "coder" if ctx.persona.key == "nora" else ctx.project_dir / "data" / ctx.persona.key / "coder"
        self.job: Job | None = None
        self._task: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def resolve_project(self, project: str) -> Path:
        name = (project or "self").strip()
        if name.lower() in ("self", "сама", "сам", "себя", "нора", "джарвиз", "джарвис", "jarvis", ""):
            return self._ctx.project_dir
        root = self._ctx.projects_root.resolve()
        candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")] if root.is_dir() else []
        for path in candidates:
            if path.name.lower() == name.lower():
                return path
        for path in candidates:
            if name.lower() in path.name.lower():
                return path
        names = ", ".join(sorted(p.name for p in candidates)) or "нет"
        raise ToolError(f"нет проекта «{name}» в {root}; есть: {names}")

    def start(self, agent: str, task: str, project: Path, mode: str, task_id: str = "") -> None:
        self._logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_id = task_id or self._ctx.tasks.create(short(task, 160), AGENT_NAMES[agent], "preparing")
        self._ctx.tasks.update(task_id, "preparing", "Подготовка окружения")
        self.job = Job(agent, task, project, mode, time.monotonic(), self._logs / f"{stamp}-{agent}-{mode}.log", task_id=task_id)
        self._task = asyncio.create_task(self._run(self.job))

    async def _run(self, job: Job) -> None:
        is_self = job.project == self._ctx.project_dir
        name = AGENT_NAMES[job.agent]
        activity = f"{name} работает"
        self._ctx.activity.add(activity)
        try:
            if job.mode == "change" and is_self:
                await commit_all(job.project, "snapshot before coder run", self._ctx.persona)
            out_file = job.log_path.with_suffix(".answer.txt")
            self_note = SELF_NOTE.format(assistant_name=self._ctx.persona.name) if is_self else ""
            brief = BRIEF.format(project=job.project, self_note=self_note, mode_note=MODE_NOTES[job.mode], assistant_name=self._ctx.persona.name)
            cmd = command(job.agent, job.task, job.project, job.mode, brief, out_file)
            env = {**os.environ}
            env.pop("VIRTUAL_ENV", None)  # let `uv run` inside the agent pick the project's own env
            job.process = await asyncio.create_subprocess_exec(
                *cmd, cwd=job.project, env=env, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            self._ctx.tasks.update(job.task_id, "running", f"Процесс запущен, PID {job.process.pid}")
            try:
                out, err = await asyncio.wait_for(job.process.communicate(), timeout=TIMEOUT_S)
            except TimeoutError:
                job.process.kill()
                await job.process.wait()
                raise ToolError(f"не уложился в {TIMEOUT_S // 60} минут") from None
            stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
            job.log_path.write_text(f"$ {' '.join(cmd[:2])} …\n\n{stdout}\n--- stderr ---\n{stderr}")
            if job.process.returncode != 0:
                raise ToolError(stderr.strip()[-200:] or f"код выхода {job.process.returncode}")
            job.result = parse_answer(job.agent, stdout, out_file)
            summary = summarize(job.result) or "Готово, но без пояснений."
            if job.mode == "change" and is_self:
                self._ctx.tasks.update(job.task_id, "checking", "Запускаю тесты изменённого кода")
                ok, test_result = await run_tests(job.project)
                if not ok:
                    raise ToolError(f"изменения внесены, но тесты не прошли: {test_result}. Перезапуск не выполнялся")
                if await commit_all(job.project, COMMIT_PREFIX + job.task[:200], self._ctx.persona):
                    summary += " Изменения сохранены; чтобы они заработали, скажи «перезапустись»."
                else:
                    summary += " Файлы не изменились."
            job.summary = f"{name} {'закончил' if job.mode == 'change' else 'ответил'}. {summary}"
            self._ctx.tasks.update(job.task_id, "done", job.summary)
        except asyncio.CancelledError:
            if job.process and job.process.returncode is None:
                job.process.kill()
                await job.process.wait()
            self._ctx.tasks.update(job.task_id, "cancelled", "Ассистент завершил работу")
            self._ctx.activity.discard(activity)
            raise
        except ToolError as error:
            job.summary = f"{name} не справился: {error}"
            self._ctx.tasks.update(job.task_id, "failed", job.summary)
        except Exception as error:
            log.exception("coder run failed")
            job.summary = f"{name} не справился: {type(error).__name__}"
            self._ctx.tasks.update(job.task_id, "failed", job.summary)
        self._ctx.activity.discard(activity)
        log.info("coder done in %.0fs: %s", time.monotonic() - job.started, job.summary)
        self._ctx.announce(job.summary)


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    coder = Coder(ctx)

    @reg.tool(
        "Поручить задачу программисту-агенту (Claude или Codex), он работает в фоне несколько минут и потом "
        "сообщает итог. mode=question — сложный вопрос, разбор кода, исследование, ничего не меняет. "
        "mode=change — написать или исправить код (например, добавить тебе новое умение), спросит подтверждение.",
        {
            "task": {"type": "string", "description": "Дословная цитата запроса пользователя, без пересказа, сокращений и собственных требований."},
            "agent": {"type": "string", "enum": ["claude", "codex"], "description": "По умолчанию codex. Claude только по явной просьбе пользователя."},
            "project": {"type": "string", "description": "self — твой собственный код; иначе имя папки проекта в ~/Projects."},
            "mode": {"type": "string", "enum": ["question", "change"]},
        },
        required=("task", "mode"),
        final=True,
    )
    async def ask_coder(task: str, mode: str, agent: str = "codex", project: str = "self") -> str:
        agent = agent if agent in AGENT_NAMES else "codex"
        if shutil.which(agent) is None:
            raise ToolError(f"не установлен {agent}")
        if coder.busy:
            raise ToolError(f"{AGENT_NAMES[coder.job.agent]} ещё работает над прошлой задачей")
        path = coder.resolve_project(project)
        if mode == "change" and path == ctx.project_dir and reg.raw_user_text is not None:
            # The model cannot rewrite the user's self-upgrade request.
            task = reg.coder_request or build_task(reg.raw_user_text, persona=ctx.persona)
        name = AGENT_NAMES[agent]
        where = "мой код" if path == ctx.project_dir else f"проект {path.name}"
        if mode != "change":
            coder.start(agent, task, path, "question")
            return f"Задача принята. Запускаю {name} для ответа на вопрос; результат сообщу отдельно."

        task_id = ctx.tasks.create(short(task, 160), name)

        async def go() -> str:
            if coder.busy:
                ctx.tasks.update(task_id, "failed", "Агент уже занят")
                raise ToolError("агент уже занят")
            coder.start(agent, task, path, "change", task_id=task_id)
            return f"Задача принята. Запускаю {name}; фактический статус виден на экране, итог сообщу отдельно."

        return ctx.pending.request(
            f"Поручить {name}у изменить {where}: {short(task)}?", go,
            on_cancel=lambda: ctx.tasks.update(task_id, "cancelled", "Подтверждение не получено"),
        )

    @reg.tool("Что сейчас делает программист-агент и чем закончилась последняя задача (с подробностями).", final=True)
    def coder_status() -> str:
        ctx.pending.expire()
        active = ctx.tasks.snapshot()
        waiting = next((t for t in active if t["state"] == "awaiting_confirmation"), None)
        if waiting and not coder.busy:
            return f"Задача ещё не запущена: {waiting['title']}. Жду отдельного подтверждения «да» или «нет»."
        job = coder.job
        if job is None:
            return "Агенту ещё ничего не поручали."
        if coder.busy:
            from ..tasks import task_summary
            task = next((t for t in active if t["id"] == job.task_id), {})
            return f"{AGENT_NAMES[job.agent]}: {task_summary(task)}. {int(time.monotonic() - job.started) // 60} мин с принятия задачи: {job.task[:150]}."
        details = f" Полный ответ: {job.result[:1500]}" if job.result else ""
        return f"Последняя задача: {job.task[:150].rstrip('.')}. {job.summary}{details}"

    @reg.tool(
        "Откатить последнее изменение твоего кода, сделанное агентом (спросит подтверждение).",
        final=True,
    )
    async def undo_coder_change() -> str:
        project = ctx.project_dir
        if not (project / ".git").exists():
            raise ToolError("в проекте нет истории изменений")
        last = await git(project, "log", "-1", "--format=%H %s")
        sha, _, subject = last.partition(" ")
        if not subject.startswith(COMMIT_PREFIX):
            raise ToolError("последнее изменение сделано не агентом")

        async def revert() -> str:
            await git(project, "-c", f"user.name={ctx.persona.name}", "-c", f"user.email={ctx.persona.key}@localhost", "revert", "--no-edit", sha)
            return "Откатила. Чтобы вернуться к старой версии, скажи «перезапустись»."

        return ctx.pending.request(f"Откатить изменение «{subject[len(COMMIT_PREFIX):][:100]}»?", revert)

    @reg.tool("Перезапустить себя, чтобы заработал новый код (сначала проверит тесты, спросит подтверждение).", final=True, timeout_s=310)
    async def restart_assistant() -> str:
        async def restart() -> str:
            ok, line = await run_tests(ctx.project_dir)
            if not ok:
                raise ToolError(f"тесты не проходят ({line}), перезапуск отменён")
            loop = asyncio.get_running_loop()
            loop.call_later(4, os.execv, sys.executable, [sys.executable, "-m", "voice_assistant", *sys.argv[1:]])
            return "Тесты прошли, перезапускаюсь."

        return ctx.pending.request("Перезапуститься?", restart)
