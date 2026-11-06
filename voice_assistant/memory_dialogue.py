"""Explicit memory commands: actual writes, unambiguous targets and clarifications."""
import json
import re
import time

from .memory import Memory, normalize_fact
from .memory_rules import automatic_fact
from .tools import ToolRegistry

_PREFIX = r"^\W*(?:(?:нора|джарви[зс]|jarvis)[,\s]+)?(?:пожалуйста[,\s]+)?"
_SAVE = re.compile(_PREFIX + r"(?:запомни|запоминай|запиши\s+в\s+память|сохрани\s+в\s+память)\b[,:\s]*(?:пожалуйста[,\s]+)?(?:что\s+)?(.*)$", re.I | re.S)
_DELETE = re.compile(_PREFIX + r"(?:забудь\b|удали\s+(?:из\s+памяти|факт|информацию|запись)\b)\s*(.*)$", re.I | re.S)
_UPDATE = re.compile(_PREFIX + r"(?:перезапиши|обнови|исправь|измени|замени)\s+(?:(?:в\s+памяти|факт|информацию|запись)\s+)?(.+?)\s+(?:на|теперь)\s+(.+)$", re.I | re.S)
_SHOW = re.compile(_PREFIX + r"(?:что\s+ты\s+(?:обо\s+мне\s+)?(?:помнишь|запомнил[аи]?)|что\s+ты\s+знаешь\s+обо\s+мне|(?:покажи|перечисли)\s+(?:все\s+)?(?:факты|память|сохраненные\s+факты))\W*$", re.I)
_STOP = {"факт", "факты", "номер", "информацию", "информация", "запись", "памяти", "память", "про", "о", "об", "обо", "что", "мой", "моя", "мое", "меня", "я", "это"}


class MemoryDialogue:
    def __init__(self, memory: Memory, tools: ToolRegistry):
        self.memory, self.tools = memory, tools
        self.pending = None
        self.used_tool = False
        self.changed = False

    async def call(self, name: str, args: dict) -> str:
        self.used_tool = True
        previous = self.tools.memory_edit_allowed
        previous_save = self.tools.memory_save_allowed
        self.tools.memory_edit_allowed = name in {"update_fact", "forget"}
        self.tools.memory_save_allowed = name == "remember"
        try:
            result = await self.tools.call(name, json.dumps(args, ensure_ascii=False))
        finally:
            self.tools.memory_edit_allowed = previous
            self.tools.memory_save_allowed = previous_save
        self.changed = result.ok and name != "list_facts"
        return result.text

    async def handle(self, text: str) -> str | None:
        self.used_tool = self.changed = False
        pending, self.pending = self.pending, None
        if pending and time.monotonic() - pending[0] < 90:
            _, operation, target, replacement, choices = pending
            if re.fullmatch(r"\W*(?:нет|отмена|не\s+надо)\W*", text, re.I):
                return "Ладно, память не меняю."
            if operation == "save":
                if not (_SHOW.match(text) or _DELETE.match(text) or _SAVE.match(text) or _UPDATE.match(text)):
                    return await self.call("remember", {"fact": text.strip()})
            elif not (_SHOW.match(text) or _DELETE.match(text) or _SAVE.match(text) or _UPDATE.match(text)):
                ordinal = {"первый": 0, "первую": 0, "второй": 1, "вторую": 1, "третий": 2, "третью": 2}
                t = text.lower().strip(" .!?")
                chosen = [choices[ordinal[t]]] if t in ordinal and ordinal[t] < len(choices) else self.find(text, choices)
                if len(chosen) == 1:
                    return await self.edit(operation, chosen[0], replacement)
        if _SHOW.match(text):
            return await self.call("list_facts", {})
        match = _SAVE.match(text)
        if match:
            fact = match[1].strip()
            if normalize_fact(fact) in ("", "это", "вот это"):
                self.pending = (time.monotonic(), "save", "", "", [])
                return "Что именно запомнить? Продиктуй факт."
            return await self.call("remember", {"fact": fact})
        match = _UPDATE.match(text)
        if match:
            if not re.search(r"\b(?:факт|памяти|информацию|запись)\b", text, re.I) and not self.find(match[1], self.memory.facts(limit=-1)):
                return None
            return await self.resolve("update_fact", match[1], match[2].strip())
        match = _DELETE.match(text)
        if match:
            return await self.resolve("forget", match[1], "")
        return None

    def find(self, query: str, facts: list) -> list:
        query = query.strip(' ,:«»"')
        id_match = re.fullmatch(r"(?:факт\s+)?(?:номер\s*)?#?\s*(\d+)[.!]?", query, re.I)
        if id_match:
            return [f for f in facts if f.id == int(id_match[1])]
        normal = normalize_fact(query)
        exact = [f for f in facts if normalize_fact(f.text) == normal]
        if exact:
            return exact
        words = [w for w in normal.split() if w not in _STOP]
        if not words:
            return []
        # Conservative inflection tolerance; several matches always require a choice.
        return [f for f in facts if all(any(a.startswith(w[:max(4, len(w)-2)]) for a in normalize_fact(f.text).split()) for w in words)]

    async def resolve(self, operation: str, target: str, replacement: str) -> str:
        if operation not in self.tools.names:
            return "Этот инструмент памяти отключён в настройках."
        facts = self.memory.facts(limit=-1)
        matches = self.find(target, facts)
        if len(matches) == 1:
            return await self.edit(operation, matches[0], replacement)
        choices = matches or facts
        if not choices:
            return "В памяти пока нет подходящего факта. Ничего не изменено."
        self.pending = (time.monotonic(), operation, target, replacement, choices)
        return "Какой именно факт? Назови его номер: " + "; ".join(f"#{f.id}: {f.text}" for f in choices) + "."

    async def edit(self, operation: str, fact, replacement: str) -> str:
        args = {"fact_id": fact.id}
        if operation == "update_fact":
            args["fact"] = replacement
        return await self.call(operation, args)

    def observe(self, text: str) -> str:
        if "remember" not in self.tools.names:
            return ""
        candidate = automatic_fact(text)
        if candidate:
            fact, key, important = candidate
            if self.memory.observe_fact(fact, key=key, important=important):
                return self.tools.persona.stock("Запомнила: ") + fact + ". "
        return ""
