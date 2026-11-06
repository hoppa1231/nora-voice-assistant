from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest

from voice_assistant.memory import Memory
from voice_assistant.speech import SentenceSplitter, clean_for_speech
from voice_assistant.tools import ToolContext, ToolError, ToolRegistry, build_tools
from voice_assistant.tools.alarms import parse_duration, parse_local_time
from voice_assistant.tools.web import parse_html, render_page
from voice_assistant.tts import normalize_for_silero

MSK = ZoneInfo("Europe/Moscow")


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("5 минут", 300),
        ("пять минут", 300),
        ("полтора часа", 5400),
        ("1.5 часа", 5400),
        ("1,5 часа", 5400),
        ("1 час 20 минут", 4800),
        ("час двадцать", 4800),
        ("полчаса", 1800),
        ("минуту", 60),
        ("90 секунд", 90),
        ("двадцать пять минут", 1500),
        ("2 минуты 30 секунд", 150),
        ("10", 600),
        ("PT1H30M", 5400),
        ("5m", 300),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_rejects_nonsense():
    with pytest.raises(ToolError):
        parse_duration("когда-нибудь")


def test_local_time_ignores_bogus_utc_suffix():
    ctx = ToolContext(Memory(":memory:"), None, MSK, "")
    due = parse_local_time("2026-10-27T09:00:00Z", ctx)
    assert (due.hour, due.tzinfo) == (9, MSK)


def test_splitter_streams_sentences():
    splitter = SentenceSplitter()
    out = []
    for piece in ["Привет", "! Сейчас 14:05. Погода", " хорошая", ", тепло"]:
        out += splitter.feed(piece)
    assert out == ["Привет!", "Сейчас 14:05."]
    assert splitter.flush() == ["Погода хорошая, тепло"]


def test_splitter_cuts_long_sentence_at_comma():
    splitter = SentenceSplitter(soft_limit=60)
    splitter.feed("Готово. ")  # past the first piece
    text = "Сначала обжарь курицу до золотистой корочки, потом добавь рис и залей водой так"
    assert splitter.feed(text)[0].endswith("корочки,")


def test_splitter_cuts_first_piece_early():
    splitter = SentenceSplitter()
    assert splitter.feed("Небо голубое из-за того, что молекулы воздуха рассеивают, ") == ["Небо голубое из-за того, что молекулы воздуха рассеивают,"]
    assert splitter.feed("синий свет сильнее, чем красный, и") == []  # later pieces wait for a sentence end
    splitter.flush()
    assert splitter.feed("Короткий ответ, да. ") == ["Короткий ответ, да."]
    splitter.flush()
    long_clause = "Небо голубое из-за рассеяния солнечного света молекулами воздуха в атмосфере Земли днём"
    assert splitter.feed(long_clause) == ["Небо голубое из-за рассеяния солнечного света молекулами воздуха в атмосфере Земли"]


def test_clean_for_speech():
    assert clean_for_speech("**Важно**: смотри https://x.ru 😀") == "Важно : смотри"
    assert clean_for_speech("...") == ""


def test_normalize_for_silero_keeps_stress_marks():
    assert normalize_for_silero("Н+ора, привет") == "Н+ора, привет"
    assert normalize_for_silero("Меня зовут Нора.") == "Меня зовут Н+ора."


def test_normalize_for_silero():
    assert normalize_for_silero("в 14:05, +7°C") == "в четырнадцать пять, плюс семь градусов"
    assert normalize_for_silero("+2°C, ветер 1 м/с, 23%") == "плюс два градуса, ветер один метр в секунду, двадцать три процента"


def test_memory_roundtrip(tmp_path):
    memory = Memory(tmp_path / "m.db")
    fact = memory.add_fact("Пьёт кофе без сахара")
    memory.save_routine("Расписание", "открыть a")
    memory.save_routine("расписание", "открыть b")
    assert [f.text for f in memory.facts()] == ["Пьёт кофе без сахара"]
    assert [(r.name, r.instructions) for r in memory.routines()] == [("расписание", "открыть b")]
    assert memory.delete_fact(fact) and not memory.delete_fact(fact)

    now = datetime.now(UTC)
    past = memory.add_alarm("timer", now - timedelta(seconds=1), "чай")
    memory.add_alarm("reminder", now + timedelta(hours=1), "позвонить")
    assert [a.id for a in memory.due_alarms(now)] == [past]
    assert memory.finish_alarm(past) and not memory.finish_alarm(past)
    assert memory.due_alarms(now) == []


SCHEDULE_HTML = """
<html><head><title>Расписание ИВТ</title><style>.x{}</style></head><body>
<nav><a href="/groups">Все группы</a> <a href="https://vk.com/uni">ВК</a></nav>
<form action="/schedule"><select name="group">
  <option value="101">ИВТ-21</option><option value="102">ИВТ-22</option><option value="201">ПМИ-21</option>
</select></form>
<h2>Понедельник 26.10</h2>
<table><tr><th>Время</th><th>Предмет</th><th>Ауд.</th></tr>
<tr><td>9:00</td><td>Матанализ</td><td>301</td></tr></table>
<h2>Вторник 27.10</h2>
<table><tr><td>10:40</td><td>Физика</td><td>214</td></tr></table>
<script>alert(1)</script>
</body></html>
"""


def test_parse_schedule_page():
    page = parse_html(SCHEDULE_HTML, "https://uni.example/schedule")
    assert page.title == "Расписание ИВТ"
    assert "9:00 | Матанализ | 301" in page.lines
    assert ("Все группы", "https://uni.example/groups") in page.links
    name, options, action = page.selects[0]
    assert (name, action) == ("group", "https://uni.example/schedule")
    assert ("ИВТ-22", "102") in options

    rendered = render_page(page, "ИВТ-22 вторник")
    assert "ИВТ-22=102" in rendered and "ИВТ-21=101" not in rendered
    assert "10:40 | Физика | 214" in rendered
    assert "alert" not in rendered


async def test_registry_reports_errors_to_model():
    reg = ToolRegistry()

    @reg.tool("x", {"n": {"type": "integer"}}, required=("n",))
    def double(n: int) -> str:
        if n < 0:
            raise ToolError("отрицательное")
        return str(n * 2)

    assert (await reg.call("double", '{"n": 2, "junk": 1}')).text == "4"
    failed = await reg.call("double", '{"n": -1}')
    assert (failed.text, failed.ok) == ("Ошибка: отрицательное", False)
    assert (await reg.call("double", "not json")).text.startswith("Ошибка: аргументы")
    assert (await reg.call("nope", "{}")).text.startswith("Ошибка: нет инструмента")


async def test_all_tools_register():
    async with httpx.AsyncClient() as http:
        reg = build_tools(ToolContext(Memory(":memory:"), http, MSK, ""))
    assert {"set_timer", "add_reminder", "remember", "save_routine", "web_fetch", "web_search", "get_weather", "media_control", "volume"} <= set(reg.names)
    timer = await reg.call("set_timer", '{"duration": "полтора часа", "label": "пельмени"}')
    assert (timer.text, timer.final) == ("Поставила таймер «пельмени» на 1 час 30 минут.", True)
    listed = await reg.call("list_alarms", "{}")
    assert "пельмени" in listed.text and not listed.final


def test_say_duration_plurals():
    from voice_assistant.tools.alarms import say_duration

    assert say_duration(61) == "1 минуту 1 секунду"
    assert say_duration(2 * 3600 + 22 * 60) == "2 часа 22 минуты"
    assert say_duration(11 * 60) == "11 минут"


@pytest.mark.parametrize(
    ("text", "forced"),
    [
        ("Запомни, что я не люблю лук", True),
        ("Пожалуйста, напомни завтра позвонить", True),
        ("Поставь таймер на 5 минут", True),
        ("Сделай погромче", True),
        ("Как меня зовут?", False),
        ("Что ты запомнил обо мне?", False),
    ],
)
def test_action_commands_force_a_tool(text, forced):
    from voice_assistant.assistant import ACTION_COMMAND

    assert bool(ACTION_COMMAND.match(text)) is forced


def test_group_code_matches_whole():
    from voice_assistant.tools.web import _keywords, _matches

    keywords = _keywords("группа 5130904/50002")
    assert _matches("5130904/50002 -> https://x/1", keywords)
    assert not _matches("5130904/50003 -> https://x/2", keywords)


def test_match_routine_by_triggers():
    from voice_assistant.memory import Routine
    from voice_assistant.routines import match_routine

    schedule = Routine("расписание", "открой X", ("расписание", "пары", "занятия"))
    assert match_routine("Какие у меня пары в четверг?", [schedule]) is schedule
    assert match_routine("Во сколько завтра первая пара?", [schedule]) is schedule
    assert match_routine("Какая завтра погода?", [schedule]) is None


async def test_learner_never_saves_an_unopened_url():
    from voice_assistant.llm import Reply, ToolCall
    from voice_assistant.routines import learn_routine

    class FakeLlm:
        async def complete(self, messages, tools, on_text, require_tool=False):
            args = {"name": "Расписание", "triggers": "пары, занятия", "url": "https://made.up/page", "query_hint": "день"}
            return Reply("", [ToolCall("1", "learn_routine", json.dumps(args))])

    import json
    memory = Memory(":memory:")
    turn = [
        {"role": "user", "content": "[ctx]\nмоё расписание на ruz, группа 1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "web_fetch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "Адрес: https://ruz.example/groups/7\nТекст: ..."},
        {"role": "assistant", "content": "Завтра две пары."},
    ]
    routine = await learn_routine(FakeLlm(), memory, turn)
    assert routine.name == "расписание"
    assert routine.url == "https://ruz.example/groups/7" and "made.up" not in routine.instructions
    assert memory.routines()[0].triggers == ("расписание", "пары", "занятия")


def test_expand_relative_dates():
    from voice_assistant.tools.web import expand_relative_dates

    monday = datetime(2026, 10, 26, 4, 0, tzinfo=MSK)
    assert expand_relative_dates("пары завтра", monday) == "пары 27 окт 27.10 вторник"
    assert expand_relative_dates("пары в четверг", monday) == "пары в 29 окт 29.10 четверг"
    assert expand_relative_dates("что в понедельник", monday) == "что в 26 окт 26.10 понедельник"


def test_memory_dedupes_triggers_and_keeps_url():
    memory = Memory(":memory:")
    memory.save_routine("Расписание", "x", ["расписание", "Расписание", "пары"], "https://r/1")
    routine = memory.routines()[0]
    assert (routine.triggers, routine.url) == (("расписание", "пары"), "https://r/1")


@pytest.mark.parametrize(
    ("text", "request_text"),
    [
        ("Нора, поставь таймер на 5 минут", "поставь таймер на 5 минут"),
        ("Нора", ""),
        ("Инора, какая погода", "какая погода"),
        ("Эйнора, какая погода?", "какая погода?"),
        ("Эй, Нора, какая погода", "какая погода"),
        ("Слушай, Норо, включи музыку", "включи музыку"),
        ("Нуора, какие у меня пары завтра", "какие у меня пары завтра"),
        ("Сколько времени, Нора?", "Сколько времени"),
        ("Я вчера видел нору лисы в лесу", None),
        ("Нормально всё, спасибо", None),
        ("Какая погода завтра?", None),
        ("Видел нору", None),
        ("Ора, привет.", "привет."),
        ("Сколько времени, Нра?", "Сколько времени"),
        ("Что ты можешь, Нура", "Что ты можешь"),
        ("Пора спать", None),
        ("Кора дуба", None),
    ],
)
def test_wake_word(text, request_text):
    from voice_assistant.wake import find_wake_word

    assert find_wake_word(text, "Нора") == request_text


@pytest.mark.parametrize(
    ("text", "echo", "result"),
    [
        ("Нора, стоп", "Сегодня в Петербурге плюс пять, облачно.", ""),
        ("Хватит", "Сегодня в Петербурге плюс пять, облачно.", ""),
        ("в Петербурге плюс пять нора подожди", "Сегодня в Петербурге плюс пять, облачно.", ""),
        ("Нора, поставь таймер на пять минут", "Сегодня в Петербурге плюс пять, облачно.", "поставь таймер на пять минут"),
        ("Сегодня в Петербурге плюс пять", "Сегодня в Петербурге плюс пять, облачно.", None),
        ("Меня зовут Нора", "Меня зовут Нора, я голосовой ассистент.", None),
        ("Какая-то музыка играет", "Сегодня облачно.", None),
    ],
)
def test_interruption(text, echo, result):
    from voice_assistant.wake import find_interruption

    assert find_interruption(text, "Нора", echo) == result
