"""Web search and page reading, compact enough for a small model's context.

A page becomes plain text, one line per block element, with link targets
written inline ("5130904/50002 → https://…") so the model never has to match a
name to a separate list of links. <select> lists (group / faculty pickers) are
listed with their option values. With a `query`, only the matching parts are
kept, so a whole week of a schedule fits in a few KB.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString

from . import ToolContext, ToolError, ToolRegistry

MAX_TEXT_CHARS = 3000
MAX_OPTIONS = 25
MAX_LINKS_PER_LINE = 2  # a line listing ten groups is noise with URLs, useful without

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
)

BLOCK_TAGS = [
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset", "figcaption",
    "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p",
    "pre", "section", "table", "tr", "ul",
]
MONTHS_SHORT = ("янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
RELATIVE_DAYS = {"сегодня": 0, "завтра": 1, "послезавтра": 2}
WEEKDAY_STEMS = ("понедельн", "вторник", "сред", "четверг", "пятниц", "суббот", "воскресен")

_LINK_START, _LINK_END = "\x02", "\x03"
_LINK = re.compile(_LINK_START + r"([^\x03]*)" + _LINK_END)


@dataclass
class Page:
    url: str
    title: str
    lines: list[str]
    links: list[tuple[str, str]] = field(default_factory=list)
    selects: list[tuple[str, list[tuple[str, str]], str]] = field(default_factory=list)  # (name, options, form url)


def _squash(text: str) -> str:
    return " ".join(text.split())


def parse_html(html: str, url: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    title = _squash(soup.title.get_text()) if soup.title else ""
    for tag in soup(["head", "title", "script", "style", "noscript", "svg", "template", "iframe"]):
        tag.decompose()

    selects = []
    for select in soup.find_all("select"):
        name = select.get("name") or select.get("id") or "select"
        form = select.find_parent("form")
        action = urljoin(url, form.get("action") or url) if form else url
        options = [(_squash(o.get_text()), o.get("value", "")) for o in select.find_all("option")]
        selects.append((name, [o for o in options if o[0]], action))
        select.decompose()

    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        text = _squash(a.get_text(" ")) or _squash(a.get("title", ""))
        href = urljoin(url, a["href"]).split("#", 1)[0]
        if not text or not href.startswith("http"):
            continue
        if href not in seen:
            seen.add(href)
            links.append((text[:80], href))
        a.append(NavigableString(f"{_LINK_START}{href}{_LINK_END}"))

    for row in soup.find_all("tr"):
        for cell in row.find_all(["td", "th"]):
            cell.insert_after(NavigableString(" | "))
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))

    lines, previous = [], None
    for raw in soup.get_text(" ").splitlines():
        line = _squash(raw).strip(" |")
        targets = _LINK.findall(line)
        if len(targets) > MAX_LINKS_PER_LINE:
            line = _LINK.sub("", line)
        else:
            line = _LINK.sub(lambda m: f" → {m.group(1)}", line)
        line = _squash(line)
        if line and line != previous:
            lines.append(line)
            previous = line
    return Page(url, title, lines, links, selects)


def _say_date(day: datetime) -> str:
    return f"{day.day} {MONTHS_SHORT[day.month - 1]} {day:%d.%m} {WEEKDAYS[day.weekday()]}"


def expand_relative_dates(query: str, now: datetime) -> str:
    """'пары завтра' -> 'пары 27 окт 27.10 вторник', 'в четверг' -> the coming Thursday's date:
    sites print dates, not 'tomorrow'."""
    def relative(match: re.Match[str]) -> str:
        return _say_date(now + timedelta(days=RELATIVE_DAYS[match.group(0).lower()]))

    def weekday(match: re.Match[str]) -> str:
        target = next(i for i, stem in enumerate(WEEKDAY_STEMS) if match.group(0).lower().startswith(stem))
        return _say_date(now + timedelta(days=(target - now.weekday()) % 7))

    query = re.sub(r"\b(?:послезавтра|сегодня|завтра)\b", relative, query, flags=re.IGNORECASE)
    if not re.search(r"\d{1,2}\.\d{2}", query):  # already expanded "завтра" names its weekday too
        query = re.sub(r"\b(?:" + "|".join(WEEKDAY_STEMS) + r")[а-я]*", weekday, query, flags=re.IGNORECASE)
    return query


@dataclass(frozen=True)
class Keyword:
    pattern: re.Pattern[str]
    strong: bool  # dates and codes: when one of them matches, plain words are ignored


def _keywords(query: str) -> list[Keyword]:
    query = query.lower().replace("ё", "е")
    keywords = []
    date_words = r"(?:" + "|".join(MONTHS_SHORT) + r")[а-я]*"
    for match in re.finditer(r"\b(\d{1,2})\s+" + date_words, query):
        keywords.append(Keyword(re.compile(rf"(?<!\d){match.group(1)}\s+{match.group(0).split()[1][:3]}"), True))
    query = re.sub(r"\b\d{1,2}\s+" + date_words, " ", query)
    for word in re.findall(r"\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?|[\w/-]+", query):
        if any(c.isdigit() for c in word):
            # Codes and dates match whole: "27" must not hit "5132704", "ИВТ-22" not "ИВТ-21".
            keywords.append(Keyword(re.compile(rf"(?<![\w/]){re.escape(word)}(?![\w/])"), True))
        elif len(word) >= 3:
            # Crude stemming: Russian endings vary, so plain words match on their first 5 letters.
            keywords.append(Keyword(re.compile(re.escape(word[:5])), False))
    return keywords


def _matches(text: str, keywords: list[Keyword]) -> bool:
    lowered = text.lower().replace("ё", "е")
    return any(k.pattern.search(lowered) for k in keywords)


def _effective(keywords: list[Keyword], lines: list[str]) -> list[Keyword]:
    strong = [k for k in keywords if k.strong]
    if strong and any(_matches(line, strong) for line in lines):
        return strong
    return keywords


def select_text(lines: list[str], query: str | None, limit: int = MAX_TEXT_CHARS) -> str:
    keywords = _effective(_keywords(query or ""), lines)
    hits = [i for i, line in enumerate(lines) if keywords and _matches(line, keywords)]
    if not hits:
        chosen = list(range(len(lines)))
    else:
        keep = set()
        for i in hits:
            # A short hit is usually a header ("27 окт., вт"): take the block under it.
            after = 30 if len(lines[i]) <= 30 else 6
            keep.update(range(max(0, i - 2), min(len(lines), i + after)))
        chosen = sorted(keep)
    out, size, last = [], 0, None
    for i in chosen:
        if last is not None and i != last + 1:
            out.append("…")
        out.append(lines[i])
        size += len(lines[i]) + 1
        last = i
        if size >= limit:
            out.append("… (текст обрезан; уточни query, чтобы увидеть другую часть)")
            break
    return "\n".join(out)


def render_page(page: Page, query: str | None) -> str:
    parts = [f"Адрес: {page.url}"]
    if page.title:
        parts.append(f"Заголовок: {page.title}")

    keywords = _keywords(query or "")
    strong = [k for k in keywords if k.strong]
    exact = [(t, h) for t, h in page.links if strong and any(k.pattern.fullmatch(t.lower()) for k in strong)]
    if exact:
        parts.append("Точное совпадение: " + "; ".join(f"{t} → {h}" for t, h in exact[:3]))

    parts.append("Текст:\n" + (select_text(page.lines, query) or "(пусто — возможно, страница рисуется JavaScript)"))

    for name, options, action in page.selects:
        shown = [o for o in options if _matches(o[0], keywords)] if keywords else []
        shown = (shown or options)[:MAX_OPTIONS]
        rendered = "; ".join(f"{text}={value}" for text, value in shown)
        more = f" (всего {len(options)})" if len(options) > len(shown) else ""
        parts.append(f"Список «{name}» (форма {action}, параметр {name}){more}: {rendered}")
    return "\n\n".join(parts)


async def fetch_page(http: httpx.AsyncClient, url: str) -> tuple[str, str, str]:
    """Returns (final url, content type, body)."""
    if not re.match(r"^https?://", url):
        url = "https://" + url.lstrip("/")
    try:
        response = await http.get(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"})
    except httpx.HTTPError as error:
        raise ToolError(f"не удалось открыть {url}: {type(error).__name__}") from error
    if response.status_code >= 400:
        raise ToolError(f"{url} ответил {response.status_code}")
    return str(response.url), response.headers.get("content-type", ""), response.text


async def search(http: httpx.AsyncClient, query: str, limit: int = 6) -> list[tuple[str, str, str]]:
    try:
        response = await http.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "kl": "ru-ru"},
            headers={"User-Agent": USER_AGENT},
        )
    except httpx.HTTPError as error:
        raise ToolError(f"поиск недоступен: {type(error).__name__}") from error
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for item in soup.select(".result"):
        link = item.select_one("a.result__a")
        if not link or not link.get("href"):
            continue
        href = link["href"]
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        if "duckduckgo.com/y.js" in href:
            continue  # ads
        snippet = item.select_one(".result__snippet")
        results.append((_squash(link.get_text()), href, _squash(snippet.get_text()) if snippet else ""))
        if len(results) >= limit:
            break
    return results


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool(
        "Открыть веб-страницу и прочитать её текст. Ссылки записаны прямо в тексте как «название → адрес»: "
        "чтобы перейти, вызови снова с этим адресом.",
        {
            "url": {"type": "string"},
            "query": {"type": "string", "description": "Что ищешь на странице: код группы, дата, «завтра», день недели. Вернутся только подходящие части."},
            "params": {"type": "object", "description": "GET-параметры формы, например {\"group\": \"123\"} для выбора из списка."},
        },
        required=("url",),
    )
    async def web_fetch(url: str, query: str = "", params: dict | None = None) -> str:
        if params:
            url = url + ("&" if "?" in url else "?") + urlencode({k: str(v) for k, v in params.items()})
        query = expand_relative_dates(query, datetime.now(ctx.timezone))
        final_url, content_type, body = await fetch_page(ctx.http, url)
        if "json" in content_type:
            try:
                body = json.dumps(json.loads(body), ensure_ascii=False, indent=1)
            except ValueError:
                pass
            return f"Адрес: {final_url}\nJSON:\n" + select_text(body.splitlines(), query)
        if "html" not in content_type and "<html" not in body[:500].lower():
            return f"Адрес: {final_url}\nТекст:\n" + select_text(body.splitlines(), query)
        return render_page(parse_html(body, final_url), query)

    @reg.tool(
        "Поиск в интернете. Возвращает заголовки, адреса и описания результатов.",
        {"query": {"type": "string"}},
        required=("query",),
    )
    async def web_search(query: str) -> str:
        results = await search(ctx.http, query)
        if not results:
            return "Ничего не нашлось."
        return "\n".join(f"- {title} → {url}\n  {snippet}" for title, url, snippet in results)
