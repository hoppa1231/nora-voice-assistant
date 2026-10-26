"""Weather from Open-Meteo: free, no API key."""

from __future__ import annotations

import httpx

from . import ToolContext, ToolError, ToolRegistry

WMO = {
    0: "ясно", 1: "в основном ясно", 2: "переменная облачность", 3: "пасмурно",
    45: "туман", 48: "изморозь", 51: "слабая морось", 53: "морось", 55: "сильная морось",
    56: "ледяная морось", 57: "сильная ледяная морось", 61: "небольшой дождь", 63: "дождь",
    65: "сильный дождь", 66: "ледяной дождь", 67: "сильный ледяной дождь", 71: "небольшой снег",
    73: "снег", 75: "сильный снег", 77: "снежная крупа", 80: "ливень", 81: "сильный ливень",
    82: "очень сильный ливень", 85: "снегопад", 86: "сильный снегопад", 95: "гроза",
    96: "гроза с градом", 99: "сильная гроза с градом",
}
DAYS = ("сегодня", "завтра", "послезавтра")
CITY_ALIASES = {
    "питер": "Санкт-Петербург", "спб": "Санкт-Петербург", "петербург": "Санкт-Петербург",
    "мск": "Москва", "екб": "Екатеринбург", "ебург": "Екатеринбург", "нижний": "Нижний Новгород",
    "нск": "Новосибирск", "ростов": "Ростов-на-Дону",
}


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool(
        "Погода сейчас и на 3 дня.",
        {"city": {"type": "string", "description": "Город. Если пользователь не назвал — не передавай."}},
    )
    async def get_weather(city: str = "") -> str:
        city = city.strip() or ctx.default_city
        city = CITY_ALIASES.get(city.lower().removeprefix("в ").strip(), city)
        if not city:
            raise ToolError("город не указан и DEFAULT_CITY не задан — спроси пользователя")
        try:
            geo = await ctx.http.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 5, "language": "ru"},
            )
            places = geo.json().get("results") or []
            if not places:
                raise ToolError(f"не нашёл город «{city}»")
            # Many villages share a city's name; the user almost always means the big one.
            place = max(places, key=lambda p: p.get("population") or 0)
            forecast = await ctx.http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "timezone": "auto",
                    "forecast_days": 3,
                    "wind_speed_unit": "ms",
                },
            )
            data = forecast.json()
        except httpx.HTTPError as error:
            raise ToolError(f"сервис погоды недоступен: {type(error).__name__}") from error

        now = data["current"]
        lines = [
            f"{place['name']}, сейчас: {WMO.get(now['weather_code'], 'без описания')}, "
            f"{now['temperature_2m']:+.0f}°C (ощущается {now['apparent_temperature']:+.0f}°C), "
            f"ветер {now['wind_speed_10m']:.0f} м/с."
        ]
        daily = data["daily"]
        for i, day in enumerate(DAYS[: len(daily["time"])]):
            rain = daily["precipitation_probability_max"][i]
            lines.append(
                f"{day.capitalize()}: {WMO.get(daily['weather_code'][i], 'без описания')}, "
                f"от {daily['temperature_2m_min'][i]:+.0f} до {daily['temperature_2m_max'][i]:+.0f}°C"
                + (f", вероятность осадков {rain}%" if rain is not None else "")
                + "."
            )
        return "\n".join(lines)
