# Нора / Джарвиз — голосовой ассистент

Русскоязычный ассистент для Linux: микрофон → Silero VAD → GigaAM → локальная LLM в LM Studio → Silero TTS. Два профиля с отдельной памятью: Нора и Джарвиз.

Умеет ставить таймеры и напоминания, запоминать факты, управлять музыкой и громкостью, искать информацию и читать сайты. Дополнительно может передавать задачи установленному Codex или Claude CLI.

## Требования

- Linux, Python **3.12**, [uv](https://docs.astral.sh/uv/getting-started/installation/), Git.
- [LM Studio](https://lmstudio.ai/) с загруженной моделью и включённым OpenAI-совместимым сервером.
- Для голоса: микрофон, колонки/наушники и PortAudio. Для управления звуком и эхоподавления — PipeWire/PulseAudio, `pactl`, `wpctl`; для музыки — `playerctl`.
- Индикатор требует Wayland с поддержкой layer-shell, системный Python с GTK3, PyGObject, Cairo и gtk-layer-shell. На других окружениях задайте `INDICATOR=off`.

Пример системных пакетов для Arch Linux:

```bash
sudo pacman -S --needed git portaudio libsndfile pipewire-pulse wireplumber playerctl python-gobject python-cairo gtk3 gtk-layer-shell
```

Интеграция окон и рабочих столов рассчитана на Hyprland. Работа на Windows/macOS не проверена. Зависимости Python используют CPU-сборку PyTorch; дискретная NVIDIA не требуется. Потребление памяти зависит прежде всего от выбранной LLM.

## Быстрый старт

```bash
git clone https://github.com/hoppa1231/nora-voice-assistant.git
cd nora-voice-assistant
uv sync --locked
cp .env.example .env
```

В LM Studio загрузите `qwen3-4b-instruct-2507` или другую совместимую модель, включите сервер на `http://localhost:1234/v1`. В `.env` укажите её точный идентификатор в `LM_MODEL` (остальные параметры подключения: `LM_BASE_URL`, `LM_API_KEY`).

Сначала проверьте текстовый режим:

```bash
uv run python -m voice_assistant --text --mute
```

Затем запустите голосовой профиль:

```bash
uv run python -m voice_assistant --profile nora
# либо
uv run python -m voice_assistant --profile jarvis
```

Обращение: «Нора, поставь таймер на минуту» или «Джарвиз, какая погода?». Выход — `Ctrl+C`. При первом голосовом запуске скачиваются модели распознавания и синтеза, поэтому нужен интернет. Для отдельного потокового детектора перебивания Vosk:

```bash
uv sync --locked --extra barge-in
uv run python -m voice_assistant.barge --download-model
```

## Настройка и данные

Все параметры с пояснениями — в [.env.example](.env.example). Память и таймеры сохраняются в `data/assistant.db` и `data/jarvis/assistant.db`. Модели — в `models/`. Эти каталоги и `.env` исключены из Git: у друга будет собственная память и настройки.

Основной режим использует локальную LLM и Silero. Поиск, погода и чтение сайтов обращаются в интернет; при выборе Edge TTS текст отправляется сервису озвучки. Codex/Claude требуют отдельной установки и авторизации; связанные функции можно отключить через `DISABLED_TOOLS=ask_coder`.

[Подробное руководство](docs/guide.md): команды, настройка микрофона, перебивания, профили, инструменты, устройство проекта и ограничения. Приведённые там замеры относятся к машине разработки; скорость на другом компьютере может отличаться.

## Проверка

```bash
uv sync --locked
uv run pytest -q
```

Автотесты проверяют программную логику. Качество микрофона, эхоподавления и распознавания в конкретной комнате нужно проверять отдельно.
