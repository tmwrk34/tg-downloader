# tg-downloader

A high-performance Telegram bot built with **aiogram 3** and **yt-dlp** for downloading media across multiple platforms, featuring local Telegram Bot API server integration for 2GB payload uploads and a tabletop D&D dice rolling engine.

---

## Features

### 🎬 Media Downloader
- **TikTok**:
  - Downloads videos with clean audio and metadata.
  - Resolves short links (`vt.tiktok.com`, `vm.tiktok.com`) automatically.
  - **Photo Mode / Slideshows** (`/photo/`): Downloads image albums as Telegram carousels / media groups alongside the original background music track.
- **Instagram**:
  - Supports Reels, Posts, and Carousels (`/reel/`, `/p/`).
  - Downloads multi-item galleries and carousels seamlessly.
  - Authenticated session support via Netscape cookies (`data/cookies.txt` or `INSTAGRAM_COOKIES` secret) to bypass empty media response restrictions.
- **YouTube**:
  - **YouTube Shorts (`/shorts/`)**: Direct, instant download in the highest available quality without prompting.
  - **Standard Videos (`/watch?v=...`, `youtu.be/...`)**: Interactive inline keyboard to choose resolutions (`1080p`, `720p`, `360p`) or extract audio (`🎵 Audio MP3` with thumbnail cover and tags).
  - Automatically cleans tracking parameters and playlist arguments (`&list=...`, `&index=...`).
- **Pinterest**:
  - Supports standard image pins, video pins, and multi-page idea pin (story/carousel) albums.
  - Automatically resolves shortlinks (`pin.it/...`) with in-memory LRU caching.
  - Native yt-dlp extraction with automatic direct API scraper fallback for unauthenticated image and story pins.

### 🚀 2GB Upload Mode
- Bundled **local Telegram Bot API server** container (`aiogram/telegram-bot-api:latest`) configured via `docker-compose.yml`.
- Bypasses the default Telegram cloud 50MB bot upload limit, enabling file transfers up to **2000MB (2GB)**.
- Automatically activated when `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` are provided in the environment.

### ⚡ High-Speed Caching Layer & Bounded Concurrency
- **Instant Media Redelivery**: Caches Telegram `file_id` for uploaded videos, photos, audios, and carousel albums in a two-tier cache (in-memory L1 LRU cache + persistent SQLite WAL database at `data/media_cache.db`).
- **Zero I/O Cache Hits**: Bypasses `yt-dlp` and `ffmpeg` transcoding on duplicate or reposted URLs, delivering media within milliseconds (< 200ms).
- **Bounded Worker Queue & Singleflight Deduplication**: Limits simultaneous `yt-dlp` downloads (`MAX_CONCURRENT_DOWNLOADS=2`) and `ffmpeg` transcodes (`MAX_CONCURRENT_TRANSCODES=1`) via asyncio semaphores so Raspberry Pi CPU/RAM are never exhausted during burst traffic, while keeping `/roll`, `/help`, and cached links instantaneous. Simultaneous requests for the exact same link are deduplicated via a singleflight lock.
- **Native Telegram Queue Feedback**: Reacts with `⏳` when a request enters the queue, transitions to `⚡` with periodic `ChatAction` (`uploading video...` / `recording voice...`) during active processing, and clears reactions upon completion.
- **Aggressive URL Normalization**: Automatically strips platform tracking parameters (`?igsh=...`, `?si=...`, `?rpxt=...`, `&utm_source=...`) and resolves shortlinks (`vt.tiktok.com`, `vm.tiktok.com`) with an in-memory LRU cache so different share links pointing to the same media map to the same cache key.
- **Graceful Error Recovery**: If a cached `file_id` is invalidated or expired by Telegram, the bot purges the entry and falls back to a fresh download automatically.

### 🎲 Tabletop D&D Dice Roller
- **Commands**:
  - `/roll [expression]` (aliases: `/ролл`, `/кинь`, `/кидок`) — roll dice publicly (defaults to `d20`).
  - `/dmroll [expression]` (aliases: `/дмролл`, `/дм`, `/секретролл`) — roll secret dice. In group chats, posts an interactive `[ 👁 View Secret Roll ]` button so only the roller can view the result in an alert dialog.
- **Tabletop Notation Support**:
  - **Basic**: `2d6+3`, `d20`, `1d20-2`.
  - **Cyrillic Transliteration**: Full support for `д` and `к` notation (e.g. `2д20`, `2к20`, `д20`, `к20`).
  - **Text Modifiers**: `p` / `п` for plus (`2d6p3`, `2д6п3`), `m` / `м` for minus (`2d6m3`, `2к6м2`).
  - **Keep Dice**: `4d6k3` / `4к6к3` / `4d6kh3` (keep 3 highest), `4d6kl2` / `4к6кн2` (keep 2 lowest).
  - **Drop Dice**: `4d6d2` / `4к6d2` / `4d6dl2` (drop 2 lowest), `4d6dh1` / `4к6св1` (drop 1 highest).
  - **Combined Modifiers**: `4d6k3+2`, `4к6к3+2`, `4d6d2m3`.
- **Natural Language Parsing**:
  - Plain chat input: `2d20`, `d6`, `4d6k3+2`, `2д20`, `к20`.
  - Action phrases: `roll 2d6`, `throw d20`, `roll 20 sided dice`, `кинь 2д20`, `ролл 2к20`.
  - Secret rolls: `dm roll 2d20`, `hidden roll 4d6k3`, `дм ролл 2д20`.
- **Inline Mode (`@bot ...`)**:
  - Type `@bot` in any chat to view standard presets (`d20`, `Advantage (2d20kh1)`, `Disadvantage (2d20kl1)`, `d100`, `d12`, `d10`, `d8`, `d6`, `d4`).
  - Type any custom formula (e.g. `@bot 2d6+3`) to evaluate on the fly and share the result card.
- **Visual Feedback & Rules**:
  - Strikethrough for dropped dice in roll breakdown: `[6, 5, 4, <s>2</s>] + 2`.
  - D&D Critical badges: `✨ CRITICAL HIT! (Natural 20) 🎯` and `💀 CRITICAL FAIL! (Natural 1) 💥`.
  - Strict English UI localization.

### 🎭 Tavern Chatterbot (SLM / Markov Engine)
- **Zero-CPU In-Memory / SQLite Markov Chain**: A lightweight chatterbot that passively learns from natural group conversation and generates amusing, context-inspired sentences without heavyweight LLMs or PyTorch dependencies (running comfortably on Raspberry Pi).
- **Per-Chat Isolation & In-Chat Configuration**:
  - Disabled by default. Enabled and configured **directly inside Telegram** via the `/chatter` command without modifying environment variables or restarting the bot.
  - Interactive Inline Keyboard to toggle state, adjust reply frequency, emoji reaction chance, cooldown, and manage vocabulary.
  - Commands:
    - `/chatter` (aliases: `/бард`, `/дурачок`, `/балабол`) — Open interactive configuration card.
    - `/chatter on` / `/chatter off` — Enable or disable chatterbot in the current chat.
    - `/chatter chance <0-100>` — Set random message response chance (e.g. `5` for 5%).
    - `/chatter react <0-100>` — Set random emoji reaction chance (e.g. `10` for 10%).
    - `/chatter cooldown <seconds>` — Set anti-spam cooldown between spontaneous messages (defaults to `30s`).
    - `/chatter learn on/off` — Toggle passive vocabulary training.
    - `/chatter clear` — Wipe the current chat's learned vocabulary.
- **Natural Triggers & Reactions**:
  - Always replies (100%) when mentioned (`@bot`, "бард", "дурачок") or when a user replies to the bot's message.
  - Uses candidate seed words from the user's message to anchor responses relevant to the ongoing conversation.
  - Reacts with thematic tavern emojis (`🤡`, `🎲`, `🍺`, `🔥`, `🗿`, `⚡`, `🫡`, `🤔`, etc.).

### 🤖 Munito AI Assistant (Grok / DeepSeek)
- **Tag & Reply Triggers**: Mention `@bard_taverna_bot <prompt>` or reply to any photo/message to invoke the AI assistant.
- **Multimodal Vision**: Accurately analyzes photos and reply images using EU-hosted open models.
- **Adjustable Reasoning Effort (CoT)**:
  - `⚡ Low (~15s)` (default): Limits chain-of-thought generation tokens, speeding up response times by 75% compared to unbounded reasoning.
  - `⚖️ Medium (~35s)`: Balanced thinking for deeper mathematical or code queries.
  - `🧠 High (~90s)`: Full chain-of-thought exploration for complex questions.
- **Dynamic Model Switching**: Switch models on the fly between `deepseek-ai/deepseek-v4.1-flash-1017`, `deepseek-ai/deepseek-v4-flash`, `qwen/qwen3.8-27b`, `meta-llama/llama-4-scout-17b-16k`, and `google/gemma-4-12b`.
- **Telegram HTML Markdown Interpreter**: Formats headings, bold, italic, code blocks, bullet points, and inline links cleanly for Telegram with safe character escaping and plain-text fallback.
- **Token Accounting & Settings (`/ai`)**:
  - `/ai` — Interactive settings card with inline buttons to toggle state, switch models, adjust thinking effort, and view token usage.
  - `/ai on` / `/ai off` — Enable or disable AI assistant for the chat.
  - `/ai model <name>` — Set the active model.
  - `/ai effort <low|medium|high>` — Set reasoning effort level.
  - `/ai reset` — Reset chat token metrics.

### 🔒 Security & Access Control
- **Whitelist Middleware**: Restricts all interactions strictly to chat IDs specified in `ALLOWED_CHAT_IDS`. Unauthorized messages are dropped silently.
- **Admin Authorization**: Configuration commands (`/chatter`, `/ai`) are strictly limited to chat administrators and bot superadmins.

---

## Architecture & Codebase Layout

The project follows a modular, decoupled architecture using **aiogram 3** routers and persistent SQLite databases in WAL mode:

```
tg-downloader/
├── bot.py                  # Application entry point: service setup, middleware, polling loop (~200 lines)
├── config.py               # Environment variables, type parsing, and safety bounds
│
├── handlers/               # aiogram 3 domain routers
│   ├── base.py             # Basic commands: /start, /help, /ping, /id
│   ├── downloader.py       # Media pipeline: URL detection, queueing, cache delivery, file upload
│   ├── dice.py             # Tabletop dice roller: /roll, /dmroll, inline queries
│   ├── chatter.py          # Tavern chatterbot settings card and callbacks
│   └── ai.py               # Munito AI: mentions, /ai settings, vision, group message router
│
├── downloader/             # Media extraction engine (yt-dlp wrappers and platform scrapers)
│   ├── core.py             # MediaDownloaderCore (download, format selection, ffmpeg compression)
│   ├── models.py           # Typed dataclasses (DownloadedMedia, MediaMetadata)
│   └── platforms/          # Dedicated platform modules (TikTok, Pinterest, YouTube, Instagram)
│
├── services/               # Background business logic & database managers
│   ├── cache.py            # SQLite URL -> file_id caching layer (media_cache.db)
│   ├── worker.py           # Concurrency limiting, bounded task queues, and singleflight locks
│   ├── dice.py             # D&D dice formula parser and roll engine
│   ├── chatter.py          # Markov chain text generator & vocabulary trainer (chatter.db)
│   └── munito.py           # Munito AI client, HTML parser, and token accounting (ai.db)
│
├── filters/                # Declarative message filters (admin, dice, media URLs, triggers)
├── middlewares/            # Whitelist security middleware
└── utils/                  # Cross-cutting utilities (typing action manager, photo downloader)
```

For complete architectural details, see [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md).

---

## Installation & Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tmwrk34/tg-downloader.git
   cd tg-downloader
   ```

2. **Install dependencies using [uv](https://github.com/astral-sh/uv)**:
   ```bash
   uv sync
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your BOT_TOKEN and ALLOWED_CHAT_IDS
   ```

4. **Run the bot**:
   ```bash
   uv run python bot.py
   ```

---

## Docker Deployment

To run the bot alongside the local Telegram Bot API server in Docker:

1. **Configure `.env`**:
   ```bash
   cp .env.example .env
   # Set BOT_TOKEN, ALLOWED_CHAT_IDS, TELEGRAM_API_ID, TELEGRAM_API_HASH
   chmod 600 .env
   ```

2. **Start services with Docker Compose**:
   ```bash
   docker compose up -d --build
   ```

3. **View live logs**:
   ```bash
   docker compose logs -f
   ```

4. **Stop services**:
   ```bash
   docker compose down
   ```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `BOT_TOKEN` | Yes | Telegram Bot API token obtained from `@BotFather`. |
| `ALLOWED_CHAT_IDS` | Yes | Comma-separated list of allowed user or group IDs (e.g. `1234567,-100123456789`). |
| `TELEGRAM_API_ID` | Optional | Telegram App API ID from [my.telegram.org](https://my.telegram.org) (enables 2GB upload mode). |
| `TELEGRAM_API_HASH` | Optional | Telegram App API Hash from [my.telegram.org](https://my.telegram.org) (enables 2GB upload mode). |
| `LOCAL_BOT_API_URL` | Optional | Custom URL for local Bot API server (defaults to `http://telegram-bot-api:8081`). |
| `INSTAGRAM_COOKIES` | Optional | Netscape formatted cookies content for Instagram authentication. |
| `COOKIES_FILE` | Optional | Path to a Netscape cookie file (e.g. `data/cookies.txt`). |
| `MAX_CONCURRENT_DOWNLOADS` | Optional | Maximum simultaneous `yt-dlp` downloads (defaults to `2`). |
| `MAX_CONCURRENT_TRANSCODES` | Optional | Maximum simultaneous `ffmpeg` transcoding tasks (defaults to `1`). |
| `MUNITO_API_KEY` | Optional | Munito AI API key for Grok-style AI assistant features. |
| `MUNITO_MODEL` | Optional | Model identifier (defaults to `deepseek-ai/deepseek-v4.1-flash-1017`). |
| `MUNITO_BASE_URL` | Optional | Munito AI base URL (defaults to `https://api.munito.ai/inference/v1`). |
| `MUNITO_SYSTEM_PROMPT` | Optional | Custom system instructions override for AI interactions. |

---

## Continuous Deployment (CI/CD)

The project includes automated deployment to a self-hosted runner (e.g. Raspberry Pi) on push to `main` via `.github/workflows/deploy.yml`:
- Injects repository secrets (`BOT_TOKEN`, `ALLOWED_CHAT_IDS`, `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `INSTAGRAM_COOKIES`).
- Automatically compiles and installs `data/cookies.txt`.
- Rebuilds and launches the containers with zero downtime.

---

## Testing & Quality Assurance

Enable the included Git hooks to enforce Conventional Commits and branch naming:

```bash
git config core.hooksPath .githooks
# or run
./.githooks/install.sh
```

Run linting, formatting, and unit tests:

```bash
# Code linting
uv run ruff check .

# Formatting check
uv run ruff format --check .

# Run test suite
uv run pytest
```
