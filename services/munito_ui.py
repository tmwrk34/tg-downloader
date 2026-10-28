"""
Prompts, tool schemas, data models, Telegram UI cards, and Markdown-to-HTML
formatting for the Munito AI Assistant.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

DEFAULT_SYSTEM_PROMPT = (
    "You are Bard (Бард), a witty, observant, and genuinely helpful AI assistant and digital companion in this Telegram chat and tavern.\n\n"
    "Identity & Capabilities:\n"
    "- Your name is Bard (@bard_taverna_bot).\n"
    "- You are the heart of this tavern bot. Your actual capabilities include:\n"
    "  1. Media Downloader: Automatically downloads videos and audio from TikTok, YouTube, Instagram Reels/Posts, and Pinterest when users share links in the chat.\n"
    "  2. Tabletop Dice Roller: Rolls dice for tabletop RPGs (D&D, Pathfinder, etc.) via /roll (e.g. /roll 2d20+5, 1d100, 4d6kh3, or natural requests like 'кинь d20').\n"
    "  3. AI Assistant & Vision: You understand images/photos, answer questions, provide summaries, and maintain rolling conversation context memory per chat.\n"
    "  4. Tavern Chatter: You learn local chat lore and occasionally chime in with humor or reactions.\n"
    "- If asked who you are or what you can do, proudly describe your identity and features as Bard without mentioning underlying cloud or model providers (like Munito, DeepSeek, or Qwen) unless explicitly asked about technical architecture.\n\n"
    "Tool Usage:\n"
    "- You have access to real tools such as roll_dice to execute cryptographically secure tabletop dice rolls.\n"
    "- Whenever a user asks you to roll a die, make a skill check, roll for initiative/damage, or settle a dispute with dice, ALWAYS invoke the roll_dice tool instead of guessing or hallucinating numbers.\n\n"
    "Communication Style:\n"
    "- Always respond in the exact same language as the user's current prompt or message (e.g. if asked in Ukrainian, reply in Ukrainian; if asked in Russian, reply in Russian; if asked in English, reply in English).\n"
    "- Be witty, charismatic, and concise. Avoid robotic filler, repetitive disclaimers, or corporate apologies.\n"
    "- Format answers cleanly using Telegram-friendly HTML/Markdown when appropriate."
)

AI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": (
                "Roll tabletop RPG dice using standard notation (e.g. '1d20', '2d6+5', '4d6kh3', '1d100', '2d20kl1'). "
                "Evaluates real cryptographically secure random rolls with kept/dropped status, modifiers, and critical hit/fail detection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Standard dice notation expression, e.g. '1d20', '2d6+4', '3d8', '1d100', '4d6kh3'",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional in-game reason, action, or skill check name (e.g. 'stealth check', 'атака', 'перевірка уважності')",
                    },
                },
                "required": ["expression"],
            },
        },
    }
]

DEFAULT_MODEL = "deepseek-ai/deepseek-v4.1-flash-1017"

AVAILABLE_MODELS = [
    "deepseek-ai/deepseek-v4.1-flash-1017",
    "qwen/qwen3.8-27b",
    "deepseek-ai/deepseek-v4-flash",
    "zai-org/glm-5.3-flash",
    "google/gemma-4-12b",
]

REASONING_EFFORT_LEVELS = ["low", "medium", "high"]
CONTEXT_SIZE_OPTIONS = [0, 5, 10, 15, 20]
CONTEXT_MODE_OPTIONS = ["mentions", "all"]


@dataclass
class AIChatConfig:
    chat_id: int
    enabled: bool = True
    model: str = DEFAULT_MODEL
    system_prompt: str | None = None
    reasoning_effort: str = "low"
    context_mode: str = "mentions"  # "mentions" | "all"
    context_size: int = 10  # number of recent messages to inject (0 = disabled)


@dataclass(frozen=True)
class AIChatTurn:
    id: int
    chat_id: int
    user_id: int | None
    user_name: str | None
    role: str
    content: str
    has_image: bool
    created_at: float


@dataclass(frozen=True)
class AIChatStats:
    chat_id: int
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def get_next_model(current_model: str) -> str:
    """Cycle to the next available model."""
    try:
        idx = AVAILABLE_MODELS.index(current_model)
        return AVAILABLE_MODELS[(idx + 1) % len(AVAILABLE_MODELS)]
    except ValueError:
        return AVAILABLE_MODELS[0]


def get_next_reasoning_effort(current_effort: str) -> str:
    """Cycle reasoning effort levels: low -> medium -> high -> low."""
    try:
        idx = REASONING_EFFORT_LEVELS.index(current_effort.lower())
        return REASONING_EFFORT_LEVELS[(idx + 1) % len(REASONING_EFFORT_LEVELS)]
    except ValueError:
        return REASONING_EFFORT_LEVELS[0]


def get_next_context_size(current_size: int) -> int:
    """Cycle context size: 0 -> 5 -> 10 -> 15 -> 20 -> 0."""
    try:
        idx = CONTEXT_SIZE_OPTIONS.index(current_size)
        return CONTEXT_SIZE_OPTIONS[(idx + 1) % len(CONTEXT_SIZE_OPTIONS)]
    except ValueError:
        return 10


def get_next_context_mode(current_mode: str) -> str:
    """Toggle context collection mode between 'mentions' and 'all'."""
    return "all" if current_mode.lower() == "mentions" else "mentions"


def format_ai_card(
    chat_title: str,
    chat_id: int,
    cfg: AIChatConfig,
    stats: AIChatStats,
    history_count: int = 0,
) -> str:
    """Format AI assistant status, model, context memory, and token usage into an HTML message."""
    status_str = "🟢 Enabled" if cfg.enabled else "🔴 Disabled"
    effort_labels = {
        "low": "⚡ Low (Fast, ~15s)",
        "none": "⚡ Low (Fast, ~15s)",
        "medium": "⚖️ Medium (~35s)",
        "high": "🧠 High (Deep CoT, ~90s)",
    }
    effort_display = effort_labels.get(cfg.reasoning_effort, cfg.reasoning_effort)
    mode_labels = {
        "mentions": "🎯 Mentions Only",
        "all": "🌐 All Chat Messages",
    }
    mode_display = mode_labels.get(cfg.context_mode, cfg.context_mode)

    if cfg.context_size > 0:
        memory_str = f"🟢 Active ({history_count}/{cfg.context_size} msgs, {mode_display})"
    else:
        memory_str = "⚪ Disabled (0 msgs)"

    return (
        f"🤖 <b>Munito AI Assistant Settings</b>\n"
        f"<b>Chat:</b> <code>{chat_title}</code> (<code>{chat_id}</code>)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ <b>Status:</b> {status_str}\n"
        f"🧠 <b>Model:</b> <code>{cfg.model}</code>\n"
        f"💭 <b>Thinking:</b> <code>{effort_display}</code>\n"
        f"💬 <b>Memory:</b> <code>{memory_str}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Token Usage:</b>\n"
        f"  • Total requests: <b>{stats.request_count:,}</b>\n"
        f"  • Prompt tokens: <b>{stats.prompt_tokens:,}</b>\n"
        f"  • Generated tokens: <b>{stats.completion_tokens:,}</b>\n"
        f"  • Total tokens: <b>{stats.total_tokens:,}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Triggered exclusively via bot tag: @bot &lt;prompt&gt;</i>"
    )


def build_ai_keyboard(
    chat_id: int, cfg: AIChatConfig, confirm_reset: bool = False
) -> InlineKeyboardMarkup:
    """Construct inline keyboard for AI configuration."""
    toggle_text = "🔴 Disable" if cfg.enabled else "🟢 Enable"
    model_short = cfg.model.split("/")[-1]
    mode_short = "🎯 Mentions" if cfg.context_mode == "mentions" else "🌐 All Chat"
    ctx_text = f"💬 Context: {cfg.context_size} msgs" if cfg.context_size > 0 else "💬 Context: Off"

    rows = [
        [
            InlineKeyboardButton(
                text=f"Status: {toggle_text}",
                callback_data=f"ai:toggle:{chat_id}",
            )
        ],
        [
            InlineKeyboardButton(
                text=f"🧠 Model: {model_short}",
                callback_data=f"ai:model:{chat_id}",
            )
        ],
        [
            InlineKeyboardButton(
                text=f"💭 Thinking: {cfg.reasoning_effort.title()}",
                callback_data=f"ai:effort:{chat_id}",
            )
        ],
        [
            InlineKeyboardButton(
                text=ctx_text,
                callback_data=f"ai:ctx_size:{chat_id}",
            ),
            InlineKeyboardButton(
                text=f"Mode: {mode_short}",
                callback_data=f"ai:ctx_mode:{chat_id}",
            ),
        ],
        [
            InlineKeyboardButton(
                text="🧠 Forget Context",
                callback_data=f"ai:clear_mem:{chat_id}",
            )
        ],
    ]

    if confirm_reset:
        rows.append(
            [
                InlineKeyboardButton(
                    text="⚠️ Confirm Reset Stats",
                    callback_data=f"ai:reset_confirm:{chat_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Cancel",
                    callback_data=f"ai:refresh:{chat_id}",
                ),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🧹 Reset Token Stats",
                    callback_data=f"ai:reset_prompt:{chat_id}",
                )
            ]
        )

    rows.append(
        [
            InlineKeyboardButton(
                text="🔄 Refresh",
                callback_data=f"ai:refresh:{chat_id}",
            )
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=rows)


def split_telegram_message(text: str, max_chars: int = 4000) -> list[str]:
    """
    Split a long response into chunks suitable for Telegram message limits (4096 chars).
    Prefers splitting by double newlines, single newlines, or spaces.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = text

    while len(current) > max_chars:
        split_at = current.rfind("\n\n", 0, max_chars)
        if split_at == -1 or split_at < max_chars // 2:
            split_at = current.rfind("\n", 0, max_chars)
        if split_at == -1 or split_at < max_chars // 2:
            split_at = current.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars

        chunks.append(current[:split_at].strip())
        current = current[split_at:].strip()

    if current:
        chunks.append(current)

    return chunks


def markdown_to_telegram_html(text: str) -> str:
    """
    Convert Markdown text from LLM completions into valid Telegram HTML format.
    Safely escapes raw HTML characters outside of tags, formats code blocks,
    inline code, bold, italic, strikethrough, links (including URLs with balanced parentheses),
    and bullet points.
    """
    if not text:
        return ""

    placeholders: list[str] = []

    def save_placeholder(content: str) -> str:
        idx = len(placeholders)
        placeholders.append(content)
        return f"\x00CODE{idx}\x00"

    # 1. Protect code blocks: ```lang\ncode\n```
    def replace_code_block(match: re.Match[str]) -> str:
        lang = match.group(1) or ""
        code = match.group(2)
        code_escaped = html.escape(code, quote=False)
        if lang:
            tag = f'<pre><code class="language-{html.escape(lang.strip(), quote=True)}">{code_escaped}</code></pre>'
        else:
            tag = f"<pre><code>{code_escaped}</code></pre>"
        return save_placeholder(tag)

    result = re.sub(r"```([a-zA-Z0-9_\-\+]*)\n?([\s\S]*?)```", replace_code_block, text)

    # 2. Protect inline code: `code`
    def replace_inline_code(match: re.Match[str]) -> str:
        code = match.group(1)
        return save_placeholder(f"<code>{html.escape(code, quote=False)}</code>")

    result = re.sub(r"`([^`\n]+)`", replace_inline_code, result)

    # 3. Escape HTML characters in remaining text (&, <, >)
    result = html.escape(result, quote=False)

    # 4. Headers: # Title -> <b>Title</b>
    result = re.sub(r"(?m)^#{1,6}\s+(.+)$", r"<b>\1</b>", result)

    # 5. Bold: **text** or __text__
    result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"__(.+?)__", r"<b>\1</b>", result)

    # 6. Italic: *text* or _text_
    result = re.sub(r"(?<!\w)\*([^\*\n]+?)\*(?!\w)", r"<i>\1</i>", result)
    result = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", result)

    # 7. Strikethrough: ~~text~~
    result = re.sub(r"~~(.+?)~~", r"<s>\1</s>", result)

    # 8. Links: [label](url) supporting balanced parentheses inside URLs (e.g. Wikipedia)
    def replace_link(match: re.Match[str]) -> str:
        label = match.group(1)
        url = match.group(2)
        return f'<a href="{url}">{label}</a>'

    result = re.sub(
        r"\[([^\]]+)\]\((https?://(?:[^\s\(\)]|\([^\s\(\)]*\))+)\)",
        replace_link,
        result,
    )

    # 9. List bullets: convert leading "- " or "* " to clean Unicode bullet "• "
    result = re.sub(r"(?m)^[\*\-]\s+", "• ", result)

    # 10. Restore code blocks and inline code
    for idx, content in enumerate(placeholders):
        result = result.replace(f"\x00CODE{idx}\x00", content)

    return result
