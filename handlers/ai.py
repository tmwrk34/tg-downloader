from __future__ import annotations

import html
import logging
import re

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from filters.admin import is_chat_admin
from services.munito import (
    AVAILABLE_MODELS,
    build_ai_keyboard,
    format_ai_card,
    format_memory_card,
    get_next_context_mode,
    get_next_context_size,
    get_next_model,
    get_next_reasoning_effort,
    markdown_to_telegram_html,
    split_telegram_message,
)
from services.worker import set_message_reaction
from utils.proxy import bot_proxy as bot
from utils.responses import safe_reply
from utils.telegram_media import (
    download_telegram_file_id,
    download_telegram_photo,
    select_optimal_photo,
)
from utils.typing import keep_typing

logger = logging.getLogger(__name__)

ai_router = Router(name="ai")


async def _extract_bot_tag_prompt(message: Message, active_bot: Bot) -> tuple[bool, str]:
    """Check if message explicitly contains @bot_username tag and extract prompt."""
    bot_user = await bot._get_cached_bot_user(active_bot)
    if not bot_user:
        return False, ""

    username = getattr(bot_user, "username", None)
    if not isinstance(username, str) or not username.strip():
        return False, ""

    raw_text = message.text or message.caption or ""
    if not raw_text:
        return False, ""

    pattern = rf"@{re.escape(username)}\b"
    if not re.search(pattern, raw_text, re.IGNORECASE):
        return False, ""

    cleaned_prompt = re.sub(pattern, "", raw_text, flags=re.IGNORECASE).strip()
    return True, cleaned_prompt


async def _handle_ai_mention(message: Message, prompt: str, active_bot: Bot) -> None:
    """Handle explicit @bot_username tag using Munito AI with text or vision context."""
    if not bot.munito_service.is_configured:
        await safe_reply(
            message,
            "⚠️ Munito AI is not configured on the server (missing <code>MUNITO_API_KEY</code>).",
            parse_mode="HTML",
        )
        return

    context_text: str | None = None
    image_bytes: bytes | None = None

    # Check if this message is a reply to another message
    if message.reply_to_message:
        replied = message.reply_to_message
        if replied.photo:
            optimal_photo = select_optimal_photo(replied.photo)
            image_bytes = await download_telegram_photo(active_bot, optimal_photo)
            if replied.caption:
                context_text = replied.caption
        elif replied.text:
            context_text = replied.text
    elif message.photo:
        optimal_photo = select_optimal_photo(message.photo)
        image_bytes = await download_telegram_photo(active_bot, optimal_photo)

    if not prompt and not context_text and not image_bytes:
        await safe_reply(
            message,
            "Hello! I am connected to an AI model. Ask me a question or tag me in reply to a message or photo.",
        )
        return

    chat_id = message.chat.id
    image_mime = "image/jpeg" if image_bytes else None
    user_id = message.from_user.id if message.from_user else None
    user_name = message.from_user.first_name if message.from_user else None

    try:
        async with keep_typing(active_bot, chat_id, action="typing"):
            reply_text = await bot.munito_service.generate_reply(
                prompt=prompt,
                context_text=context_text,
                image_bytes=image_bytes,
                image_mime=image_mime,
                chat_id=chat_id,
                user_id=user_id,
                user_name=user_name,
            )
    except Exception as e:
        logger.exception("Munito AI generation failed")
        err_msg = (
            str(e).strip()
            if str(e).strip()
            else f"{type(e).__name__} (request timed out or failed)"
        )
        await safe_reply(message, f"❌ AI Error: {err_msg}")
        return

    raw_chunks = split_telegram_message(reply_text)
    for idx, raw_chunk in enumerate(raw_chunks):
        formatted_chunk = markdown_to_telegram_html(raw_chunk)
        try:
            if idx == 0:
                await safe_reply(message, formatted_chunk, parse_mode="HTML")
            else:
                await message.answer(formatted_chunk, parse_mode="HTML")
        except Exception as e:
            logger.warning("Failed to send formatted HTML chunk: %s, falling back to plain text", e)
            if idx == 0:
                await safe_reply(message, raw_chunk)
            else:
                await message.answer(raw_chunk)


@ai_router.message(Command("ai", "munito", "grok", "assistant", "асистент"))
async def ai_command_handler(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id if message.from_user else 0
    chat_id = message.chat.id
    chat_title = message.chat.title or message.chat.first_name or "Private Chat"

    if not await is_chat_admin(message.bot, chat_id, user_id):
        await safe_reply(message, "⚠️ Only chat administrators can configure the AI assistant.")
        return

    args = (command.args or "").strip().split()
    cfg = await bot.munito_service.get_chat_config(chat_id)

    if args:
        subcmd = args[0].lower()
        if subcmd in {"on", "start", "enable", "1", "увімк", "вкл"}:
            cfg = await bot.munito_service.update_chat_config(chat_id, enabled=True)
        elif subcmd in {"off", "stop", "disable", "0", "вимк", "викл"}:
            cfg = await bot.munito_service.update_chat_config(chat_id, enabled=False)
        elif subcmd in {"model", "модель"} and len(args) > 1:
            req_model = args[1].strip()
            matched = None
            for m in AVAILABLE_MODELS:
                if req_model.lower() == m.lower() or req_model.lower() == m.split("/")[-1].lower():
                    matched = m
                    break
            if matched:
                cfg = await bot.munito_service.update_chat_config(chat_id, model=matched)
            else:
                models_list = "\n".join(f"• <code>{m}</code>" for m in AVAILABLE_MODELS)
                await safe_reply(
                    message,
                    f"⚠️ Unknown model: <code>{req_model}</code>.\nAvailable models:\n{models_list}",
                    parse_mode="HTML",
                )
                return
        elif subcmd in {"models", "моделі"}:
            models_list = "\n".join(f"• <code>{m}</code>" for m in AVAILABLE_MODELS)
            await safe_reply(
                message,
                f"🧠 <b>Available Munito AI models:</b>\n{models_list}\n\n"
                f"Current: <code>{cfg.model}</code>\n"
                f"Change: <code>/ai model &lt;name&gt;</code> or use the inline buttons.",
                parse_mode="HTML",
            )
            return
        elif subcmd in {"effort", "thinking", "reasoning"} and len(args) > 1:
            req_effort = args[1].lower()
            if req_effort in {"low", "none", "medium", "high"}:
                eff = "low" if req_effort == "none" else req_effort
                cfg = await bot.munito_service.update_chat_config(chat_id, reasoning_effort=eff)
            else:
                await safe_reply(
                    message,
                    "⚠️ Invalid thinking level. Choose one of: <code>low</code> (fast, ~15s), "
                    "<code>medium</code> (balanced, ~35s), <code>high</code> (deep reasoning, ~90s).",
                    parse_mode="HTML",
                )
                return
        elif subcmd in {"context", "контекст", "memory", "пам'ять"} and len(args) > 1:
            try:
                val = max(0, min(30, int(args[1].strip())))
                cfg = await bot.munito_service.update_chat_config(chat_id, context_size=val)
            except ValueError:
                await safe_reply(
                    message,
                    "⚠️ Invalid context size. Enter a number between 0 and 30 messages (e.g. <code>/ai context 10</code>).",
                    parse_mode="HTML",
                )
                return
        elif subcmd in {"mode", "режим"} and len(args) > 1:
            req_mode = args[1].lower().strip()
            if req_mode in {"mentions", "згадки", "mention", "tag", "тег"}:
                cfg = await bot.munito_service.update_chat_config(chat_id, context_mode="mentions")
            elif req_mode in {"all", "всі", "все", "чат", "chat"}:
                cfg = await bot.munito_service.update_chat_config(chat_id, context_mode="all")
            else:
                await safe_reply(
                    message,
                    "⚠️ Invalid mode. Choose <code>mentions</code> (record only @bot) or <code>all</code> (record all chat messages).",
                    parse_mode="HTML",
                )
                return
        elif subcmd in {"forget", "забудь", "clear_memory", "clearmem", "очистити_пам'ять"}:
            deleted = await bot.munito_service.clear_chat_history(chat_id)
            stats = await bot.munito_service.get_chat_stats(chat_id)
            mem_stats = await bot.munito_service.get_memory_stats(chat_id)
            text = format_ai_card(
                chat_title, chat_id, cfg, stats, history_count=0, memory_stats=mem_stats
            )
            keyboard = build_ai_keyboard(chat_id, cfg)
            await safe_reply(
                message,
                f"🧹 <b>Conversation memory wiped!</b> (Removed {deleted} historical messages)\n\n"
                + text,
                reply_markup=keyboard,
                parse_mode="HTML",
            )
            return
        elif subcmd in {"reset", "скинути", "clear"}:
            await bot.munito_service.reset_chat_stats(chat_id)
            stats = await bot.munito_service.get_chat_stats(chat_id)
            history_count = await bot.munito_service.get_history_count(chat_id)
            mem_stats = await bot.munito_service.get_memory_stats(chat_id)
            text = format_ai_card(
                chat_title,
                chat_id,
                cfg,
                stats,
                history_count=history_count,
                memory_stats=mem_stats,
            )
            keyboard = build_ai_keyboard(chat_id, cfg)
            await safe_reply(
                message,
                "🧹 <b>Token usage statistics reset!</b>\n\n" + text,
                reply_markup=keyboard,
                parse_mode="HTML",
            )
            return

    stats = await bot.munito_service.get_chat_stats(chat_id)
    history_count = await bot.munito_service.get_history_count(chat_id)
    mem_stats = await bot.munito_service.get_memory_stats(chat_id)
    text = format_ai_card(
        chat_title, chat_id, cfg, stats, history_count=history_count, memory_stats=mem_stats
    )
    keyboard = build_ai_keyboard(chat_id, cfg)
    await safe_reply(message, text, reply_markup=keyboard, parse_mode="HTML")


@ai_router.message(Command("memory", "lore", "досьє", "память", "пам'ять"))
async def memory_command_handler(message: Message, command: CommandObject | None = None) -> None:
    """Inspect, search, add, or clear long-term hybrid tavern memories and member dossiers."""
    import html as html_lib

    chat_id = message.chat.id
    chat_title = message.chat.title or message.chat.first_name or "Private Chat"
    user_id = message.from_user.id if message.from_user else 0
    user_name = message.from_user.first_name if message.from_user else "Traveler"

    raw_args = (command.args or "").strip() if command else ""
    parts = raw_args.split(maxsplit=1)
    subcmd = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if subcmd in {"search", "find", "пошук"}:
        if not rest:
            await safe_reply(
                message,
                "🔍 Usage: <code>/memory search &lt;query&gt;</code>",
                parse_mode="HTML",
            )
            return
        results = await bot.munito_service.search_memories(
            chat_id, query=rest, top_k=5, min_score=0.30
        )
        if not results:
            await safe_reply(
                message,
                f"🔍 <b>Tavern Archives:</b> No matching memories found for <code>{html_lib.escape(rest)}</code>.",
                parse_mode="HTML",
            )
            return
        lines = [f"🔍 <b>Search Results for</b> <code>{html_lib.escape(rest)}</code>:"]
        for m in results:
            subj = f"<b>[{html_lib.escape(m.subject)}]</b> " if m.subject else ""
            lines.append(
                f"• ({m.category}, score={m.score:.2f}) {subj}{html_lib.escape(m.content)}"
            )
        await safe_reply(message, "\n".join(lines), parse_mode="HTML")
        return

    if subcmd in {"remember", "add", "save", "запам'ятай", "додати"}:
        if not rest:
            await safe_reply(
                message,
                "📝 Usage: <code>/memory remember &lt;fact or lore&gt;</code>",
                parse_mode="HTML",
            )
            return
        entry, is_new = await bot.munito_service.remember_fact(
            chat_id=chat_id,
            content=rest,
            category="lore",
            subject="Лор чату",
            participants=user_name,
        )
        status_label = "Recorded new lore" if is_new else "Updated existing lore"
        await safe_reply(
            message,
            f"📜 <b>{status_label} (#{entry.id}):</b> {html_lib.escape(entry.content)}",
            parse_mode="HTML",
        )
        return

    if subcmd in {"dedup", "дедуп"} or (
        subcmd in {"dossier", "dossiers", "досьє"} and rest.lower() == "dedup"
    ):
        cleaned = await bot.munito_service.deduplicate_user_dossiers(chat_id)
        dossiers = await bot.munito_service.get_user_dossiers(chat_id, limit=10)
        lines = [
            f"🧹 <b>Dossier Deduplication Complete!</b> (Cleaned/merged {cleaned} items)\n",
            "🎭 <b>Cleaned Member Dossiers:</b>",
        ]
        if dossiers:
            for d in dossiers:
                uname = f" (@{html.escape(d.username)})" if d.username else ""
                traits = "; ".join(html.escape(t) for t in d.traits)
                lines.append(f"• <b>{html.escape(d.display_name)}</b>{uname}: {traits}")
        else:
            lines.append("<i>No member dossiers recorded yet.</i>")
        await safe_reply(message, "\n".join(lines), parse_mode="HTML")
        return

    if subcmd in {"forget", "delete", "remove", "видалити", "забудь"}:
        if not rest:
            await safe_reply(
                message,
                "🗑 Usage: <code>/memory forget &lt;query or trait&gt;</code>",
                parse_mode="HTML",
            )
            return
        removed_memories = await bot.munito_service.forget_memory(chat_id, rest)
        removed_dossier_traits = await bot.munito_service.forget_dossier_traits(chat_id, rest)
        total = len(removed_memories) + len(removed_dossier_traits)
        if total == 0:
            await safe_reply(
                message,
                f"ℹ️ Nothing found to forget matching <code>{html.escape(rest)}</code> in lore or dossiers.",
                parse_mode="HTML",
            )
            return
        lines = [f"🗑 <b>Forgotten {total} items matching</b> <code>{html.escape(rest)}</code>:"]
        for m in removed_memories:
            lines.append(f"• [Lore/Episode] {html.escape(m.content)}")
        for dt in removed_dossier_traits:
            lines.append(f"• [Dossier: {html.escape(dt['user_name'])}] {html.escape(dt['trait'])}")
        await safe_reply(message, "\n".join(lines), parse_mode="HTML")
        return

    if subcmd in {"dossier", "dossiers", "досьє"}:
        d_parts = rest.split(maxsplit=2)
        if d_parts and d_parts[0].lower() in {"delete", "remove", "del", "видалити"}:
            target_user = d_parts[1] if len(d_parts) > 1 else ""
            trait_query = d_parts[2] if len(d_parts) > 2 else None
            if not target_user:
                await safe_reply(
                    message,
                    "Usage: <code>/memory dossier delete &lt;user&gt; [trait keyword]</code>",
                    parse_mode="HTML",
                )
                return
            success, count, _ = await bot.munito_service.delete_user_dossier(
                chat_id, target_user, trait_query
            )
            if not success:
                await safe_reply(
                    message,
                    f"⚠️ No dossier found for <code>{html.escape(target_user)}</code>.",
                    parse_mode="HTML",
                )
                return
            if trait_query:
                await safe_reply(
                    message,
                    f"🗑 Removed {count} trait(s) matching <code>{html.escape(trait_query)}</code> from <b>{html.escape(target_user)}</b>'s dossier.",
                    parse_mode="HTML",
                )
            else:
                await safe_reply(
                    message,
                    f"🗑 Completely deleted dossier for <b>{html.escape(target_user)}</b>.",
                    parse_mode="HTML",
                )
            return

        dossiers = await bot.munito_service.get_user_dossiers(
            chat_id=chat_id,
            active_user_id=user_id,
            active_user_name=user_name,
            limit=10,
        )
        if not dossiers:
            await safe_reply(
                message,
                "🎭 <b>Tavern Dossiers:</b> No member traits recorded in this chat yet.",
                parse_mode="HTML",
            )
            return
        lines = ["🎭 <b>Tavern Member Dossiers:</b>"]
        for d in dossiers:
            uname = f" (@{html_lib.escape(d.username)})" if d.username else ""
            traits = "; ".join(html_lib.escape(t) for t in d.traits)
            lines.append(f"• <b>{html_lib.escape(d.display_name)}</b>{uname}: {traits}")
        await safe_reply(message, "\n".join(lines), parse_mode="HTML")
        return

    if subcmd in {"clear", "wipe", "reset", "очистити"}:
        if not await is_chat_admin(message.bot, chat_id, user_id):
            await safe_reply(
                message,
                "⚠️ Only chat administrators can wipe long-term tavern archives.",
            )
            return
        deleted = await bot.munito_service.clear_long_term_memory(chat_id)
        await safe_reply(
            message,
            f"🧹 <b>Long-term tavern archives wiped!</b> (Removed {deleted} entries)",
            parse_mode="HTML",
        )
        return

    mem_stats = await bot.munito_service.get_memory_stats(chat_id)
    lore_items = await bot.munito_service.get_chat_lore(chat_id, limit=6)
    dossiers = await bot.munito_service.get_user_dossiers(
        chat_id, active_user_id=user_id, active_user_name=user_name, limit=5
    )
    card_html = format_memory_card(chat_title, chat_id, mem_stats, lore_items, dossiers)
    await safe_reply(message, card_html, parse_mode="HTML")


@ai_router.message(Command("prompt", "sysprompt", "системник"))
async def prompt_command_handler(message: Message, command: CommandObject) -> None:
    """View, download, update, or reset the AI assistant's system prompt."""
    chat_id = message.chat.id
    user_id = message.from_user.id if message.from_user else 0
    raw_args = (command.args or "").strip()
    parts = raw_args.split(maxsplit=1)
    subcommand = parts[0].lower() if parts else ""
    extra_text = parts[1].strip() if len(parts) > 1 else ""

    # 1. Download as file: /prompt file or /prompt -f or /prompt doc
    if subcommand in ("file", "-f", "--file", "doc", "document"):
        current_prompt = bot.munito_service.get_system_prompt()
        is_custom = bot.munito_service.is_custom_prompt()
        source_label = (
            "Custom (data/system_prompt.txt)"
            if is_custom
            else "Factory Default (prompts/system_prompt.txt)"
        )
        doc = BufferedInputFile(
            current_prompt.encode("utf-8"),
            filename="system_prompt.txt",
        )
        caption = (
            f"📜 <b>Active System Prompt</b>\n"
            f"• <b>Status:</b> <code>{source_label}</code>\n"
            f"• <b>Length:</b> {len(current_prompt)} chars ({len(current_prompt.splitlines())} lines)"
        )
        await message.reply_document(doc, caption=caption, parse_mode="HTML")
        return

    # 2. Set/update prompt: /prompt set <text> (Admin only)
    if subcommand in ("set", "update", "save"):
        if not await is_chat_admin(message.bot, chat_id, user_id):
            await safe_reply(
                message,
                "⛔ <b>Access Denied:</b> Only chat administrators can modify the system prompt.",
                parse_mode="HTML",
            )
            return

        new_prompt = extra_text
        # If no inline text, check replied message
        if not new_prompt and message.reply_to_message:
            replied = message.reply_to_message
            if replied.document and message.bot:
                doc_bytes = await download_telegram_file_id(message.bot, replied.document.file_id)
                if doc_bytes:
                    new_prompt = doc_bytes.decode("utf-8", errors="replace").strip()
            elif replied.text:
                new_prompt = replied.text.strip()
            elif replied.caption:
                new_prompt = replied.caption.strip()

        if not new_prompt:
            await safe_reply(
                message,
                "ℹ️ <b>System Prompt Usage:</b>\n\n"
                "• <code>/prompt set &lt;new prompt text&gt;</code>\n"
                "• Reply to a text message with <code>/prompt set</code>\n"
                "• Reply to a <code>.txt</code> file with <code>/prompt set</code>",
                parse_mode="HTML",
            )
            return

        saved_path = bot.munito_service.set_system_prompt(new_prompt)
        await safe_reply(
            message,
            "✅ <b>System Prompt Updated &amp; Persisted!</b>\n\n"
            f"📁 <b>Saved to:</b> <code>{saved_path}</code>\n"
            f"📊 <b>Length:</b> {len(new_prompt)} chars ({len(new_prompt.splitlines())} lines)\n"
            "🔄 <b>Persistence:</b> Changes are saved to persistent storage and will survive container redeployments and restarts.",
            parse_mode="HTML",
        )
        return

    # 3. Reset prompt: /prompt reset (Admin only)
    if subcommand in ("reset", "revert", "factory"):
        if not await is_chat_admin(message.bot, chat_id, user_id):
            await safe_reply(
                message,
                "⛔ <b>Access Denied:</b> Only chat administrators can reset the system prompt.",
                parse_mode="HTML",
            )
            return

        reloaded = bot.munito_service.reset_system_prompt()
        await safe_reply(
            message,
            "🔄 <b>System Prompt Reset to Factory Default!</b>\n\n"
            "📁 <b>Template:</b> <code>prompts/system_prompt.txt</code>\n"
            f"📊 <b>Length:</b> {len(reloaded)} chars ({len(reloaded.splitlines())} lines)",
            parse_mode="HTML",
        )
        return

    # 4. Help: /prompt help or /prompt ?
    if subcommand in ("help", "?", "довідка", "хелп"):
        is_custom = bot.munito_service.is_custom_prompt()
        source_label = (
            "Custom (data/system_prompt.txt)"
            if is_custom
            else "Factory Default (prompts/system_prompt.txt)"
        )
        help_text = (
            "📜 <b>System Prompt Management</b>\n\n"
            f"• <b>Current Source:</b> <code>{source_label}</code>\n\n"
            "<b>Commands:</b>\n"
            "• <code>/prompt</code> or <code>/prompt get</code> — View current active prompt\n"
            "• <code>/prompt file</code> — Download prompt as <code>.txt</code> document\n"
            "• <code>/prompt set &lt;text&gt;</code> — 👑 Update prompt (saved to persistent storage)\n"
            "• Reply to a message or <code>.txt</code> file with <code>/prompt set</code> — 👑 Set prompt from reply\n"
            "• <code>/prompt reset</code> — 👑 Revert to factory default prompt"
        )
        await safe_reply(message, help_text, parse_mode="HTML")
        return

    # 5. Default action: view active system prompt (/prompt or /prompt get)
    current_prompt = bot.munito_service.get_system_prompt()
    is_custom = bot.munito_service.is_custom_prompt()
    source_label = (
        "Custom (data/system_prompt.txt)"
        if is_custom
        else "Factory Default (prompts/system_prompt.txt)"
    )

    if len(current_prompt) <= 3800:
        escaped_prompt = html.escape(current_prompt)
        text = (
            f"📜 <b>Active System Prompt</b> [<code>{source_label}</code>]\n"
            f"📊 {len(current_prompt)} chars | {len(current_prompt.splitlines())} lines\n\n"
            f'<pre><code class="language-markdown">{escaped_prompt}</code></pre>\n\n'
            "<i>Tip: Use <code>/prompt file</code> to download as a .txt file, or <code>/prompt set &lt;text&gt;</code> (Admin) to update.</i>"
        )
        await safe_reply(message, text, parse_mode="HTML")
    else:
        doc = BufferedInputFile(
            current_prompt.encode("utf-8"),
            filename="system_prompt.txt",
        )
        caption = (
            f"📜 <b>Active System Prompt</b> [<code>{source_label}</code>]\n\n"
            f"📊 <b>Length:</b> {len(current_prompt)} chars ({len(current_prompt.splitlines())} lines)\n"
            "<i>(Prompt exceeds 3,800 characters and was sent as a text document)</i>"
        )
        await message.reply_document(doc, caption=caption, parse_mode="HTML")


@ai_router.callback_query(F.data.startswith("ai:"))
async def ai_callback_handler(callback: CallbackQuery) -> None:
    if not callback.data:
        return

    parts = callback.data.split(":")
    if len(parts) < 3:
        return

    _, action, chat_id_str = parts[:3]
    try:
        target_chat_id = int(chat_id_str)
    except ValueError:
        return

    user_id = callback.from_user.id if callback.from_user else 0
    if not await is_chat_admin(callback.bot, target_chat_id, user_id):
        await callback.answer(
            "⚠️ Only chat administrators can modify AI settings.",
            show_alert=True,
        )
        return

    cfg = await bot.munito_service.get_chat_config(target_chat_id)
    confirm_reset = False

    if action == "toggle":
        cfg = await bot.munito_service.update_chat_config(target_chat_id, enabled=not cfg.enabled)
        await callback.answer(f"AI Assistant: {'Enabled' if cfg.enabled else 'Disabled'}")
    elif action == "model":
        next_m = get_next_model(cfg.model)
        cfg = await bot.munito_service.update_chat_config(target_chat_id, model=next_m)
        await callback.answer(f"Model: {next_m.split('/')[-1]}")
    elif action == "effort":
        next_eff = get_next_reasoning_effort(cfg.reasoning_effort)
        cfg = await bot.munito_service.update_chat_config(target_chat_id, reasoning_effort=next_eff)
        await callback.answer(f"Thinking: {next_eff.title()}")
    elif action == "ctx_size":
        next_size = get_next_context_size(cfg.context_size)
        cfg = await bot.munito_service.update_chat_config(target_chat_id, context_size=next_size)
        await callback.answer(
            f"Context: {next_size} msgs" if next_size > 0 else "Context: Disabled"
        )
    elif action == "ctx_mode":
        next_mode = get_next_context_mode(cfg.context_mode)
        cfg = await bot.munito_service.update_chat_config(target_chat_id, context_mode=next_mode)
        await callback.answer(
            f"Mode: {'🎯 Mentions' if next_mode == 'mentions' else '🌐 All Chat'}"
        )
    elif action == "clear_mem":
        deleted = await bot.munito_service.clear_chat_history(target_chat_id)
        await callback.answer(f"🧹 Memory wiped ({deleted} msgs deleted)!", show_alert=True)
    elif action == "reset_prompt":
        confirm_reset = True
        await callback.answer()
    elif action == "reset_confirm":
        await bot.munito_service.reset_chat_stats(target_chat_id)
        await callback.answer("Token statistics have been reset!", show_alert=True)
    elif action == "refresh":
        await callback.answer("Refreshed!")

    target_msg = callback.message if isinstance(callback.message, Message) else None
    if target_msg:
        chat_title = target_msg.chat.title or target_msg.chat.first_name or "Chat"
        stats = await bot.munito_service.get_chat_stats(target_chat_id)
        history_count = await bot.munito_service.get_history_count(target_chat_id)
        mem_stats = await bot.munito_service.get_memory_stats(target_chat_id)
        text = format_ai_card(
            chat_title,
            target_chat_id,
            cfg,
            stats,
            history_count=history_count,
            memory_stats=mem_stats,
        )
        keyboard = build_ai_keyboard(target_chat_id, cfg, confirm_reset=confirm_reset)
        try:
            await target_msg.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        except Exception:
            pass


@ai_router.message(F.text | F.caption)
async def chatter_message_listener(message: Message, active_bot: Bot | None = None) -> None:
    """
    Message listener for group conversation:
    1. If explicitly tagged with @bot_username, routes to Munito AI assistant.
    2. If replying directly to a message from the bot and Munito AI is enabled, routes to Munito AI.
    3. If AI memory mode is 'all', passively records chat message into conversation memory.
    4. Otherwise, passes through to passive Markov chatterbot engine.
    """
    raw_text = message.text or message.caption or ""
    if not raw_text or raw_text.startswith("/"):
        return

    current_bot = active_bot or message.bot
    bot_user = await bot._get_cached_bot_user(current_bot) if current_bot else None

    # 1. Explicit mention check for Munito AI assistant
    if current_bot:
        is_tag, prompt = await _extract_bot_tag_prompt(message, current_bot)
        if is_tag:
            await _handle_ai_mention(message, prompt, current_bot)
            return

    is_reply_to_bot = bool(
        message.reply_to_message
        and message.reply_to_message.from_user
        and bot_user
        and message.reply_to_message.from_user.id == bot_user.id
    )

    # 2. Direct reply to a bot message routes to Munito AI when configured & enabled
    if is_reply_to_bot and current_bot and bot.munito_service.is_configured:
        try:
            ai_cfg = await bot.munito_service.get_chat_config(message.chat.id)
            if ai_cfg.enabled:
                await _handle_ai_mention(message, raw_text.strip(), current_bot)
                return
        except Exception as e:
            logger.debug("Failed checking AI config on direct reply: %s", e)

    # 3. Passive conversational memory capture when mode is 'all'
    if bot.munito_service.is_configured:
        try:
            ai_cfg = await bot.munito_service.get_chat_config(message.chat.id)
            if ai_cfg.enabled and ai_cfg.context_size > 0 and ai_cfg.context_mode == "all":
                sender_name = message.from_user.first_name if message.from_user else "User"
                sender_id = message.from_user.id if message.from_user else None
                rec_text = raw_text
                if message.photo:
                    rec_text = f"[Photo attached] {raw_text}".strip()
                await bot.munito_service.record_turn(
                    chat_id=message.chat.id,
                    role="user",
                    content=rec_text,
                    user_id=sender_id,
                    user_name=sender_name,
                    has_image=bool(message.photo),
                )
        except Exception as e:
            logger.debug("Failed to passively record message in AI context: %s", e)

    # 4. Only text messages without @bot tag participate in Markov chatterbot
    if not message.text:
        return

    # Passive vocabulary learning
    await bot.chatter_service.train(message.chat.id, message.text)

    cfg = await bot.chatter_service.get_config(message.chat.id)
    if not cfg.enabled:
        return

    # Random emoji reactions
    if bot.chatter_service.should_react(cfg):
        reaction_emoji = bot.chatter_service.pick_reaction()
        await set_message_reaction(message, reaction_emoji)

    # Check chatterbot keywords or reply to the bot without tag
    text_lower = message.text.lower()
    is_chatter_keyword = bool("бард" in text_lower or "дурачок" in text_lower)

    if bot.chatter_service.should_reply(
        cfg, is_mention_or_reply=(is_chatter_keyword or is_reply_to_bot)
    ):
        reply_sentence = await bot.chatter_service.generate(message.chat.id, seed_text=message.text)
        if reply_sentence:
            await bot.chatter_service.record_reply(message.chat.id)
            await safe_reply(message, reply_sentence)
        elif is_chatter_keyword or is_reply_to_bot:
            await safe_reply(
                message,
                "<i>*Blinks confusedly...*</i>\n"
                "(My vocabulary in this tavern is still empty! Talk more so I can learn to speak.)",
                parse_mode="HTML",
            )


__all__ = [
    "ai_callback_handler",
    "ai_command_handler",
    "ai_router",
    "memory_command_handler",
    "prompt_command_handler",
]
