import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.filters import CommandObject
from aiogram.types import CallbackQuery, Chat, Message, User

from services.munito import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    AIChatConfig,
    AIChatStats,
    MunitoAIService,
    build_ai_keyboard,
    format_ai_card,
    get_next_context_mode,
    get_next_context_size,
    get_next_model,
    markdown_to_telegram_html,
    split_telegram_message,
)


def test_split_telegram_message():
    # Short message
    assert split_telegram_message("Hello world") == ["Hello world"]

    # Exactly max chars
    short = "a" * 100
    assert split_telegram_message(short, max_chars=100) == [short]

    # Message splitting on paragraphs
    p1 = "Paragraph 1 " * 20
    p2 = "Paragraph 2 " * 20
    full = f"{p1}\n\n{p2}"
    chunks = split_telegram_message(full, max_chars=len(p1) + 10)
    assert len(chunks) == 2
    assert chunks[0] == p1.strip()
    assert chunks[1] == p2.strip()

    # Message splitting on space when no newline
    long_line = "word " * 50
    chunks_line = split_telegram_message(long_line, max_chars=100)
    assert len(chunks_line) > 1
    assert "".join(chunks_line).replace(" ", "") == long_line.replace(" ", "")


def test_markdown_to_telegram_html():
    # Empty
    assert markdown_to_telegram_html("") == ""

    # Bold and Italic
    md = "**bold text** and *italic text* and __other bold__ and _other italic_"
    expected = (
        "<b>bold text</b> and <i>italic text</i> and <b>other bold</b> and <i>other italic</i>"
    )
    assert markdown_to_telegram_html(md) == expected

    # Headers and Bullets
    md_list = "### Header Title\n- Item 1\n* Item 2"
    html_list = markdown_to_telegram_html(md_list)
    assert "<b>Header Title</b>" in html_list
    assert "• Item 1" in html_list
    assert "• Item 2" in html_list

    # Inline code and HTML escaping
    md_code = "Use `<tag>` & `x > y` in `script.py`"
    html_code = markdown_to_telegram_html(md_code)
    assert "<code>&lt;tag&gt;</code>" in html_code
    assert "&amp;" in html_code
    assert "<code>x &gt; y</code>" in html_code

    # Code blocks
    md_block = "```python\nprint('<hello>')\n```"
    html_block = markdown_to_telegram_html(md_block)
    assert (
        "<pre><code class=\"language-python\">print('&lt;hello&gt;')\n</code></pre>" in html_block
    )

    # Links
    md_link = "[Munito AI](https://munito.ai)"
    assert markdown_to_telegram_html(md_link) == '<a href="https://munito.ai">Munito AI</a>'


def test_munito_is_configured():
    svc_empty = MunitoAIService(api_key=None)
    assert not svc_empty.is_configured

    svc_whitespace = MunitoAIService(api_key="   ")
    assert not svc_whitespace.is_configured

    svc_valid = MunitoAIService(api_key="munitoai_testkey123")
    assert svc_valid.is_configured


@pytest.mark.asyncio
async def test_munito_generate_reply_unconfigured():
    svc = MunitoAIService(api_key=None)
    with pytest.raises(RuntimeError, match="MUNITO_API_KEY is missing"):
        await svc.generate_reply("Hello")


@pytest.mark.asyncio
async def test_munito_generate_reply_text_success():
    svc = MunitoAIService(api_key="test_key")

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Це відповідь від штучного інтелекту.",
                }
            }
        ]
    }
    mock_resp.__aenter__.return_value = mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__.return_value = mock_session

    with patch("aiohttp.ClientSession", return_value=mock_session):
        reply = await svc.generate_reply("Хто ти?")
        assert reply == "Це відповідь від штучного інтелекту."

        call_args = mock_session.post.call_args
        assert call_args is not None
        _, kwargs = call_args
        json_data = kwargs.get("json", {})
        assert json_data["model"] == "deepseek-ai/deepseek-v4.1-flash-1017"
        assert len(json_data["messages"]) == 2
        assert json_data["messages"][0]["role"] == "system"
        assert json_data["messages"][1]["role"] == "user"
        assert json_data["messages"][1]["content"][0]["text"] == "Хто ти?"


@pytest.mark.asyncio
async def test_munito_generate_reply_reasoning_fallback():
    svc = MunitoAIService(api_key="test_key")

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "Thought process result.",
                }
            }
        ]
    }
    mock_resp.__aenter__.return_value = mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__.return_value = mock_session

    with patch("aiohttp.ClientSession", return_value=mock_session):
        reply = await svc.generate_reply("Think about this")
        assert reply == "Thought process result."


@pytest.mark.asyncio
async def test_munito_generate_reply_with_context_and_image():
    svc = MunitoAIService(api_key="test_key")

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "На картинці зображено кота.",
                }
            }
        ]
    }
    mock_resp.__aenter__.return_value = mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__.return_value = mock_session

    with patch("aiohttp.ClientSession", return_value=mock_session):
        fake_bytes = b"fake_jpeg_image_data"
        reply = await svc.generate_reply(
            prompt="Що на зображенні?",
            context_text="Попереднє повідомлення користувача",
            image_bytes=fake_bytes,
        )
        assert reply == "На картинці зображено кота."

        call_args = mock_session.post.call_args
        _, kwargs = call_args
        user_content = kwargs["json"]["messages"][1]["content"]

        # Should contain both text (with context and prompt) and image_url
        assert len(user_content) == 2
        assert user_content[0]["type"] == "text"
        assert "Попереднє повідомлення користувача" in user_content[0]["text"]
        assert "Що на зображенні?" in user_content[0]["text"]

        assert user_content[1]["type"] == "image_url"
        assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_munito_error_statuses():
    svc = MunitoAIService(api_key="test_key")

    for status_code, err_substr in [
        (401, "Invalid or unauthorized Munito API key"),
        (429, "rate limit reached"),
        (503, "overloaded"),
    ]:
        mock_resp = AsyncMock()
        mock_resp.status = status_code
        mock_resp.__aenter__.return_value = mock_resp

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__.return_value = mock_session

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            pytest.raises(RuntimeError, match=err_substr),
        ):
            await svc.generate_reply("Test")


@pytest.mark.asyncio
async def test_extract_bot_tag_prompt():
    from bot import _extract_bot_tag_prompt

    bot_mock = AsyncMock()
    bot_user = User(id=999, is_bot=True, first_name="BardBot", username="bard_downloader_bot")
    bot_mock.get_me.return_value = bot_user

    # Mention in message
    msg_tagged = MagicMock(spec=Message)
    msg_tagged.text = "@bard_downloader_bot переклади цей текст"
    msg_tagged.caption = None
    is_tag, prompt = await _extract_bot_tag_prompt(msg_tagged, bot_mock)
    assert is_tag is True
    assert prompt == "переклади цей текст"

    # Mention with extra spaces / mixed case
    msg_case = MagicMock(spec=Message)
    msg_case.text = "Привіт @BARD_DOWNLOADER_BOT як справи?"
    msg_case.caption = None
    is_tag2, prompt2 = await _extract_bot_tag_prompt(msg_case, bot_mock)
    assert is_tag2 is True
    assert prompt2 == "Привіт  як справи?"

    # No mention
    msg_untagged = MagicMock(spec=Message)
    msg_untagged.text = "Привіт усім у чаті"
    msg_untagged.caption = None
    is_tag3, prompt3 = await _extract_bot_tag_prompt(msg_untagged, bot_mock)
    assert is_tag3 is False
    assert prompt3 == ""

    # Caption mention
    msg_caption = MagicMock(spec=Message)
    msg_caption.text = None
    msg_caption.caption = "@bard_downloader_bot describe image"
    is_tag4, prompt4 = await _extract_bot_tag_prompt(msg_caption, bot_mock)
    assert is_tag4 is True
    assert prompt4 == "describe image"


@pytest.mark.asyncio
async def test_handle_ai_mention_unconfigured():
    import bot
    from bot import _handle_ai_mention

    msg = AsyncMock()
    bot_mock = AsyncMock()

    with patch.object(bot.munito_service, "api_key", None):
        await _handle_ai_mention(msg, "Hello", bot_mock)
        assert msg.reply.call_count == 1
        assert "MUNITO_API_KEY" in msg.reply.call_args[0][0]


@pytest.mark.asyncio
async def test_download_telegram_photo_local_file_and_fallback(tmp_path: Path):
    from bot import _download_telegram_photo

    local_img = tmp_path / "photo_local.jpg"
    local_img.write_bytes(b"local_image_bytes")

    bot_mock = AsyncMock()
    fake_photo = MagicMock()
    fake_photo.file_id = "f_123"

    # Case 1: local file on disk
    mock_file_local = MagicMock()
    mock_file_local.file_path = str(local_img)
    bot_mock.get_file = AsyncMock(return_value=mock_file_local)

    res_local = await _download_telegram_photo(bot_mock, fake_photo)
    assert res_local == b"local_image_bytes"

    # Case 2: standard bot.download fallback
    import io

    mock_file_remote = MagicMock()
    mock_file_remote.file_path = "photos/file_999.jpg"
    bot_mock.get_file = AsyncMock(return_value=mock_file_remote)
    bot_mock.download = AsyncMock(return_value=io.BytesIO(b"remote_image_bytes"))

    res_remote = await _download_telegram_photo(bot_mock, fake_photo)
    assert res_remote == b"remote_image_bytes"


@pytest.mark.asyncio
async def test_handle_ai_mention_with_reply_photo():
    import io

    import bot
    from bot import _handle_ai_mention

    msg = AsyncMock()
    msg.chat.id = 12345
    msg.photo = None

    replied_msg = AsyncMock()
    fake_photo = MagicMock()
    replied_msg.photo = [fake_photo]
    replied_msg.text = None
    replied_msg.caption = "Original photo caption"
    msg.reply_to_message = replied_msg

    bot_mock = AsyncMock()
    mock_file_info = MagicMock()
    mock_file_info.file_path = "photos/test.jpg"
    bot_mock.get_file = AsyncMock(return_value=mock_file_info)
    bot_mock.download = AsyncMock(return_value=io.BytesIO(b"binary_photo_data"))

    with (
        patch.object(bot.munito_service, "api_key", "valid_key"),
        patch.object(
            bot.munito_service,
            "generate_reply",
            new_callable=AsyncMock,
            return_value="AI analyzed this image.",
        ) as mock_gen,
    ):
        await _handle_ai_mention(msg, "переклади це", bot_mock)

        assert mock_gen.call_count == 1
        _, kwargs = mock_gen.call_args
        assert kwargs["prompt"] == "переклади це"
        assert kwargs["context_text"] == "Original photo caption"
        assert kwargs["image_bytes"] == b"binary_photo_data"
        assert msg.reply.call_count == 1
        assert msg.reply.call_args[0][0] == "AI analyzed this image."


@pytest.mark.asyncio
async def test_handle_ai_mention_with_reply_text():
    import bot
    from bot import _handle_ai_mention

    msg = AsyncMock()
    msg.chat.id = 12345
    msg.photo = None

    replied_msg = AsyncMock()
    replied_msg.photo = None
    replied_msg.text = "Here is some code in python"
    msg.reply_to_message = replied_msg

    bot_mock = AsyncMock()

    with (
        patch.object(bot.munito_service, "api_key", "valid_key"),
        patch.object(
            bot.munito_service,
            "generate_reply",
            new_callable=AsyncMock,
            return_value="Here is the explanation.",
        ) as mock_gen,
    ):
        await _handle_ai_mention(msg, "explain this", bot_mock)

        assert mock_gen.call_count == 1
        _, kwargs = mock_gen.call_args
        assert kwargs["prompt"] == "explain this"
        assert kwargs["context_text"] == "Here is some code in python"
        assert kwargs["image_bytes"] is None
        assert kwargs["chat_id"] == 12345
        assert msg.reply.call_args[0][0] == "Here is the explanation."


@pytest.mark.asyncio
async def test_munito_db_config_and_usage(tmp_path: Path):
    db_file = tmp_path / "test_ai.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)

    # 1. Default config
    cfg = await svc.get_chat_config(1001)
    assert cfg.chat_id == 1001
    assert cfg.enabled is True
    assert cfg.model == DEFAULT_MODEL

    # 2. Update config
    updated = await svc.update_chat_config(1001, enabled=False, model="qwen/qwen3.8-27b")
    assert updated.enabled is False
    assert updated.model == "qwen/qwen3.8-27b"

    fetched = await svc.get_chat_config(1001)
    assert fetched.enabled is False
    assert fetched.model == "qwen/qwen3.8-27b"

    # 3. Usage stats
    stats_empty = await svc.get_chat_stats(1001)
    assert stats_empty.request_count == 0
    assert stats_empty.total_tokens == 0

    # Record usage
    stats1 = await svc.record_usage(1001, prompt_tokens=100, completion_tokens=50, total_tokens=150)
    assert stats1.request_count == 1
    assert stats1.prompt_tokens == 100
    assert stats1.completion_tokens == 50
    assert stats1.total_tokens == 150

    # Record another usage
    stats2 = await svc.record_usage(1001, prompt_tokens=50, completion_tokens=25, total_tokens=75)
    assert stats2.request_count == 2
    assert stats2.prompt_tokens == 150
    assert stats2.completion_tokens == 75
    assert stats2.total_tokens == 225

    # 4. Reset stats
    await svc.reset_chat_stats(1001)
    stats_reset = await svc.get_chat_stats(1001)
    assert stats_reset.request_count == 0
    assert stats_reset.total_tokens == 0

    svc.close()


def test_munito_models_and_cards():
    assert len(AVAILABLE_MODELS) >= 3
    assert DEFAULT_MODEL == AVAILABLE_MODELS[0]

    # Model cycling
    m1 = AVAILABLE_MODELS[0]
    m2 = get_next_model(m1)
    assert m2 == AVAILABLE_MODELS[1]
    last_m = AVAILABLE_MODELS[-1]
    assert get_next_model(last_m) == AVAILABLE_MODELS[0]
    assert get_next_model("unknown/model") == AVAILABLE_MODELS[0]

    # Format card
    cfg = AIChatConfig(chat_id=123, enabled=True, model=DEFAULT_MODEL)
    stats = AIChatStats(
        chat_id=123, request_count=5, prompt_tokens=500, completion_tokens=200, total_tokens=700
    )
    card = format_ai_card("D&D Chat", 123, cfg, stats)
    assert "Munito AI Assistant Settings" in card
    assert "🟢 Enabled" in card
    assert DEFAULT_MODEL in card
    assert "700" in card

    # Build keyboard
    kb = build_ai_keyboard(123, cfg)
    assert any("Disable" in btn.text for row in kb.inline_keyboard for btn in row)
    assert any(
        DEFAULT_MODEL.split("/")[-1] in btn.text for row in kb.inline_keyboard for btn in row
    )

    kb_confirm = build_ai_keyboard(123, cfg, confirm_reset=True)
    assert any("Confirm Reset" in btn.text for row in kb_confirm.inline_keyboard for btn in row)


@pytest.mark.asyncio
async def test_munito_command_and_callback_handlers(tmp_path: Path):
    from bot import (
        ai_callback_handler,
        ai_command_handler,
    )

    # Use a temporary DB for tests
    test_svc = MunitoAIService(api_key="valid_key", db_path=tmp_path / "bot_ai.db")
    chat_id = -100555666
    chat = Chat(id=chat_id, type="supergroup", title="AI Test Tavern")
    user = User(id=98765, is_bot=False, first_name="AdminPlayer")

    bot = MagicMock()
    bot.get_me = AsyncMock(
        return_value=User(id=999999, is_bot=True, first_name="BardBot", username="bard_bot")
    )
    member_mock = MagicMock()
    member_mock.status = "administrator"
    bot.get_chat_member = AsyncMock(return_value=member_mock)

    msg = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="/ai",
    ).as_(bot)

    with patch("bot.munito_service", test_svc):
        # 1. /ai without args shows card
        with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
            await ai_command_handler(msg, CommandObject(prefix="/", command="ai", args=""))
            assert mock_reply.call_count == 1
            card_text = mock_reply.call_args[0][0]
            assert "Munito AI Assistant Settings" in card_text
            assert "🟢 Enabled" in card_text

        # 2. /ai off
        with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
            await ai_command_handler(msg, CommandObject(prefix="/", command="ai", args="off"))
            assert mock_reply.call_count == 1
            cfg = await test_svc.get_chat_config(chat_id)
            assert cfg.enabled is False

        # 3. /ai on
        with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
            await ai_command_handler(msg, CommandObject(prefix="/", command="ai", args="on"))
            assert mock_reply.call_count == 1
            cfg = await test_svc.get_chat_config(chat_id)
            assert cfg.enabled is True

        # 4. /ai model qwen3.8-27b
        with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
            await ai_command_handler(
                msg, CommandObject(prefix="/", command="ai", args="model qwen3.8-27b")
            )
            assert mock_reply.call_count == 1
            cfg = await test_svc.get_chat_config(chat_id)
            assert cfg.model == "qwen/qwen3.8-27b"

        # 5. /ai models
        with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
            await ai_command_handler(msg, CommandObject(prefix="/", command="ai", args="models"))
            assert mock_reply.call_count == 1
            models_text = mock_reply.call_args[0][0]
            assert "Available Munito AI models" in models_text
            assert "deepseek-ai/deepseek-v4.1-flash-1017" in models_text

        # 6. /ai effort
        with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
            await ai_command_handler(
                msg, CommandObject(prefix="/", command="ai", args="effort medium")
            )
            assert mock_reply.call_count == 1
            assert "Thinking:" in mock_reply.call_args[0][0]
            assert "Medium" in mock_reply.call_args[0][0]
            cfg = await test_svc.get_chat_config(chat_id)
            assert cfg.reasoning_effort == "medium"

        with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
            await ai_command_handler(
                msg, CommandObject(prefix="/", command="ai", args="effort invalid_val")
            )
            assert mock_reply.call_count == 1
            assert "Invalid thinking level" in mock_reply.call_args[0][0]

        # 7. /ai reset
        await test_svc.record_usage(chat_id, 10, 20, 30)
        with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
            await ai_command_handler(msg, CommandObject(prefix="/", command="ai", args="reset"))
            assert mock_reply.call_count == 1
            assert "Token usage statistics reset!" in mock_reply.call_args[0][0]
            stats = await test_svc.get_chat_stats(chat_id)
            assert stats.request_count == 0

        # 8. Callbacks
        cb_msg = Message(
            message_id=2,
            date=datetime.datetime.now(datetime.UTC),
            chat=chat,
            from_user=user,
            text="AI status card",
        ).as_(bot)

        # Toggle callback
        cb_toggle = CallbackQuery(
            id="cb_tog",
            from_user=user,
            chat_instance="ci1",
            data=f"ai:toggle:{chat_id}",
            message=cb_msg,
        ).as_(bot)

        with (
            patch.object(CallbackQuery, "answer", new_callable=AsyncMock) as mock_cb_answer,
            patch.object(Message, "edit_text", new_callable=AsyncMock) as mock_edit,
        ):
            await ai_callback_handler(cb_toggle)
            assert mock_cb_answer.call_count == 1
            assert mock_edit.call_count == 1
            cfg = await test_svc.get_chat_config(chat_id)
            assert cfg.enabled is False

        # Model callback
        cb_model = CallbackQuery(
            id="cb_mod",
            from_user=user,
            chat_instance="ci2",
            data=f"ai:model:{chat_id}",
            message=cb_msg,
        ).as_(bot)

        with (
            patch.object(CallbackQuery, "answer", new_callable=AsyncMock) as mock_cb_answer,
            patch.object(Message, "edit_text", new_callable=AsyncMock) as mock_edit,
        ):
            await ai_callback_handler(cb_model)
            assert mock_cb_answer.call_count == 1
            assert mock_edit.call_count == 1
            cfg = await test_svc.get_chat_config(chat_id)
            assert cfg.model != "qwen/qwen3.8-27b"

        # Reasoning effort callback
        cb_effort = CallbackQuery(
            id="cb_eff",
            from_user=user,
            chat_instance="ci3",
            data=f"ai:effort:{chat_id}",
            message=cb_msg,
        ).as_(bot)

        with (
            patch.object(CallbackQuery, "answer", new_callable=AsyncMock) as mock_cb_answer,
            patch.object(Message, "edit_text", new_callable=AsyncMock) as mock_edit,
        ):
            await ai_callback_handler(cb_effort)
            assert mock_cb_answer.call_count == 1
            assert mock_edit.call_count == 1
            cfg = await test_svc.get_chat_config(chat_id)
            assert cfg.reasoning_effort in ["low", "none", "high"]

    test_svc.close()


@pytest.mark.asyncio
async def test_munito_generate_reply_with_chat_id_and_token_tracking(tmp_path: Path):
    db_file = tmp_path / "test_track.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = 998877

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Відповідь з підрахунком токенів.",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }
    mock_resp.__aenter__.return_value = mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__.return_value = mock_session

    with patch("aiohttp.ClientSession", return_value=mock_session):
        reply = await svc.generate_reply("Привіт", chat_id=chat_id)
        assert reply == "Відповідь з підрахунком токенів."

        stats = await svc.get_chat_stats(chat_id)
        assert stats.request_count == 1
        assert stats.prompt_tokens == 12
        assert stats.completion_tokens == 8
        assert stats.total_tokens == 20

    # When disabled, should raise RuntimeError
    await svc.update_chat_config(chat_id, enabled=False)
    with pytest.raises(RuntimeError, match="AI assistant is disabled"):
        await svc.generate_reply("Привіт знову", chat_id=chat_id)

    svc.close()


def test_select_optimal_photo():
    from utils.telegram_media import select_optimal_photo

    p1 = MagicMock(width=90, height=90)
    p2 = MagicMock(width=320, height=240)
    p3 = MagicMock(width=800, height=600)
    p4 = MagicMock(width=4000, height=3000)

    # When 4 sizes exist, should pick p3 (<= 1280px, largest available)
    assert select_optimal_photo([p1, p2, p3, p4]) is p3

    # When only 2 sizes exist, picks the last one
    assert select_optimal_photo([p1, p2]) is p2

    # When all sizes are large (>1280px), picks second-to-largest
    big1 = MagicMock(width=2000, height=1500)
    big2 = MagicMock(width=4000, height=3000)
    big3 = MagicMock(width=6000, height=4500)
    assert select_optimal_photo([big1, big2, big3]) is big2


@pytest.mark.asyncio
async def test_munito_generate_reply_timeout():
    svc = MunitoAIService(api_key="test_key", db_path=":memory:")

    post_cm = MagicMock()
    post_cm.__aenter__.side_effect = TimeoutError()
    post_cm.__aexit__.return_value = None

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=post_cm)
    mock_session.__aenter__.return_value = mock_session

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(RuntimeError, match="timed out"):
            await svc.generate_reply("Test timeout")

    svc.close()


def test_context_cycling_helpers():
    # Context size cycling: 0 -> 5 -> 10 -> 15 -> 20 -> 0
    assert get_next_context_size(0) == 5
    assert get_next_context_size(5) == 10
    assert get_next_context_size(10) == 15
    assert get_next_context_size(15) == 20
    assert get_next_context_size(20) == 0
    assert get_next_context_size(99) == 10

    # Context mode cycling: mentions <-> all
    assert get_next_context_mode("mentions") == "all"
    assert get_next_context_mode("all") == "mentions"
    assert get_next_context_mode("MENTIONS") == "all"


@pytest.mark.asyncio
async def test_munito_history_crud_and_pruning(tmp_path: Path):
    db_file = tmp_path / "test_hist.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = 12345

    # Initially empty
    assert await svc.get_history_count(chat_id) == 0
    assert await svc.get_chat_history(chat_id) == []

    # Record turns
    await svc.record_turn(
        chat_id=chat_id,
        role="user",
        content="Hello bot",
        user_id=1,
        user_name="Alice",
    )
    await svc.record_turn(
        chat_id=chat_id,
        role="assistant",
        content="Hello Alice!",
        user_id=None,
        user_name="Bard-Bot",
    )

    assert await svc.get_history_count(chat_id) == 2
    history = await svc.get_chat_history(chat_id, limit=10)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].user_name == "Alice"
    assert history[0].content == "Hello bot"
    assert history[1].role == "assistant"
    assert history[1].content == "Hello Alice!"

    # Pruning test: insert 55 more turns (total 57) -> table should keep at most 50
    for i in range(55):
        await svc.record_turn(
            chat_id=chat_id,
            role="user",
            content=f"Message {i}",
            user_id=2,
            user_name="Bob",
        )

    count = await svc.get_history_count(chat_id)
    assert count == 50

    # Clear history
    deleted = await svc.clear_chat_history(chat_id)
    assert deleted == 50
    assert await svc.get_history_count(chat_id) == 0
    assert await svc.get_chat_history(chat_id) == []

    svc.close()


@pytest.mark.asyncio
async def test_munito_generate_reply_injects_and_records_context(tmp_path: Path):
    db_file = tmp_path / "test_gen_ctx.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = 777

    # Enable chat with context_size = 5
    await svc.update_chat_config(chat_id, enabled=True, context_size=5)

    # Pre-populate 2 historical turns
    await svc.record_turn(
        chat_id=chat_id,
        role="user",
        content="What is my name?",
        user_id=10,
        user_name="Charlie",
    )
    await svc.record_turn(
        chat_id=chat_id,
        role="assistant",
        content="Your name is Charlie.",
        user_id=None,
        user_name="Bard-Bot",
    )

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "You asked about capital cities.",
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp.__aenter__.return_value = mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__.return_value = mock_session

    with patch("aiohttp.ClientSession", return_value=mock_session):
        reply = await svc.generate_reply(
            prompt="And what did I ask before that?",
            chat_id=chat_id,
            user_id=10,
            user_name="Charlie",
        )
        assert reply == "You asked about capital cities."

        # Check payload messages sent to API
        call_args = mock_session.post.call_args
        _, kwargs = call_args
        payload = kwargs["json"]
        sent_messages = payload["messages"]

        # 1 system + 2 historical + 1 current user turn = 4
        assert len(sent_messages) == 4
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1]["role"] == "user"
        assert sent_messages[1]["content"] == "[Charlie]: What is my name?"
        assert sent_messages[2]["role"] == "assistant"
        assert sent_messages[2]["content"] == "Your name is Charlie."
        assert sent_messages[3]["role"] == "user"
        assert (
            sent_messages[3]["content"][0]["text"] == "[Charlie]: And what did I ask before that?"
        )

        # History should now have 4 turns (2 old + 1 new user + 1 new assistant)
        assert await svc.get_history_count(chat_id) == 4

    svc.close()


@pytest.mark.asyncio
async def test_ai_command_context_and_mode_and_forget(tmp_path: Path):
    from handlers.ai import ai_command_handler

    db_file = tmp_path / "test_cmd_ctx.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100555
    admin_id = 999

    fake_bot = AsyncMock()
    fake_bot.get_chat_member.return_value = MagicMock(status="administrator")

    chat = Chat(id=chat_id, type="supergroup", title="Test Tavern")
    user = User(id=admin_id, is_bot=False, first_name="Admin")
    msg = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="/ai",
    ).as_(fake_bot)

    with (
        patch("handlers.ai.bot.munito_service", svc),
        patch.object(Message, "reply", new_callable=AsyncMock),
    ):
        # 1. /ai context 15
        cmd_ctx = CommandObject(prefix="/", command="ai", args="context 15")
        await ai_command_handler(msg, cmd_ctx)
        cfg = await svc.get_chat_config(chat_id)
        assert cfg.context_size == 15

        # 2. /ai mode all
        cmd_mode = CommandObject(prefix="/", command="ai", args="mode all")
        await ai_command_handler(msg, cmd_mode)
        cfg = await svc.get_chat_config(chat_id)
        assert cfg.context_mode == "all"

        # 3. Add turns then /ai forget
        await svc.record_turn(chat_id, role="user", content="Test message")
        assert await svc.get_history_count(chat_id) == 1

        cmd_forget = CommandObject(prefix="/", command="ai", args="forget")
        await ai_command_handler(msg, cmd_forget)
        assert await svc.get_history_count(chat_id) == 0

    svc.close()


@pytest.mark.asyncio
async def test_ai_callback_context_and_mode(tmp_path: Path):
    from handlers.ai import ai_callback_handler

    db_file = tmp_path / "test_cb_ctx.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100888
    admin_id = 999

    fake_bot = AsyncMock()
    fake_bot.get_chat_member.return_value = MagicMock(status="administrator")

    chat = Chat(id=chat_id, type="supergroup", title="Test Tavern")
    user = User(id=admin_id, is_bot=False, first_name="Admin")

    target_msg = Message(
        message_id=2,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="Card",
    ).as_(fake_bot)

    with (
        patch("handlers.ai.bot.munito_service", svc),
        patch.object(CallbackQuery, "answer", new_callable=AsyncMock),
        patch.object(Message, "edit_text", new_callable=AsyncMock),
    ):
        # 1. Cycle context size: default 10 -> 15
        cb_size = CallbackQuery(
            id="cb_size",
            from_user=user,
            chat_instance="ci1",
            data=f"ai:ctx_size:{chat_id}",
            message=target_msg,
        ).as_(fake_bot)

        await ai_callback_handler(cb_size)
        cfg = await svc.get_chat_config(chat_id)
        assert cfg.context_size == 15

        # 2. Toggle context mode: default "mentions" -> "all"
        cb_mode = CallbackQuery(
            id="cb_mode",
            from_user=user,
            chat_instance="ci2",
            data=f"ai:ctx_mode:{chat_id}",
            message=target_msg,
        ).as_(fake_bot)

        await ai_callback_handler(cb_mode)
        cfg = await svc.get_chat_config(chat_id)
        assert cfg.context_mode == "all"

        # 3. Clear memory callback
        await svc.record_turn(chat_id, role="user", content="Ping")
        assert await svc.get_history_count(chat_id) == 1

        cb_clear = CallbackQuery(
            id="cb_clear",
            from_user=user,
            chat_instance="ci3",
            data=f"ai:clear_mem:{chat_id}",
            message=target_msg,
        ).as_(fake_bot)

        await ai_callback_handler(cb_clear)
        assert await svc.get_history_count(chat_id) == 0

    svc.close()
