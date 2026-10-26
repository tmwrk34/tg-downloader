from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import Message, User

from services.munito import MunitoAIService, split_telegram_message


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
        (401, "Невірний або недійсний ключ"),
        (429, "Перевищено ліміт"),
        (503, "перевантажена"),
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
        assert msg.reply.call_args[0][0] == "Here is the explanation."
