from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.types import Chat, Message, User

from handlers.quote import quote_command_handler


def create_user(user_id: int = 12345, name: str = "Bard") -> User:
    return User(id=user_id, is_bot=False, first_name=name)


def create_message(
    text: str | None = None,
    reply_to: Message | None = None,
    user: User | None = None,
) -> Message:
    return Message(
        message_id=100,
        date=datetime.now(UTC),
        chat=Chat(id=-100123456789, type="supergroup"),
        from_user=user or create_user(),
        text=text,
        reply_to_message=reply_to,
    )


@pytest.mark.asyncio
async def test_quote_command_no_args_no_reply():
    bot = AsyncMock()
    msg = create_message(text="/quote")

    with patch("handlers.quote.safe_reply", new_callable=AsyncMock) as mock_safe_reply:
        await quote_command_handler(msg, bot)
        mock_safe_reply.assert_awaited_once()
        args, kwargs = mock_safe_reply.call_args
        assert "Як створити сувій цитати" in args[1]


@pytest.mark.asyncio
async def test_quote_command_with_direct_text():
    bot = AsyncMock()
    bot.send_photo = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.get_user_profile_photos = AsyncMock(return_value=None)

    msg = create_message(text="/quote Прекрасний вечір у таверні!")

    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 1000

    with patch(
        "handlers.quote.quote_service.async_render_quote", new_callable=AsyncMock
    ) as mock_render:
        mock_render.return_value = fake_jpeg
        await quote_command_handler(msg, bot)

        mock_render.assert_awaited_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["text"] == "Прекрасний вечір у таверні!"
        assert kwargs["author_name"] == "Bard"
        bot.send_photo.assert_awaited_once()


@pytest.mark.asyncio
async def test_quote_command_reply_to_message():
    bot = AsyncMock()
    bot.send_photo = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.get_user_profile_photos = AsyncMock(return_value=None)

    replied_user = create_user(user_id=777, name="Geralt of Rivia")
    replied_msg = create_message(text="Зло — це зло, Стрегобор.", user=replied_user)
    cmd_msg = create_message(text="/q", reply_to=replied_msg)

    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 1000

    with patch(
        "handlers.quote.quote_service.async_render_quote", new_callable=AsyncMock
    ) as mock_render:
        mock_render.return_value = fake_jpeg
        await quote_command_handler(cmd_msg, bot)

        mock_render.assert_awaited_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["text"] == "Зло — це зло, Стрегобор."
        assert kwargs["author_name"] == "Geralt of Rivia"
        bot.send_photo.assert_awaited_once()
        photo_kwargs = bot.send_photo.call_args.kwargs
        assert photo_kwargs["reply_to_message_id"] == replied_msg.message_id
