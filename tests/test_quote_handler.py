from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import Chat, Message, PhotoSize, User

from handlers.quote import _download_avatar_bytes, quote_command_handler
from utils.telegram_media import download_telegram_file_id


def create_user(user_id: int = 12345, name: str = "Bard") -> User:
    return User(id=user_id, is_bot=False, first_name=name)


def create_message(
    text: str | None = None,
    reply_to: Message | None = None,
    user: User | None = None,
    forward_from: User | None = None,
    forward_sender_name: str | None = None,
) -> Message:
    return Message(
        message_id=100,
        date=datetime.now(UTC),
        chat=Chat(id=-100123456789, type="supergroup"),
        from_user=user or create_user(),
        text=text,
        reply_to_message=reply_to,
        forward_from=forward_from,
        forward_sender_name=forward_sender_name,
    )


def create_mock_bot() -> AsyncMock:
    bot = AsyncMock()
    bot.send_photo = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.get_user_profile_photos = AsyncMock(return_value=None)
    bot.get_chat = AsyncMock(return_value=None)
    return bot


@pytest.mark.asyncio
async def test_quote_command_no_args_no_reply():
    bot = create_mock_bot()
    msg = create_message(text="/quote")

    with patch("handlers.quote.safe_reply", new_callable=AsyncMock) as mock_safe_reply:
        await quote_command_handler(msg, bot)
        mock_safe_reply.assert_awaited_once()
        args, _kwargs = mock_safe_reply.call_args
        assert "Як створити сувій цитати" in args[1]


@pytest.mark.asyncio
async def test_quote_command_with_direct_text():
    bot = create_mock_bot()
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
    bot = create_mock_bot()
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


@pytest.mark.asyncio
async def test_quote_command_forwarded_message_and_render_error():
    bot = create_mock_bot()
    orig_author = create_user(user_id=555, name="Dandelion")
    replied_msg = create_message(
        text="Киньте гріш Відьмакові!",
        forward_from=orig_author,
    )
    cmd_msg = create_message(text="/quote", reply_to=replied_msg)

    with (
        patch(
            "handlers.quote.quote_service.async_render_quote",
            new_callable=AsyncMock,
            side_effect=RuntimeError("render boom"),
        ),
        patch("handlers.quote.safe_reply", new_callable=AsyncMock) as mock_safe_reply,
    ):
        await quote_command_handler(cmd_msg, bot)
        mock_safe_reply.assert_awaited_once()
        assert "Не вдалося закарбувати" in mock_safe_reply.call_args.args[1]


@pytest.mark.asyncio
async def test_download_avatar_bytes_via_get_chat_fallback():
    bot = create_mock_bot()
    chat_obj = MagicMock()
    chat_obj.photo.big_file_id = "big_avatar_fid_999"
    bot.get_chat = AsyncMock(return_value=chat_obj)

    with patch(
        "handlers.quote.download_telegram_file_id",
        new_callable=AsyncMock,
        return_value=b"avatar_raw_bytes",
    ) as mock_dl:
        res = await _download_avatar_bytes(bot, user_id=42)
        assert res == b"avatar_raw_bytes"
        mock_dl.assert_awaited_once_with(bot, "big_avatar_fid_999")


@pytest.mark.asyncio
async def test_download_telegram_file_id_local_bot_api_http_fallback(tmp_path: Path):
    active_bot = AsyncMock()
    active_bot.token = "123456:ABCDEF"
    file_info = MagicMock()
    file_info.file_path = "/var/lib/telegram-bot-api/123456:ABCDEF/photos/file_42.jpg"
    active_bot.get_file = AsyncMock(return_value=file_info)
    active_bot.download = AsyncMock(side_effect=RuntimeError("no direct download"))

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.read = AsyncMock(return_value=b"http_local_api_photo_bytes")
    mock_resp.__aenter__.return_value = mock_resp

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__.return_value = mock_session

    fake_cfg = MagicMock(local_bot_api_url="http://telegram-bot-api:8081")
    fake_bot_mod = MagicMock(config=fake_cfg)

    with (
        patch.dict("sys.modules", {"bot": fake_bot_mod}),
        patch("aiohttp.ClientSession", return_value=mock_session),
    ):
        data = await download_telegram_file_id(active_bot, "fid_42")
        assert data == b"http_local_api_photo_bytes"
        called_url = mock_session.get.call_args.args[0]
        assert called_url == "http://telegram-bot-api:8081/file/bot123456:ABCDEF/photos/file_42.jpg"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cmd_text", "expected_seal", "expected_dropcap"),
    [
        ("/qs Звичайна цитата", False, True),
        ("/qf Звичайна цитата", True, False),
        ("/qsf Звичайна цитата", False, False),
        ("/qfs Звичайна цитата", False, False),
        ("/q -s Звичайна цитата", False, True),
        ("/q -f Звичайна цитата", True, False),
        ("/q -sf Звичайна цитата", False, False),
    ],
)
async def test_quote_command_modifiers(cmd_text: str, expected_seal: bool, expected_dropcap: bool):
    bot = create_mock_bot()
    msg = create_message(text=cmd_text)
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 1000

    with patch(
        "handlers.quote.quote_service.async_render_quote", new_callable=AsyncMock
    ) as mock_render:
        mock_render.return_value = fake_jpeg
        await quote_command_handler(msg, bot)

        mock_render.assert_awaited_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["text"] == "Звичайна цитата"
        assert kwargs["show_seal"] == expected_seal
        assert kwargs["show_dropcap"] == expected_dropcap
        bot.send_photo.assert_awaited_once()


@pytest.mark.asyncio
async def test_quote_reply_to_photo_with_custom_text():
    bot = create_mock_bot()
    photo_sender = create_user(user_id=888, name="Painter")
    replied_photo_msg = Message(
        message_id=200,
        date=datetime.now(UTC),
        chat=Chat(id=-100123456789, type="supergroup"),
        from_user=photo_sender,
        photo=[
            PhotoSize(file_id="photo_small", file_unique_id="uniq_1", width=100, height=100),
            PhotoSize(file_id="photo_large", file_unique_id="uniq_2", width=800, height=800),
        ],
    )
    cmd_user = create_user(user_id=999, name="LetterWriter")
    cmd_msg = create_message(
        text="/q Шановний друже, надсилаю цей сувій.",
        user=cmd_user,
        reply_to=replied_photo_msg,
    )
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 1000

    with (
        patch(
            "handlers.quote.download_telegram_file_id",
            new_callable=AsyncMock,
            return_value=b"seal_image_bytes",
        ) as mock_dl,
        patch(
            "handlers.quote.quote_service.async_render_quote", new_callable=AsyncMock
        ) as mock_render,
    ):
        mock_render.return_value = fake_jpeg
        await quote_command_handler(cmd_msg, bot)

        # Photo from replied message was downloaded for the seal
        mock_dl.assert_awaited_once_with(bot, "photo_large")
        mock_render.assert_awaited_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["text"] == "Шановний друже, надсилаю цей сувій."
        assert kwargs["author_name"] == "LetterWriter"
        assert kwargs["avatar_bytes"] == b"seal_image_bytes"
        bot.send_photo.assert_awaited_once()


@pytest.mark.asyncio
async def test_quote_reply_to_photo_no_text_shows_hint():
    bot = create_mock_bot()
    replied_photo_msg = Message(
        message_id=200,
        date=datetime.now(UTC),
        chat=Chat(id=-100123456789, type="supergroup"),
        from_user=create_user(),
        photo=[PhotoSize(file_id="photo_large", file_unique_id="uniq_3", width=800, height=800)],
    )
    cmd_msg = create_message(text="/q", reply_to=replied_photo_msg)

    with patch("handlers.quote.safe_reply", new_callable=AsyncMock) as mock_safe_reply:
        await quote_command_handler(cmd_msg, bot)
        mock_safe_reply.assert_awaited_once()
        hint_text = mock_safe_reply.call_args.args[1]
        assert "Лист із зображенням на печатці" in hint_text


@pytest.mark.asyncio
async def test_quote_explicit_author_parsing():
    bot = create_mock_bot()
    msg = create_message(text="/q Борітеся — поборете! — Тарас Шевченко")
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 1000

    with patch(
        "handlers.quote.quote_service.async_render_quote", new_callable=AsyncMock
    ) as mock_render:
        mock_render.return_value = fake_jpeg
        await quote_command_handler(msg, bot)

        mock_render.assert_awaited_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["text"] == "Борітеся — поборете!"
        assert kwargs["author_name"] == "Тарас Шевченко"
