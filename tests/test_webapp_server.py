import datetime
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.types import Chat, Message, User

from handlers.games import games_hub_handler, poker_table_handler
from services.games.auth import validate_telegram_init_data
from services.games.server import GamesWebServer


def test_auth_dev_mode(monkeypatch):
    monkeypatch.setenv("WEBAPP_DEV_MODE", "1")
    user = validate_telegram_init_data("", "fake_token")
    assert user is not None
    assert user.user_id == 999999


@pytest.mark.asyncio
async def test_games_server_health():
    server = GamesWebServer(bot_token="test_token", host="127.0.0.1", port=8089)
    await server.start()

    async with aiohttp_client_session() as session:
        async with session.get("http://127.0.0.1:8089/api/health") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"

    await server.stop()


def aiohttp_client_session():
    import aiohttp

    return aiohttp.ClientSession()


@pytest.mark.asyncio
async def test_games_and_poker_handlers():
    chat = Chat(id=-100777, type="supergroup", title="Test Tavern")
    user = User(id=123, is_bot=False, first_name="Player1")
    fake_bot = AsyncMock()

    msg = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="/games",
    ).as_(fake_bot)

    with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
        # 1. /games
        await games_hub_handler(msg, fake_bot)
        mock_reply.assert_called_once()
        assert "Ігрова Зала Таверни" in mock_reply.call_args[0][0]

        # 2. /poker
        mock_reply.reset_mock()
        await poker_table_handler(msg, fake_bot)
        mock_reply.assert_called_once()
        assert "Стіл Техаського Холдему" in mock_reply.call_args[0][0]
