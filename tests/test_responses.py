from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Chat, Message, User

from utils.responses import (
    safe_reply,
    safe_reply_audio,
    safe_reply_media_group,
    safe_reply_photo,
    safe_reply_video,
)


def create_mock_message() -> Message:
    user = User(id=123, is_bot=False, first_name="TestUser")
    chat = Chat(id=456, type="private")
    return Message(message_id=789, date=0, chat=chat, from_user=user)


@pytest.mark.asyncio
async def test_safe_reply_success():
    msg = create_mock_message()
    with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
        mock_reply.return_value = msg
        result = await safe_reply(msg, "Hello World")
        assert result is msg
        mock_reply.assert_awaited_once_with("Hello World")


@pytest.mark.asyncio
async def test_safe_reply_fallback_on_deleted_message():
    msg = create_mock_message()
    bad_req = TelegramBadRequest(
        method="sendMessage", message="Bad Request: message to be replied not found"
    )

    with (
        patch.object(Message, "reply", new_callable=AsyncMock, side_effect=bad_req),
        patch.object(Message, "answer", new_callable=AsyncMock) as mock_answer,
    ):
        mock_answer.return_value = msg
        result = await safe_reply(msg, "Hello Fallback")
        assert result is msg
        mock_answer.assert_awaited_once_with("Hello Fallback")


@pytest.mark.asyncio
async def test_safe_reply_video_fallback():
    msg = create_mock_message()
    bad_req = TelegramBadRequest(
        method="sendVideo", message="Bad Request: message to be replied not found"
    )

    with (
        patch.object(Message, "reply_video", new_callable=AsyncMock, side_effect=bad_req),
        patch.object(Message, "answer_video", new_callable=AsyncMock) as mock_answer,
    ):
        mock_answer.return_value = msg
        result = await safe_reply_video(msg, video="video_file_id", caption="Test")
        assert result is msg
        mock_answer.assert_awaited_once_with(video="video_file_id", caption="Test")


@pytest.mark.asyncio
async def test_safe_reply_photo_fallback():
    msg = create_mock_message()
    bad_req = TelegramBadRequest(
        method="sendPhoto", message="Bad Request: message to be replied not found"
    )

    with (
        patch.object(Message, "reply_photo", new_callable=AsyncMock, side_effect=bad_req),
        patch.object(Message, "answer_photo", new_callable=AsyncMock) as mock_answer,
    ):
        mock_answer.return_value = msg
        result = await safe_reply_photo(msg, photo="photo_file_id", caption="Test")
        assert result is msg
        mock_answer.assert_awaited_once_with(photo="photo_file_id", caption="Test")


@pytest.mark.asyncio
async def test_safe_reply_audio_fallback():
    msg = create_mock_message()
    bad_req = TelegramBadRequest(
        method="sendAudio", message="Bad Request: message to be replied not found"
    )

    with (
        patch.object(Message, "reply_audio", new_callable=AsyncMock, side_effect=bad_req),
        patch.object(Message, "answer_audio", new_callable=AsyncMock) as mock_answer,
    ):
        mock_answer.return_value = msg
        result = await safe_reply_audio(msg, audio="audio_file_id", title="Title")
        assert result is msg
        mock_answer.assert_awaited_once_with(audio="audio_file_id", title="Title")


@pytest.mark.asyncio
async def test_safe_reply_media_group_fallback():
    msg = create_mock_message()
    bad_req = TelegramBadRequest(
        method="sendMediaGroup", message="Bad Request: message to be replied not found"
    )

    with (
        patch.object(Message, "reply_media_group", new_callable=AsyncMock, side_effect=bad_req),
        patch.object(Message, "answer_media_group", new_callable=AsyncMock) as mock_answer,
    ):
        mock_answer.return_value = [msg]
        result = await safe_reply_media_group(msg, media=[])
        assert result == [msg]
        mock_answer.assert_awaited_once_with(media=[])


@pytest.mark.asyncio
async def test_safe_reply_reraises_other_bad_request():
    msg = create_mock_message()
    bad_req = TelegramBadRequest(method="sendMessage", message="Bad Request: chat not found")

    with patch.object(Message, "reply", new_callable=AsyncMock, side_effect=bad_req):
        with pytest.raises(TelegramBadRequest) as exc_info:
            await safe_reply(msg, "Fail")
        assert "chat not found" in str(exc_info.value)
