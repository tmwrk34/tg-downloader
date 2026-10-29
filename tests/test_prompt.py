import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.filters import CommandObject
from aiogram.types import BufferedInputFile, Chat, Document, Message, User

from handlers.ai import prompt_command_handler
from services.munito import MunitoAIService
from services.prompt import (
    DEFAULT_BASE_PROMPT_PATH,
    DEFAULT_SYSTEM_PROMPT,
    FALLBACK_SYSTEM_PROMPT,
    is_custom_prompt_active,
    load_system_prompt,
    reset_system_prompt,
    resolve_custom_prompt_path,
    save_system_prompt,
)


def test_factory_system_prompt_file_exists():
    """Verify prompts/system_prompt.txt exists and contains expected persona keywords."""
    assert DEFAULT_BASE_PROMPT_PATH.is_file()
    content = DEFAULT_BASE_PROMPT_PATH.read_text(encoding="utf-8")
    assert "You are Bard" in content
    assert "80/20 Rule" in content
    assert "roll_dice" in content
    assert DEFAULT_SYSTEM_PROMPT == content.strip()


def test_load_system_prompt_fallback(tmp_path: Path):
    """When custom and base paths do not exist, fall back to fallback string."""
    empty_custom = tmp_path / "non_existent_custom.txt"
    empty_base = tmp_path / "non_existent_base.txt"
    loaded = load_system_prompt(custom_path=empty_custom, base_path=empty_base)
    assert loaded == FALLBACK_SYSTEM_PROMPT


def test_save_load_reset_lifecycle(tmp_path: Path):
    """Verify save creates custom prompt, load prioritizes it, and reset reverts it."""
    custom_file = tmp_path / "custom_prompt.txt"
    base_file = tmp_path / "base_prompt.txt"
    base_file.write_text("Base Tavern Bard Prompt\n", encoding="utf-8")

    assert not is_custom_prompt_active(custom_path=custom_file)
    assert (
        load_system_prompt(custom_path=custom_file, base_path=base_file)
        == "Base Tavern Bard Prompt"
    )

    # Save new prompt
    saved_path = save_system_prompt("Updated Sci-Fi Bard Persona", custom_path=custom_file)
    assert saved_path == custom_file
    assert is_custom_prompt_active(custom_path=custom_file)
    assert (
        load_system_prompt(custom_path=custom_file, base_path=base_file)
        == "Updated Sci-Fi Bard Persona"
    )

    # Persistence verification: reload from file
    assert custom_file.read_text(encoding="utf-8").strip() == "Updated Sci-Fi Bard Persona"

    # Reset
    reloaded = reset_system_prompt(custom_path=custom_file, base_path=base_file)
    assert reloaded == "Base Tavern Bard Prompt"
    assert not custom_file.exists()
    assert not is_custom_prompt_active(custom_path=custom_file)


def test_save_empty_prompt_raises(tmp_path: Path):
    """Saving an empty or whitespace-only prompt should raise ValueError."""
    custom_file = tmp_path / "custom.txt"
    with pytest.raises(ValueError, match="cannot be empty"):
        save_system_prompt("   \n\t  ", custom_path=custom_file)


def test_resolve_custom_prompt_path(tmp_path: Path, monkeypatch):
    """Test resolution via environment variable and writable data directory."""
    env_file = tmp_path / "env_prompt.txt"
    monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(env_file))
    assert resolve_custom_prompt_path() == env_file


def test_munito_service_prompt_methods(tmp_path: Path):
    """Test MunitoAIService integration with custom persistent prompt methods."""
    db_file = tmp_path / "test_ai.db"
    custom_file = tmp_path / "custom_data_prompt.txt"

    with patch("services.prompt.resolve_custom_prompt_path", return_value=custom_file):
        svc = MunitoAIService(api_key="test_key", db_path=db_file)

        initial_prompt = svc.get_system_prompt()
        assert "Bard" in initial_prompt
        assert svc.is_custom_prompt() is False

        # Set persistent prompt
        saved_path = svc.set_system_prompt("New persistent tavern keeper persona")
        assert saved_path == custom_file
        assert svc.get_system_prompt() == "New persistent tavern keeper persona"
        assert svc.is_custom_prompt() is True

        # Simulate service restart / redeploy
        svc2 = MunitoAIService(api_key="test_key", db_path=db_file)
        assert svc2.get_system_prompt() == "New persistent tavern keeper persona"
        assert svc2.is_custom_prompt() is True

        # Reset prompt
        reloaded = svc.reset_system_prompt()
        assert "You are Bard" in reloaded
        assert svc.get_system_prompt() == reloaded
        assert svc.is_custom_prompt() is False

        svc.close()
        svc2.close()


@pytest.mark.asyncio
async def test_prompt_command_get_and_help(tmp_path: Path):
    """Test /prompt view and /prompt help commands."""
    db_file = tmp_path / "test_get.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100111
    user_id = 456

    fake_bot = AsyncMock()
    chat = Chat(id=chat_id, type="supergroup", title="Test Tavern")
    user = User(id=user_id, is_bot=False, first_name="TavernUser")
    msg = Message(
        message_id=10,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="/prompt",
    ).as_(fake_bot)

    with (
        patch("handlers.ai.bot.munito_service", svc),
        patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply,
    ):
        # 1. /prompt (default get)
        cmd_get = CommandObject(prefix="/", command="prompt", args="")
        await prompt_command_handler(msg, cmd_get)
        mock_reply.assert_called_once()
        assert "Active System Prompt" in mock_reply.call_args[0][0]
        assert "Factory Default" in mock_reply.call_args[0][0]

        mock_reply.reset_mock()
        # 2. /prompt help
        cmd_help = CommandObject(prefix="/", command="prompt", args="help")
        await prompt_command_handler(msg, cmd_help)
        mock_reply.assert_called_once()
        assert "System Prompt Management" in mock_reply.call_args[0][0]

    svc.close()


@pytest.mark.asyncio
async def test_prompt_command_file_download(tmp_path: Path):
    """Test /prompt file downloads prompt as a .txt document."""
    db_file = tmp_path / "test_file.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100111
    user_id = 456

    fake_bot = AsyncMock()
    chat = Chat(id=chat_id, type="supergroup", title="Test Tavern")
    user = User(id=user_id, is_bot=False, first_name="TavernUser")
    msg = Message(
        message_id=11,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="/prompt file",
    ).as_(fake_bot)

    with (
        patch("handlers.ai.bot.munito_service", svc),
        patch.object(Message, "reply_document", new_callable=AsyncMock) as mock_reply_doc,
    ):
        cmd_file = CommandObject(prefix="/", command="prompt", args="file")
        await prompt_command_handler(msg, cmd_file)
        mock_reply_doc.assert_called_once()
        doc_arg = mock_reply_doc.call_args[0][0]
        assert isinstance(doc_arg, BufferedInputFile)
        assert doc_arg.filename == "system_prompt.txt"
        assert b"You are Bard" in doc_arg.data

    svc.close()


@pytest.mark.asyncio
async def test_prompt_command_admin_set_and_reset(tmp_path: Path):
    """Test /prompt set and /prompt reset with admin authorization."""
    db_file = tmp_path / "test_set.db"
    custom_file = tmp_path / "custom_prompt.txt"

    with patch("services.prompt.resolve_custom_prompt_path", return_value=custom_file):
        svc = MunitoAIService(api_key="test_key", db_path=db_file)
        chat_id = -100111
        user_id = 456

        fake_bot = AsyncMock()
        # Non-admin initially
        fake_bot.get_chat_member.return_value = MagicMock(status="member")

        chat = Chat(id=chat_id, type="supergroup", title="Test Tavern")
        user = User(id=user_id, is_bot=False, first_name="TavernUser")
        msg = Message(
            message_id=12,
            date=datetime.datetime.now(datetime.UTC),
            chat=chat,
            from_user=user,
            text="/prompt set New custom prompt text",
        ).as_(fake_bot)

        with (
            patch("handlers.ai.bot.munito_service", svc),
            patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply,
        ):
            # 1. Non-admin trying /prompt set -> Access Denied
            cmd_set = CommandObject(prefix="/", command="prompt", args="set New custom prompt text")
            await prompt_command_handler(msg, cmd_set)
            assert "Access Denied" in mock_reply.call_args[0][0]
            assert svc.is_custom_prompt() is False

            # 2. Promote to admin and try /prompt set again
            fake_bot.get_chat_member.return_value = MagicMock(status="administrator")
            mock_reply.reset_mock()
            await prompt_command_handler(msg, cmd_set)
            assert "System Prompt Updated" in mock_reply.call_args[0][0]
            assert svc.get_system_prompt() == "New custom prompt text"
            assert svc.is_custom_prompt() is True

            # 3. Non-admin trying /prompt reset -> Access Denied
            fake_bot.get_chat_member.return_value = MagicMock(status="member")
            mock_reply.reset_mock()
            cmd_reset = CommandObject(prefix="/", command="prompt", args="reset")
            await prompt_command_handler(msg, cmd_reset)
            assert "Access Denied" in mock_reply.call_args[0][0]
            assert svc.is_custom_prompt() is True

            # 4. Admin /prompt reset -> Revert to factory default
            fake_bot.get_chat_member.return_value = MagicMock(status="administrator")
            mock_reply.reset_mock()
            await prompt_command_handler(msg, cmd_reset)
            assert "System Prompt Reset" in mock_reply.call_args[0][0]
            assert "You are Bard" in svc.get_system_prompt()
            assert svc.is_custom_prompt() is False

        svc.close()


@pytest.mark.asyncio
async def test_prompt_command_reply_to_message_and_doc(tmp_path: Path):
    """Test updating prompt via /prompt set in reply to a message or .txt document."""
    db_file = tmp_path / "test_reply.db"
    custom_file = tmp_path / "custom_prompt_reply.txt"

    with patch("services.prompt.resolve_custom_prompt_path", return_value=custom_file):
        svc = MunitoAIService(api_key="test_key", db_path=db_file)
        chat_id = -100111
        admin_id = 999

        fake_bot = AsyncMock()
        fake_bot.get_chat_member.return_value = MagicMock(status="creator")

        chat = Chat(id=chat_id, type="supergroup", title="Test Tavern")
        user = User(id=admin_id, is_bot=False, first_name="Boss")

        # 1. Reply to text message
        replied_text_msg = Message(
            message_id=20,
            date=datetime.datetime.now(datetime.UTC),
            chat=chat,
            from_user=user,
            text="You are an ancient cyber-bard with a robotic lute.",
        ).as_(fake_bot)

        cmd_msg = Message(
            message_id=21,
            date=datetime.datetime.now(datetime.UTC),
            chat=chat,
            from_user=user,
            text="/prompt set",
            reply_to_message=replied_text_msg,
        ).as_(fake_bot)

        with (
            patch("handlers.ai.bot.munito_service", svc),
            patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply,
        ):
            cmd_set = CommandObject(prefix="/", command="prompt", args="set")
            await prompt_command_handler(cmd_msg, cmd_set)
            assert "System Prompt Updated" in mock_reply.call_args[0][0]
            assert svc.get_system_prompt() == "You are an ancient cyber-bard with a robotic lute."

        # 2. Reply to document .txt message
        doc = Document(
            file_id="doc_123",
            file_unique_id="uniq_123",
            file_name="custom_prompt.txt",
            mime_type="text/plain",
            file_size=42,
        )
        replied_doc_msg = Message(
            message_id=22,
            date=datetime.datetime.now(datetime.UTC),
            chat=chat,
            from_user=user,
            document=doc,
        ).as_(fake_bot)

        cmd_doc_msg = Message(
            message_id=23,
            date=datetime.datetime.now(datetime.UTC),
            chat=chat,
            from_user=user,
            text="/prompt set",
            reply_to_message=replied_doc_msg,
        ).as_(fake_bot)

        with (
            patch("handlers.ai.bot.munito_service", svc),
            patch(
                "handlers.ai.download_telegram_file_id",
                return_value=b"Prompt loaded from document file!",
            ),
            patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply,
        ):
            await prompt_command_handler(cmd_doc_msg, cmd_set)
            assert "System Prompt Updated" in mock_reply.call_args[0][0]
            assert svc.get_system_prompt() == "Prompt loaded from document file!"

        svc.close()
