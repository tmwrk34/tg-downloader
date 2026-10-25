import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.filters import CommandObject
from aiogram.types import CallbackQuery, Chat, Message, User

from services.chatter import (
    CHATTER_REACTIONS,
    ChatterService,
    clean_and_tokenize,
)


def test_clean_and_tokenize():
    # Commands must return empty list
    assert clean_and_tokenize("/roll 2d20") == []
    assert clean_and_tokenize("/chatter on") == []

    # Messages containing URLs must be ignored from training
    assert clean_and_tokenize("Check this out https://youtube.com/watch?v=123") == []
    assert clean_and_tokenize("https://vt.tiktok.com/ZS12345/") == []

    # Ukrainian text with punctuation
    tokens = clean_and_tokenize("Привіт, лицарю! Як твої справи в таверні?")
    assert len(tokens) == 2
    assert tokens[0] == ["Привіт", "лицарю"]
    assert tokens[1] == ["Як", "твої", "справи", "в", "таверні"]

    # Mentions are stripped
    tokens2 = clean_and_tokenize("@bard_taverna_bot привіт друже!")
    assert tokens2 == [["привіт", "друже"]]

    # Single-word sentences filtered out
    assert clean_and_tokenize("Привіт! Ок. Добре.") == []


@pytest.mark.asyncio
async def test_chatter_service_config_defaults_and_updates(tmp_path: Path):
    db_path = tmp_path / "chatter_test.db"
    service = ChatterService(db_path=db_path)

    # 1. Defaults
    cfg1 = await service.get_config(-1001)
    assert cfg1.chat_id == -1001
    assert cfg1.enabled is False
    assert cfg1.reply_chance == 0.05
    assert cfg1.reaction_chance == 0.10
    assert cfg1.reply_to_mention is True
    assert cfg1.cooldown_seconds == 30
    assert cfg1.learning_enabled is True

    # 2. Update settings
    updated = await service.update_config(
        -1001,
        enabled=True,
        reply_chance=0.25,
        reaction_chance=0.50,
        cooldown_seconds=60,
    )
    assert updated.enabled is True
    assert updated.reply_chance == 0.25
    assert updated.reaction_chance == 0.50
    assert updated.cooldown_seconds == 60

    # 3. Isolation: another chat still has defaults
    cfg2 = await service.get_config(-1002)
    assert cfg2.chat_id == -1002
    assert cfg2.enabled is False
    assert cfg2.reply_chance == 0.05

    # 4. Persistence across service re-instantiation
    service.close()
    service_reopened = ChatterService(db_path=db_path)
    reloaded = await service_reopened.get_config(-1001)
    assert reloaded.enabled is True
    assert reloaded.reply_chance == 0.25
    assert reloaded.cooldown_seconds == 60
    service_reopened.close()


@pytest.mark.asyncio
async def test_chatter_train_and_generate(tmp_path: Path):
    service = ChatterService(db_path=tmp_path / "chatter_train.db")

    # Empty vocabulary returns None
    assert await service.generate(-1001) is None

    # Train on sentences
    text = (
        "У таверні грає бард на лютні. "
        "Бард на лютні співає пісні про дракона. "
        "Дракон спить у темній печері біля гори. "
        "Лицарі замовляють ель у таверні."
    )
    added = await service.train(-1001, text)
    assert added > 0

    stats = await service.get_stats(-1001)
    assert stats.total_transitions > 0
    assert stats.unique_words > 5

    # Generate general sentence
    generated = await service.generate(-1001)
    assert generated is not None
    assert len(generated.split()) >= 2
    assert generated[0].isupper()
    assert generated[-1] in ".!?"

    # Generate with seed text
    seeded = await service.generate(-1001, seed_text="бард печера")
    assert seeded is not None
    assert len(seeded.split()) >= 2

    # Clear vocabulary
    cleared = await service.clear_vocabulary(-1001)
    assert cleared > 0
    stats_after = await service.get_stats(-1001)
    assert stats_after.total_transitions == 0
    assert await service.generate(-1001) is None

    service.close()


def test_chatter_should_reply_and_react():
    service = ChatterService(db_path=":memory:")
    cfg = service._sync_get_config(12345)

    # 1. When disabled, never replies or reacts
    assert cfg.enabled is False
    assert service.should_reply(cfg, is_mention_or_reply=False) is False
    assert service.should_reply(cfg, is_mention_or_reply=True) is False
    assert service.should_react(cfg) is False

    # 2. When enabled, mentions always reply
    cfg.enabled = True
    cfg.reply_to_mention = True
    assert service.should_reply(cfg, is_mention_or_reply=True) is True

    # 3. Cooldown check
    now = 1000.0
    cfg.last_reply_timestamp = 990.0
    cfg.cooldown_seconds = 30
    assert service.should_reply(cfg, is_mention_or_reply=False, current_time=now) is False

    # After cooldown elapsed
    cfg.last_reply_timestamp = 950.0  # 50s ago
    with patch("random.random", return_value=0.01):
        assert service.should_reply(cfg, is_mention_or_reply=False, current_time=now) is True

    with patch("random.random", return_value=0.99):
        assert service.should_reply(cfg, is_mention_or_reply=False, current_time=now) is False

    # 4. Reaction check
    cfg.reaction_chance = 0.20
    with patch("random.random", return_value=0.15):
        assert service.should_react(cfg) is True
    with patch("random.random", return_value=0.25):
        assert service.should_react(cfg) is False

    # 5. Reaction picker
    emoji = service.pick_reaction()
    assert emoji in CHATTER_REACTIONS

    service.close()


@pytest.mark.asyncio
async def test_chatter_handlers_and_callbacks():
    from bot import (
        chatter_callback_handler,
        chatter_command_handler,
        chatter_message_listener,
        chatter_service,
    )

    chat = Chat(id=-100123456789, type="supergroup", title="D&D Tavern")
    user = User(id=123456789, is_bot=False, first_name="Admin")
    bot = MagicMock()
    bot.get_me = AsyncMock(
        return_value=User(id=999999, is_bot=True, first_name="BardBot", username="bard_bot")
    )
    bot.get_chat_member = AsyncMock()

    member_mock = MagicMock()
    member_mock.status = "administrator"
    bot.get_chat_member.return_value = member_mock

    # 1. /chatter command without args
    msg = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="/chatter",
    ).as_(bot)

    with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
        await chatter_command_handler(msg, CommandObject(prefix="/", command="chatter", args=""))
        assert mock_reply.call_count == 1
        assert "Tavern Chatter Settings" in mock_reply.call_args[0][0]
        assert "🔴 Disabled" in mock_reply.call_args[0][0]

    # 2. /chatter on
    with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
        await chatter_command_handler(msg, CommandObject(prefix="/", command="chatter", args="on"))
        assert mock_reply.call_count == 1
        assert "🟢 Enabled" in mock_reply.call_args[0][0]

    cfg = await chatter_service.get_config(-100123456789)
    assert cfg.enabled is True

    # 3. Callback toggle
    cb_msg = Message(
        message_id=2,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="Status card",
    ).as_(bot)
    cb = CallbackQuery(
        id="cb1",
        from_user=user,
        chat_instance="ci1",
        data="chatter:toggle:-100123456789",
        message=cb_msg,
    ).as_(bot)

    with (
        patch.object(CallbackQuery, "answer", new_callable=AsyncMock) as mock_cb_answer,
        patch.object(Message, "edit_text", new_callable=AsyncMock) as mock_edit,
    ):
        await chatter_callback_handler(cb)
        assert mock_cb_answer.call_count == 1
        assert mock_edit.call_count == 1
        cfg = await chatter_service.get_config(-100123456789)
        assert cfg.enabled is False

    # 4. Message listener passive training and empty-reply
    await chatter_service.update_config(-100123456789, enabled=True, reply_chance=1.0)
    chat_msg = Message(
        message_id=3,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="Лицарі п'ють ель у таверні щовечора.",
    ).as_(bot)

    with patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply:
        await chatter_message_listener(chat_msg)
        # Vocabulary now has words, should generate reply
        assert mock_reply.call_count == 1
        reply_txt = mock_reply.call_args[0][0]
        assert len(reply_txt) > 0
