"""
Unit and integration tests for Issue #6:
- Hybrid Long-Term Memory Engine (SQLite FTS5 BM25 + 384-d Vector Cosine + LanceDB/fastembed adapter)
- Smart Noise Gate & Episodic Chunking
- Cosine Deduplication (> 0.90) & Token Budget XML Prompt Injection
- User Dossiers & Tavern Lore
- Agentic Memory Tool Calling (`remember_fact`, `update_user_dossier`, `forget_fact`, `search_tavern_archives`)
- Direct Reply Routing to Munito AI (`chatter_message_listener`)
- `/memory` Command Handler
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.filters import CommandObject
from aiogram.types import Chat, Message, User

from handlers.ai import chatter_message_listener, memory_command_handler
from services.memory import (
    EMBEDDING_DIM,
    HybridEmbedder,
    cosine_similarity,
    is_meaningful_message,
)
from services.munito import (
    AI_TOOLS,
    DEFAULT_SYSTEM_PROMPT,
    MunitoAIService,
)


def test_smart_noise_gate():
    # Rejects empty, commands, bare URLs, short reactions, laughter
    assert is_meaningful_message("") is False
    assert is_meaningful_message("   ") is False
    assert is_meaningful_message("/roll 2d20+5") is False
    assert is_meaningful_message("/ai context 15") is False
    assert is_meaningful_message("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is False
    assert is_meaningful_message("ахахахахах") is False
    assert is_meaningful_message("база") is False
    assert is_meaningful_message("ок") is False
    assert is_meaningful_message("++") is False
    assert is_meaningful_message("Привіт всім") is False  # < 25 chars

    # Accepts meaningful conversational turns
    assert (
        is_meaningful_message("Ми вчора підняли новий сервер у Франкфурті на 16 гігабайт.") is True
    )
    assert (
        is_meaningful_message("Ілля тепер грає за паладина клятви помсти в нашій кампанії D&D.")
        is True
    )


def test_multilingual_embedder_and_morphology():
    embedder = HybridEmbedder(use_fastembed=False)
    assert embedder.backend_name == "multilingual-subword-384"

    v1 = embedder.embed("Ми орендували виділений сервер у Франкфурті для нашого бота.")
    v2 = embedder.embed("Ми орендували виділений сервер в Франкфурті для нашого бота!")
    v_query = embedder.embed("Що там по серверу у Франкфурті?")
    v_unrelated = embedder.embed("Рецепт смачного яблучного пирога з корицею та ваніллю.")

    assert v1.shape == (EMBEDDING_DIM,)
    # Near-duplicates have > 0.90 cosine similarity
    assert cosine_similarity(v1, v2) > 0.90
    # Morphologically inflected query has strong similarity vs unrelated text
    assert cosine_similarity(v1, v_query) > 0.30
    assert cosine_similarity(v1, v_query) > cosine_similarity(v1, v_unrelated) + 0.20


@pytest.mark.asyncio
async def test_hybrid_search_and_deduplication(tmp_path: Path):
    db_file = tmp_path / "test_hybrid.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100777

    # 1. Insert first fact
    entry1, is_new1 = await svc.remember_fact(
        chat_id=chat_id,
        content="Наш основний продакшн сервер знаходиться у Франкфурті на Hetzner.",
        category="fact",
        subject="Сервер",
    )
    assert is_new1 is True
    assert entry1.category == "fact"

    # 2. Insert near-duplicate fact (similarity > 0.90) -> should update existing instead of duplicating
    entry2, is_new2 = await svc.remember_fact(
        chat_id=chat_id,
        content="Наш основний продакшн сервер знаходиться в Франкфурті на Hetzner!",
        category="fact",
        subject="Сервер",
    )
    assert is_new2 is False
    assert entry2.id == entry1.id

    stats = await svc.get_memory_stats(chat_id)
    assert stats.total_memories == 1
    assert stats.lore_count == 1

    # 3. Insert second distinct fact
    await svc.remember_fact(
        chat_id=chat_id,
        content="Щовівторка о 20:00 ми збираємося грати в D&D кампанію Curse of Strahd.",
        category="lore",
        subject="Розклад D&D",
    )

    # 4. Hybrid search (Vector + FTS5 BM25)
    results_srv = await svc.search_memories(
        chat_id=chat_id,
        query="Де знаходиться наш сервер?",
        top_k=3,
        min_score=0.50,
    )
    assert len(results_srv) >= 1
    assert "Франкфурті" in results_srv[0].content
    assert results_srv[0].score >= 0.72

    results_dnd = await svc.search_memories(
        chat_id=chat_id,
        query="Коли ми граємо в D&D Strahd?",
        top_k=3,
        min_score=0.50,
    )
    assert len(results_dnd) >= 1
    assert "Curse of Strahd" in results_dnd[0].content

    # 5. Chat isolation check: another chat_id sees nothing
    results_other = await svc.search_memories(
        chat_id=-100999,
        query="Де знаходиться наш сервер?",
        top_k=3,
        min_score=0.10,
    )
    assert results_other == []

    svc.close()


@pytest.mark.asyncio
async def test_episodic_chunking_and_noise_filtering(tmp_path: Path):
    db_file = tmp_path / "test_episodes.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100500

    # Short noise messages should be ignored by the episodic buffer
    await svc.record_turn(chat_id, role="user", content="ахахах", user_id=1, user_name="Ілля")
    await svc.record_turn(chat_id, role="user", content="база", user_id=2, user_name="Влад")
    await svc.record_turn(
        chat_id, role="user", content="https://tiktok.com/@a/video/1", user_id=1, user_name="Ілля"
    )

    stats = await svc.get_memory_stats(chat_id)
    assert stats.buffered_turns == 0
    assert stats.episodic_count == 0

    # Feed 5 meaningful turns -> automatically flushes into 1 episodic memory chunk
    meaningful_turns = [
        ("Ілля", 1, "Слухайте, давайте обговоримо архітектуру бази даних для нашого проекту."),
        ("Влад", 2, "Я пропоную використати SQLite в режимі WAL разом із таблицями FTS5."),
        ("Ілля", 1, "Звучить чудово, а для векторного пошуку додамо ембедінги на 384 виміри."),
        ("Олег", 3, "Головне не тягнути важкий PyTorch, щоб Docker-образ залишався легким."),
        ("Влад", 2, "Домовилися, робимо легковаговий гібридний пошук без зайвих залежностей."),
    ]
    for name, uid, text in meaningful_turns:
        await svc.record_turn(chat_id, role="user", content=text, user_id=uid, user_name=name)

    stats_after = await svc.get_memory_stats(chat_id)
    assert stats_after.buffered_turns == 0
    assert stats_after.episodic_count == 1

    # Search for the episode
    found = await svc.search_memories(
        chat_id, query="Що вирішили по SQLite та PyTorch?", top_k=3, min_score=0.50
    )
    assert len(found) == 1
    assert found[0].category == "episodic"
    assert "Ілля, Влад, Олег" in found[0].participants
    assert "PyTorch" in found[0].content

    svc.close()


@pytest.mark.asyncio
async def test_user_dossiers_and_dynamic_xml_prompt_injection(tmp_path: Path):
    db_file = tmp_path / "test_dossiers.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100321

    # Add user dossier traits (including deduplication of similar traits)
    d1 = await svc.update_user_dossier(
        chat_id=chat_id,
        user_name="Illya",
        trait_or_note="Пише бекенд на Python та підтримує тавернського бота",
        user_id=42,
        username="illya_dev",
    )
    assert len(d1.traits) == 1

    # Near-duplicate trait updates existing entry rather than adding a second one
    d2 = await svc.update_user_dossier(
        chat_id=chat_id,
        user_name="Illya",
        trait_or_note="Пише бекенд на Python та підтримує тавернського бота!",
        user_id=42,
    )
    assert len(d2.traits) == 1

    # Distinct trait is appended
    d3 = await svc.update_user_dossier(
        chat_id=chat_id,
        user_name="Illya",
        trait_or_note="Грає за барда в настільних RPG",
        user_id=42,
    )
    assert len(d3.traits) == 2

    # Add tavern lore and an episodic memory
    await svc.remember_fact(
        chat_id=chat_id,
        content="У таверні заборонено згадувати старий монолітний bot.py на 2000 рядків.",
        category="lore",
        subject="Правило таверни",
    )
    await svc.remember_fact(
        chat_id=chat_id,
        content="Минулого тижня ми зробили генератор середньовічних сувоїв із сургучною печаткою.",
        category="event",
        subject="Генератор сувоїв",
    )

    # Verify generate_reply injects <tavern_lore>, <user_dossiers>, and <retrieved_memories>
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Пам'ятаю все, друже!"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }
    mock_resp.__aenter__.return_value = mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__.return_value = mock_session

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await svc.generate_reply(
            prompt="Розкажи, що ми робили з генератором сувоїв і сургучною печаткою?",
            chat_id=chat_id,
            user_id=42,
            user_name="Illya",
        )

        sent_payload = mock_session.post.call_args[1]["json"]
        system_content = sent_payload["messages"][0]["content"]

        assert "<tavern_lore>" in system_content
        assert "Правило таверни" in system_content
        assert "<user_dossiers>" in system_content
        assert "Illya (@illya_dev)" in system_content
        assert "Грає за барда" in system_content

    svc.close()


@pytest.mark.asyncio
async def test_agentic_memory_tool_calls(tmp_path: Path):
    db_file = tmp_path / "test_agent_tools.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100888

    # Verify all 5 tools are registered in AI_TOOLS
    tool_names = {t["function"]["name"] for t in AI_TOOLS}
    assert {
        "roll_dice",
        "remember_fact",
        "update_user_dossier",
        "forget_fact",
        "search_tavern_archives",
    }.issubset(tool_names)
    assert "80/20 Rule" in DEFAULT_SYSTEM_PROMPT

    # Turn 1: Model invokes remember_fact, update_user_dossier, and search_tavern_archives
    mock_resp1 = AsyncMock()
    mock_resp1.status = 200
    mock_resp1.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_rem_1",
                            "type": "function",
                            "function": {
                                "name": "remember_fact",
                                "arguments": json.dumps(
                                    {
                                        "category": "inside_joke",
                                        "subject": "Золотий дракон",
                                        "fact": "Влад одного разу викинув три одиниці підряд на d20.",
                                    }
                                ),
                            },
                        },
                        {
                            "id": "call_dos_1",
                            "type": "function",
                            "function": {
                                "name": "update_user_dossier",
                                "arguments": json.dumps(
                                    {
                                        "user_name": "Влад",
                                        "trait_or_note": "Має легендарне невезіння на кубиках d20",
                                    }
                                ),
                            },
                        },
                        {
                            "id": "call_search_1",
                            "type": "function",
                            "function": {
                                "name": "search_tavern_archives",
                                "arguments": json.dumps({"query": "Влад одиниці d20"}),
                            },
                        },
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 25, "total_tokens": 65},
    }
    mock_resp1.__aenter__.return_value = mock_resp1

    mock_resp2 = AsyncMock()
    mock_resp2.status = 200
    mock_resp2.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Записав у тавернський літопис! Три одиниці Влада тепер не забудуться.",
                }
            }
        ],
        "usage": {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75},
    }
    mock_resp2.__aenter__.return_value = mock_resp2

    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=[mock_resp1, mock_resp2])
    mock_session.__aenter__.return_value = mock_session

    with patch("aiohttp.ClientSession", return_value=mock_session):
        reply = await svc.generate_reply(
            prompt="Запам'ятай, як Влад викинув три одиниці підряд!",
            chat_id=chat_id,
            user_id=101,
            user_name="Влад",
        )
        assert "Записав у тавернський літопис" in reply

    # Verify fact and dossier were persisted
    lore = await svc.get_chat_lore(chat_id)
    assert len(lore) == 1
    assert "три одиниці" in lore[0].content

    dossiers = await svc.get_user_dossiers(chat_id)
    assert len(dossiers) == 1
    assert dossiers[0].display_name == "Влад"
    assert "невезіння на кубиках" in dossiers[0].traits[0]

    # Now test forget_fact tool call
    messages: list[dict] = []
    svc._execute_tool_calls(
        tool_calls=[
            {
                "id": "call_forget_1",
                "type": "function",
                "function": {
                    "name": "forget_fact",
                    "arguments": json.dumps({"query": "три одиниці підряд"}),
                },
            }
        ],
        messages=messages,
        chat_id=chat_id,
    )
    assert len(messages) == 1
    out = json.loads(messages[0]["content"])
    assert out["status"] == "deleted"
    assert out["deleted_count"] == 1
    assert await svc.get_chat_lore(chat_id) == []

    svc.close()


@pytest.mark.asyncio
async def test_direct_reply_to_bot_routes_to_munito_ai(tmp_path: Path):
    db_file = tmp_path / "test_reply_route.db"
    svc = MunitoAIService(api_key="valid_key", db_path=db_file)
    chat_id = -100444

    fake_bot = MagicMock()
    bot_user = User(id=999999, is_bot=True, first_name="BardBot", username="bard_taverna_bot")
    fake_bot.get_me = AsyncMock(return_value=bot_user)

    chat = Chat(id=chat_id, type="supergroup", title="Tavern")
    human_user = User(id=12345, is_bot=False, first_name="Illya")

    bot_prev_msg = Message(
        message_id=10,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=bot_user,
        text="Якої ти думки про новий патч?",
    )

    # Human replies directly to bot's message WITHOUT typing @bard_taverna_bot
    reply_msg = Message(
        message_id=11,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=human_user,
        text="Думаю, баланс класів став значно кращим!",
        reply_to_message=bot_prev_msg,
    ).as_(fake_bot)

    with (
        patch("handlers.ai.bot.munito_service", svc),
        patch("handlers.ai._handle_ai_mention", new_callable=AsyncMock) as mock_ai_mention,
    ):
        await chatter_message_listener(reply_msg, active_bot=fake_bot)
        mock_ai_mention.assert_awaited_once_with(
            reply_msg, "Думаю, баланс класів став значно кращим!", fake_bot
        )

    svc.close()


@pytest.mark.asyncio
async def test_memory_command_handler(tmp_path: Path):
    db_file = tmp_path / "test_mem_cmd.db"
    svc = MunitoAIService(api_key="valid_key", db_path=db_file)
    chat_id = -100654

    fake_bot = MagicMock()
    member_mock = MagicMock()
    member_mock.status = "administrator"
    fake_bot.get_chat_member = AsyncMock(return_value=member_mock)

    chat = Chat(id=chat_id, type="supergroup", title="Memory Tavern")
    user = User(id=777, is_bot=False, first_name="Illya")

    msg = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="/memory",
    ).as_(fake_bot)

    with (
        patch("handlers.ai.bot.munito_service", svc),
        patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply,
    ):
        # 1. /memory remember
        await memory_command_handler(
            msg, CommandObject(prefix="/", command="memory", args="remember Таверна працює 24/7")
        )
        assert "Recorded new lore" in mock_reply.call_args[0][0]

        # 2. /memory search
        await memory_command_handler(
            msg, CommandObject(prefix="/", command="memory", args="search Таверна 24/7")
        )
        assert "Search Results" in mock_reply.call_args[0][0]

        # 3. /memory status card
        await svc.update_user_dossier(chat_id, "Illya", "Майстер таверни", user_id=777)
        await memory_command_handler(msg, CommandObject(prefix="/", command="memory", args=""))
        card_out = mock_reply.call_args[0][0]
        assert "Tavern Archives" in card_out
        assert "Таверна працює 24/7" in card_out
        assert "Майстер таверни" in card_out

        # 4. /memory dossier
        await memory_command_handler(
            msg, CommandObject(prefix="/", command="memory", args="dossier")
        )
        assert "Tavern Member Dossiers" in mock_reply.call_args[0][0]

        # 5. /memory clear
        await memory_command_handler(msg, CommandObject(prefix="/", command="memory", args="clear"))
        assert "Long-term tavern archives wiped!" in mock_reply.call_args[0][0]
        stats = await svc.get_memory_stats(chat_id)
        assert stats.total_memories == 0
        assert stats.dossier_count == 0

    svc.close()


@pytest.mark.asyncio
async def test_dossier_deduplication_and_alias_merging(tmp_path: Path):
    """Verify automatic merging of bot aliases and redundant member traits."""
    db_file = tmp_path / "test_dedup.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -1002366722481

    # 1. Simulate duplicate bot entries (e.g. from user prompt variations)
    await svc.update_user_dossier(chat_id, "Бард", "Официально признанный бармен и бард.")
    await svc.update_user_dossier(
        chat_id,
        "Bard (@bard_taverna_bot)",
        "Официально признанный бард и бармен таверны.",
    )

    # 2. Simulate duplicate human user traits
    await svc.update_user_dossier(
        chat_id,
        "Illya",
        "Разработчик таверн-бота: прикрутил память, новые фичи цитат.",
    )
    await svc.update_user_dossier(
        chat_id,
        "Illya",
        "Называет Барда балбесом и как Житомир.",
    )
    # Add an update trait that supersedes the first
    await svc.update_user_dossier(
        chat_id,
        "Illya",
        "Разработчик таверн-бота: прикрутил память, новые фичи цитат (q/f/s). Из досье балбес убрано.",
    )

    # Before dedup, get_user_dossiers automatically runs deduplication
    dossiers = await svc.get_user_dossiers(chat_id, limit=10)

    # Bot dossiers must be merged into 1 canonical dossier
    bot_dossiers = [d for d in dossiers if d.user_id == -999999 or "Бард" in d.display_name]
    assert len(bot_dossiers) == 1
    assert bot_dossiers[0].display_name == "Бард"
    assert bot_dossiers[0].username == "bard_taverna_bot"
    assert len(bot_dossiers[0].traits) == 1  # Near-duplicate traits merged into 1

    # Illya dossier should have near-duplicate developer traits merged
    illya_dossiers = [d for d in dossiers if d.display_name == "Illya"]
    assert len(illya_dossiers) == 1
    assert len(illya_dossiers[0].traits) <= 2
    # Verify the more complete trait was preserved
    combined_traits = " ".join(illya_dossiers[0].traits)
    assert "Из досье балбес убрано" in combined_traits

    svc.close()


@pytest.mark.asyncio
async def test_manage_user_dossier_actions_and_forget(tmp_path: Path):
    """Test replace, delete, merge actions and forget_dossier_traits."""
    db_file = tmp_path / "test_actions.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100555

    # 1. Add traits and replace with clean summary
    await svc.update_user_dossier(chat_id, "Farid", "Любит шашлык.")
    await svc.update_user_dossier(chat_id, "Farid", "В таверне его называют мясником.")
    d = await svc.get_user_dossiers(chat_id)
    assert len(d[0].traits) == 2

    # Replace action
    await svc.update_user_dossier(
        chat_id,
        "Farid",
        "Шеф-повар таверны: готовит лучший шашлык в округе.",
        action="replace",
    )
    d = await svc.get_user_dossiers(chat_id)
    assert len(d[0].traits) == 1
    assert "Шеф-повар таверны" in d[0].traits[0]

    # 2. Delete specific trait
    await svc.update_user_dossier(chat_id, "Emel'en", "Любит пиво.")
    await svc.update_user_dossier(chat_id, "Emel'en", "Спит на лавке.")
    success, del_count, rem = await svc.delete_user_dossier(chat_id, "Emel'en", trait_query="лавка")
    assert success is True
    assert del_count == 1
    assert len(rem) == 1
    assert rem[0] == "Любит пиво."

    # 3. forget_dossier_traits across chat
    await svc.update_user_dossier(chat_id, "Vova", "Постоянно кричит 'балбес'.")
    deleted_traits = await svc.forget_dossier_traits(chat_id, "балбес")
    assert len(deleted_traits) >= 1
    assert any("Vova" in dt["user_name"] for dt in deleted_traits)

    # 4. Merge action
    await svc.update_user_dossier(chat_id, "OldNick", "Старый следопыт.")
    merged_dossier = await svc.merge_user_dossiers(chat_id, "OldNick", "NewNick")
    assert merged_dossier.display_name == "NewNick"
    assert any("Старый следопыт" in t for t in merged_dossier.traits)

    # Verify OldNick is gone
    d_list = await svc.get_user_dossiers(chat_id)
    assert not any(d.display_name == "OldNick" for d in d_list)

    svc.close()


@pytest.mark.asyncio
async def test_memory_command_dedup_and_forget(tmp_path: Path):
    """Test /memory dedup and /memory forget commands in Telegram."""
    db_file = tmp_path / "test_cmd_dedup.db"
    svc = MunitoAIService(api_key="test_key", db_path=db_file)
    chat_id = -100777
    user_id = 999

    fake_bot = AsyncMock()
    fake_bot.get_chat_member.return_value = MagicMock(status="administrator")
    chat = Chat(id=chat_id, type="supergroup", title="Test Tavern")
    user = User(id=user_id, is_bot=False, first_name="Illya")
    msg = Message(
        message_id=50,
        date=datetime.datetime.now(datetime.UTC),
        chat=chat,
        from_user=user,
        text="/memory",
    ).as_(fake_bot)

    # Populate duplicate dossiers and lore
    await svc.remember_fact(chat_id, "В таверне запрещено разбивать кружки.", category="rule")
    await svc.update_user_dossier(chat_id, "Бард", "Официально признанный бард.")
    await svc.update_user_dossier(chat_id, "Bard (@bard_taverna_bot)", "Бард и бармен таверны.")

    with (
        patch("handlers.ai.bot.munito_service", svc),
        patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply,
    ):
        # 1. /memory dedup
        await memory_command_handler(msg, CommandObject(prefix="/", command="memory", args="dedup"))
        assert "Dossier Deduplication Complete" in mock_reply.call_args[0][0]

        # 2. /memory forget кружки (removes from lore)
        mock_reply.reset_mock()
        await memory_command_handler(
            msg, CommandObject(prefix="/", command="memory", args="forget кружки")
        )
        assert "Forgotten" in mock_reply.call_args[0][0]

        # 3. /memory dossier delete Бард
        mock_reply.reset_mock()
        await memory_command_handler(
            msg, CommandObject(prefix="/", command="memory", args="dossier delete Бард")
        )
        assert "Completely deleted dossier" in mock_reply.call_args[0][0]

    svc.close()
