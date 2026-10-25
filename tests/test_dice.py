from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from aiogram.types import Chat, Message, User

from filters.dice import DiceFilter
from services.dice import (
    DiceParser,
    get_secret_roll,
    store_secret_roll,
)


def create_message(text: str | None = None) -> Message:
    return Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=-100123456789, type="supergroup"),
        from_user=User(id=999, is_bot=False, first_name="Adventurer"),
        text=text,
    )


def test_dice_basic_rolls():
    with patch("services.dice._rng.randint", side_effect=[3, 4]):
        res = DiceParser.parse_and_roll("2d6+3")
        assert res.total == 10  # 3 + 4 + 3
        assert res.normalized_expression == "2d6+3"
        assert len(res.dice_terms) == 1
        assert res.dice_terms[0].count == 2
        assert res.dice_terms[0].sides == 6
        assert res.constant_modifier == 3
        assert "🎲 <b>Adventurer's Roll:</b> <code>2d6+3</code>" in res.format_html("Adventurer")
        assert "🎯 <b>Result:</b> <b>10</b>" in res.format_html("Adventurer")


def test_dice_text_modifiers():
    with patch("services.dice._rng.randint", side_effect=[2, 5]):
        res = DiceParser.parse_and_roll("2d6p4")
        assert res.normalized_expression == "2d6+4"
        assert res.total == 11  # 2 + 5 + 4

    with patch("services.dice._rng.randint", side_effect=[4, 3]):
        res = DiceParser.parse_and_roll("2d6m2")
        assert res.normalized_expression == "2d6-2"
        assert res.total == 5  # 4 + 3 - 2


def test_dice_keep_highest():
    # 4d6k3: keep 3 highest
    with patch("services.dice._rng.randint", side_effect=[1, 6, 4, 3]):
        res = DiceParser.parse_and_roll("4d6k3")
        assert res.total == 13  # 6 + 4 + 3 (drops 1)
        details = res.dice_terms[0].format_details()
        assert "<s>1</s>" in details
        assert "6" in details and "4" in details and "3" in details


def test_dice_keep_lowest():
    # 4d6kl2: keep 2 lowest
    with patch("services.dice._rng.randint", side_effect=[1, 6, 4, 3]):
        res = DiceParser.parse_and_roll("4d6kl2")
        assert res.total == 4  # 1 + 3 (drops 6, 4)
        details = res.dice_terms[0].format_details()
        assert "<s>6</s>" in details and "<s>4</s>" in details


def test_dice_drop_lowest():
    # 4d6d2 / 4d6dl2: drop 2 lowest
    with patch("services.dice._rng.randint", side_effect=[1, 6, 4, 3]):
        res = DiceParser.parse_and_roll("4d6d2")
        assert res.total == 10  # drops 1, 3 -> keeps 6, 4
        details = res.dice_terms[0].format_details()
        assert "<s>1</s>" in details and "<s>3</s>" in details


def test_dice_drop_highest():
    # 4d6dh1: drop 1 highest
    with patch("services.dice._rng.randint", side_effect=[1, 6, 4, 3]):
        res = DiceParser.parse_and_roll("4d6dh1")
        assert res.total == 8  # drops 6 -> keeps 1, 4, 3
        details = res.dice_terms[0].format_details()
        assert "<s>6</s>" in details


def test_dice_combined_modifiers():
    # 4d6k3+2
    with patch("services.dice._rng.randint", side_effect=[2, 5, 4, 6]):
        res = DiceParser.parse_and_roll("4d6k3+2")
        assert res.total == 17  # (5 + 4 + 6) + 2

    # 4d6d2m3
    with patch("services.dice._rng.randint", side_effect=[1, 2, 5, 6]):
        res = DiceParser.parse_and_roll("4d6d2m3")
        assert res.total == 8  # (5 + 6) - 3


def test_critical_hit_and_fail():
    # Natural 20
    with patch("services.dice._rng.randint", return_value=20):
        res = DiceParser.parse_and_roll("1d20")
        assert res.is_crit_hit is True
        assert res.is_crit_fail is False
        assert "CRITICAL HIT! (Natural 20)" in res.format_html()

    # Natural 1
    with patch("services.dice._rng.randint", return_value=1):
        res = DiceParser.parse_and_roll("d20")
        assert res.is_crit_hit is False
        assert res.is_crit_fail is True
        assert "CRITICAL FAIL! (Natural 1)" in res.format_html()


def test_natural_language_normalization():
    norm, secret = DiceParser.normalize_input("roll 2d6")
    assert norm == "2d6"
    assert secret is False

    norm, secret = DiceParser.normalize_input("throw d20")
    assert norm == "d20"
    assert secret is False

    norm, secret = DiceParser.normalize_input("roll 20 sided dice")
    assert norm == "1d20"
    assert secret is False

    norm, secret = DiceParser.normalize_input("dm roll 2d20")
    assert norm == "2d20"
    assert secret is True

    norm, secret = DiceParser.normalize_input("hidden roll 4d6k3")
    assert norm == "4d6k3"
    assert secret is True

    norm, secret = DiceParser.normalize_input("dmroll 2d6")
    assert norm == "2d6"
    assert secret is True


def test_dice_validation_errors():
    with pytest.raises(ValueError, match="empty"):
        DiceParser.parse_and_roll("")

    with pytest.raises(ValueError, match="Invalid dice term"):
        DiceParser.parse_and_roll("foo")

    with pytest.raises(ValueError, match="Dice count"):
        DiceParser.parse_and_roll("101d6")

    with pytest.raises(ValueError, match="Dice sides"):
        DiceParser.parse_and_roll("1d1001")

    with pytest.raises(ValueError, match="Keep/drop count"):
        DiceParser.parse_and_roll("4d6k5")


def test_secret_roll_cache():
    secret_id = store_secret_roll(
        user_id=12345,
        user_name="Gandalf",
        result_html="<b>Result: 20</b>",
        result_plain="Result: 20",
    )
    assert secret_id is not None
    entry = get_secret_roll(secret_id)
    assert entry is not None
    user_id, name, html, plain = entry
    assert user_id == 12345
    assert name == "Gandalf"
    assert html == "<b>Result: 20</b>"
    assert plain == "Result: 20"

    assert get_secret_roll("nonexistent") is None


@pytest.mark.asyncio
async def test_dice_filter():
    d_filter = DiceFilter()

    # Valid expressions
    res = await d_filter(create_message("2d6+3"))
    assert res == {"dice_expr": "2d6+3", "is_secret": False}

    res = await d_filter(create_message("dm roll 4d6k3"))
    assert res == {"dice_expr": "4d6k3", "is_secret": True}

    # Slash commands should be skipped by DiceFilter
    assert await d_filter(create_message("/roll 2d6")) is False
    assert await d_filter(create_message("/dmroll d20")) is False

    # Plain non-dice messages
    assert await d_filter(create_message("hello world")) is False
    assert await d_filter(create_message("https://youtube.com/watch?v=123")) is False
    assert await d_filter(create_message(None)) is False


def test_cyrillic_transliteration():
    # 2д20 and 2к20
    with patch("services.dice._rng.randint", side_effect=[12, 18]):
        res = DiceParser.parse_and_roll("2д20")
        assert res.total == 30
        assert res.normalized_expression == "2d20"

    with patch("services.dice._rng.randint", side_effect=[10, 15]):
        res = DiceParser.parse_and_roll("2к20")
        assert res.total == 25
        assert res.normalized_expression == "2d20"

    # д20 and к20
    with patch("services.dice._rng.randint", return_value=17):
        res = DiceParser.parse_and_roll("д20")
        assert res.total == 17
        assert res.normalized_expression == "1d20"

    with patch("services.dice._rng.randint", return_value=8):
        res = DiceParser.parse_and_roll("к20")
        assert res.total == 8
        assert res.normalized_expression == "1d20"

    # 4к6к3 and 4к6к3+2
    with patch("services.dice._rng.randint", side_effect=[1, 6, 4, 3]):
        res = DiceParser.parse_and_roll("4к6к3")
        assert res.total == 13
        assert res.normalized_expression == "4d6kh3"

    with patch("services.dice._rng.randint", side_effect=[2, 5, 4, 6]):
        res = DiceParser.parse_and_roll("4к6к3+2")
        assert res.total == 17
        assert res.normalized_expression == "4d6kh3+2"

    # Letter modifiers in Cyrillic: 2д6п3 (+3), 2к6м2 (-2)
    with patch("services.dice._rng.randint", side_effect=[3, 4]):
        res = DiceParser.parse_and_roll("2д6п3")
        assert res.total == 10
        assert res.normalized_expression == "2d6+3"

    with patch("services.dice._rng.randint", side_effect=[5, 3]):
        res = DiceParser.parse_and_roll("2к6м2")
        assert res.total == 6
        assert res.normalized_expression == "2d6-2"

    # Cyrillic prefixes and natural language
    norm, secret = DiceParser.normalize_input("кинь 2д20")
    assert norm == "2д20"
    assert secret is False

    norm, secret = DiceParser.normalize_input("ролл 2к20")
    assert norm == "2к20"
    assert secret is False

    norm, secret = DiceParser.normalize_input("дм ролл 2д20")
    assert norm == "2д20"
    assert secret is True

    norm, secret = DiceParser.normalize_input("дмролл 2к20")
    assert norm == "2к20"
    assert secret is True


@pytest.mark.asyncio
async def test_dice_filter_cyrillic():
    d_filter = DiceFilter()

    res = await d_filter(create_message("2д20"))
    assert res == {"dice_expr": "2д20", "is_secret": False}

    res = await d_filter(create_message("2к20"))
    assert res == {"dice_expr": "2к20", "is_secret": False}

    res = await d_filter(create_message("д20"))
    assert res == {"dice_expr": "д20", "is_secret": False}

    res = await d_filter(create_message("к20"))
    assert res == {"dice_expr": "к20", "is_secret": False}

    res = await d_filter(create_message("дм ролл 2д20"))
    assert res == {"dice_expr": "2д20", "is_secret": True}

    # Slash commands with Cyrillic aliases should be skipped by DiceFilter
    assert await d_filter(create_message("/ролл 2д20")) is False
    assert await d_filter(create_message("/дмролл 2к20")) is False
