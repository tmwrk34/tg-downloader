import asyncio

import pytest

from services.games.poker.engine import PokerTable
from services.games.poker.models import GameStage, MoveType, TableConfig


@pytest.mark.asyncio
async def test_poker_table_sit_and_start_hand():
    table = PokerTable(chat_id=-100123)

    # Player 1 sits
    ok, msg = table.sit_down(user_id=1, display_name="Alice", preferred_seat=0)
    assert ok is True
    assert table.seats[0].display_name == "Alice"

    # Cannot start with 1 player
    ok, msg = table.start_hand()
    assert ok is False
    assert "мінімум 2" in msg

    # Player 2 sits
    ok, msg = table.sit_down(user_id=2, display_name="Bob", preferred_seat=1)
    assert ok is True

    # Start hand
    ok, msg = table.start_hand()
    assert ok is True
    assert table.stage == GameStage.PREFLOP
    assert table.pot == 30  # SB (10) + BB (20)
    assert len(table.seats[0].hole_cards) == 2
    assert len(table.seats[1].hole_cards) == 2

    # In heads-up, Dealer is SB (seat 0 in this case) and acts first preflop
    assert table.current_turn_seat is not None


@pytest.mark.asyncio
async def test_poker_betting_round_and_fold():
    table = PokerTable(chat_id=-100123)
    table.sit_down(user_id=1, display_name="Alice", preferred_seat=0)
    table.sit_down(user_id=2, display_name="Bob", preferred_seat=1)
    table.start_hand()

    first_actor = table.seats[table.current_turn_seat]
    other_player = table.seats[1 if table.current_turn_seat == 0 else 0]

    # First actor folds -> other player immediately wins remaining pot
    initial_pot = table.pot
    ok, msg = table.player_move(first_actor.user_id, MoveType.FOLD)
    assert ok is True
    assert table.stage == GameStage.HAND_ENDED
    assert len(table.winners) == 1
    assert table.winners[0].player_id == other_player.user_id
    assert table.winners[0].amount_won == initial_pot


@pytest.mark.asyncio
async def test_poker_full_hand_to_showdown():
    # 2 players: Alice and Bob check/call all the way down to River showdown
    config = TableConfig(turn_timeout_seconds=5)
    table = PokerTable(chat_id=-100123, config=config)
    table.sit_down(user_id=1, display_name="Alice", preferred_seat=0)
    table.sit_down(user_id=2, display_name="Bob", preferred_seat=1)
    table.start_hand()

    # Preflop: SB calls BB (10 chips to call), BB checks
    sb_player = table.seats[table.current_turn_seat]
    bb_player = table.seats[1 if table.current_turn_seat == 0 else 0]

    # SB calls
    ok, msg = table.player_move(sb_player.user_id, MoveType.CALL)
    assert ok is True

    # BB checks option
    ok, msg = table.player_move(bb_player.user_id, MoveType.CHECK)
    assert ok is True

    # Now on Flop
    assert table.stage == GameStage.FLOP
    assert len(table.community_cards) == 3

    # Flop: both check
    actor1 = table.seats[table.current_turn_seat]
    table.player_move(actor1.user_id, MoveType.CHECK)
    actor2 = table.seats[table.current_turn_seat]
    table.player_move(actor2.user_id, MoveType.CHECK)

    # Now on Turn
    assert table.stage == GameStage.TURN
    assert len(table.community_cards) == 4

    # Turn: both check
    actor1 = table.seats[table.current_turn_seat]
    table.player_move(actor1.user_id, MoveType.CHECK)
    actor2 = table.seats[table.current_turn_seat]
    table.player_move(actor2.user_id, MoveType.CHECK)

    # Now on River
    assert table.stage == GameStage.RIVER
    assert len(table.community_cards) == 5

    # River: both check -> Showdown
    actor1 = table.seats[table.current_turn_seat]
    table.player_move(actor1.user_id, MoveType.CHECK)
    actor2 = table.seats[table.current_turn_seat]
    table.player_move(actor2.user_id, MoveType.CHECK)

    assert table.stage == GameStage.HAND_ENDED
    assert len(table.winners) >= 1
    total_won = sum(w.amount_won for w in table.winners)
    assert total_won == 40  # 20 + 20


@pytest.mark.asyncio
async def test_poker_turn_timeout_auto_resolves():
    config = TableConfig(turn_timeout_seconds=1)
    table = PokerTable(chat_id=-100123, config=config)
    table.sit_down(user_id=1, display_name="Alice", preferred_seat=0)
    table.sit_down(user_id=2, display_name="Bob", preferred_seat=1)
    table.start_hand()

    # Active player has to call or fold. Wait 1.2s for timeout
    await asyncio.sleep(1.2)

    # Hand should have auto-folded for timed out player
    assert table.stage == GameStage.HAND_ENDED
    assert len(table.winners) == 1
