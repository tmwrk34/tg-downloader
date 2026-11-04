from services.games.poker.cards import Card, Deck, Rank, Suit
from services.games.poker.evaluator import HandRank, evaluate_hand
from services.games.poker.pots import calculate_pots


def test_deck_and_cards():
    deck = Deck()
    assert deck.remaining == 52

    cards = deck.draw(5)
    assert len(cards) == 5
    assert deck.remaining == 47

    card = Card.from_str("As")
    assert card.rank == Rank.ACE
    assert card.suit == Suit.SPADES
    assert card.display == "A♠"
    assert not card.is_red

    card_red = Card.from_str("10h")
    assert card_red.rank == Rank.TEN
    assert card_red.suit == Suit.HEARTS
    assert card_red.display == "10♥"
    assert card_red.is_red


def test_hand_evaluator_royal_flush():
    cards = [
        Card.from_str("As"),
        Card.from_str("Ks"),
        Card.from_str("Qs"),
        Card.from_str("Js"),
        Card.from_str("10s"),
        Card.from_str("2c"),
        Card.from_str("3d"),
    ]
    res = evaluate_hand(cards)
    assert res.rank == HandRank.ROYAL_FLUSH
    assert res.tie_breakers == (14,)
    assert "Роял-флеш" in res.description_uk


def test_hand_evaluator_wheel_straight():
    # Ace-to-5 straight (Wheel)
    cards = [
        Card.from_str("Ah"),
        Card.from_str("2c"),
        Card.from_str("3s"),
        Card.from_str("4d"),
        Card.from_str("5h"),
        Card.from_str("9d"),
        Card.from_str("Kd"),
    ]
    res = evaluate_hand(cards)
    assert res.rank == HandRank.STRAIGHT
    assert res.tie_breakers == (5,)
    assert "Стріт до П'ятіркаа" in res.description_uk or "Стріт" in res.description_uk


def test_hand_evaluator_full_house_vs_flush():
    fh_cards = [
        Card.from_str("Kh"),
        Card.from_str("Kd"),
        Card.from_str("Ks"),
        Card.from_str("2c"),
        Card.from_str("2d"),
        Card.from_str("7h"),
        Card.from_str("8c"),
    ]
    flush_cards = [
        Card.from_str("Ah"),
        Card.from_str("Jh"),
        Card.from_str("8h"),
        Card.from_str("6h"),
        Card.from_str("2h"),
        Card.from_str("Kc"),
        Card.from_str("Qd"),
    ]
    fh_eval = evaluate_hand(fh_cards)
    flush_eval = evaluate_hand(flush_cards)

    assert fh_eval.rank == HandRank.FULL_HOUSE
    assert flush_eval.rank == HandRank.FLUSH
    assert fh_eval > flush_eval


def test_calculate_pots_simple():
    # 3 players bet 100 each, all active
    bets = {1: 100, 2: 100, 3: 100}
    active = {1, 2, 3}
    pots = calculate_pots(bets, active)
    assert len(pots) == 1
    assert pots[0].amount == 300
    assert pots[0].eligible_player_ids == {1, 2, 3}


def test_calculate_pots_side_pots():
    # Player 1 is All-In for 50
    # Player 2 & 3 bet 150 each
    bets = {1: 50, 2: 150, 3: 150}
    active = {1, 2, 3}
    pots = calculate_pots(bets, active)

    assert len(pots) == 2
    # Main pot: 50 * 3 = 150 (eligible: 1, 2, 3)
    assert pots[0].amount == 150
    assert pots[0].eligible_player_ids == {1, 2, 3}

    # Side pot: (150 - 50) * 2 = 200 (eligible: 2, 3)
    assert pots[1].amount == 200
    assert pots[1].eligible_player_ids == {2, 3}


def test_calculate_pots_folded_player_chips_in_pot():
    # Player 1 folded after putting 40
    # Player 2 and 3 bet 100 each
    bets = {1: 40, 2: 100, 3: 100}
    active = {2, 3}  # Player 1 folded
    pots = calculate_pots(bets, active)

    assert len(pots) == 1
    assert pots[0].amount == 240
    assert pots[0].eligible_player_ids == {2, 3}
