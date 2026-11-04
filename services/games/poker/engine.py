"""
Core Texas Hold'em poker table finite state machine (FSM).
Manages seats, blinds, betting rounds, turn timers, player moves,
hand evaluation, and side-pot payout resolution.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from services.games.poker.cards import Card, Deck
from services.games.poker.evaluator import evaluate_hand
from services.games.poker.models import (
    GameStage,
    HandWinner,
    MoveType,
    Player,
    PlayerStatus,
    TableConfig,
)
from services.games.poker.pots import calculate_pots

logger = logging.getLogger(__name__)


class PokerTable:
    """Represents a single Texas Hold'em table associated with a Telegram chat."""

    def __init__(
        self,
        chat_id: int,
        config: TableConfig | None = None,
        on_state_change: Callable[[PokerTable], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self.chat_id = chat_id
        self.config = config or TableConfig()
        self._on_state_change = on_state_change

        self.deck = Deck()
        self.community_cards: list[Card] = []
        self.seats: list[Player | None] = [None] * self.config.max_seats

        self.stage = GameStage.WAITING
        self.dealer_seat: int = 0
        self.current_turn_seat: int | None = None
        self.turn_deadline: float | None = None
        self._turn_timer_task: asyncio.Task[None] | None = None

        self.highest_bet: int = 0
        self.min_raise_step: int = self.config.big_blind
        self.pot: int = 0
        self.last_action_desc: str = "Очікування початку роздачі."
        self.winners: list[HandWinner] = []

    # -------------------------------------------------------------------------
    # Seat & Player Management
    # -------------------------------------------------------------------------
    def get_player(self, user_id: int) -> Player | None:
        for p in self.seats:
            if p and p.user_id == user_id:
                return p
        return None

    def get_occupied_seats(self) -> list[Player]:
        return [p for p in self.seats if p is not None]

    def get_active_in_hand_players(self) -> list[Player]:
        return [
            p
            for p in self.seats
            if p is not None and p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
        ]

    def get_acting_players(self) -> list[Player]:
        """Players who are active and have chips left to bet (not all-in, not folded)."""
        return [p for p in self.seats if p is not None and p.status == PlayerStatus.ACTIVE]

    def sit_down(
        self,
        user_id: int,
        display_name: str,
        username: str | None = None,
        preferred_seat: int | None = None,
    ) -> tuple[bool, str]:
        existing = self.get_player(user_id)
        if existing:
            return False, "Ви вже сидите за цим столом!"

        # Find seat
        if preferred_seat is not None and 0 <= preferred_seat < self.config.max_seats:
            if self.seats[preferred_seat] is not None:
                return False, f"Місце {preferred_seat + 1} вже зайняте!"
            target_seat = preferred_seat
        else:
            empty_seats = [i for i, s in enumerate(self.seats) if s is None]
            if not empty_seats:
                return False, "За столом немає вільних місць!"
            target_seat = empty_seats[0]

        new_player = Player(
            user_id=user_id,
            display_name=display_name,
            username=username,
            chips=self.config.starting_chips,
            seat_index=target_seat,
            status=PlayerStatus.ACTIVE
            if self.stage == GameStage.WAITING
            else PlayerStatus.SITTING_OUT,
        )
        self.seats[target_seat] = new_player
        self.last_action_desc = f"{display_name} сів за стіл (місце {target_seat + 1})."
        return True, f"Ви успішно сіли за місце {target_seat + 1}!"

    def stand_up(self, user_id: int) -> tuple[bool, str]:
        p = self.get_player(user_id)
        if not p:
            return False, "Вас немає за цим столом."

        seat_idx = p.seat_index
        # If hand is in progress and player was active, fold them first
        if (
            self.stage not in (GameStage.WAITING, GameStage.HAND_ENDED)
            and p.status == PlayerStatus.ACTIVE
        ):
            self._handle_fold(p)

        self.seats[seat_idx] = None
        self.last_action_desc = f"{p.display_name} покинув стіл."

        # If table drops below 2 players during hand, check if hand resolves
        if self.stage not in (GameStage.WAITING, GameStage.HAND_ENDED):
            active = self.get_active_in_hand_players()
            if len(active) <= 1:
                self._resolve_single_remaining_winner()

        return True, "Ви встали з-за столу."

    def rebuy(self, user_id: int) -> tuple[bool, str]:
        p = self.get_player(user_id)
        if not p:
            return False, "Вас немає за столом."
        if p.chips > 0:
            return False, f"У вас ще є {p.chips} фішок!"
        if p.status == PlayerStatus.ACTIVE:
            return False, "Не можна брати ребай під час активної роздачі!"

        p.chips = self.config.starting_chips
        p.status = PlayerStatus.ACTIVE
        self.last_action_desc = (
            f"{p.display_name} взяв ребай (+{self.config.starting_chips} фішок)."
        )
        return True, f"Ви отримали {self.config.starting_chips} фішок!"

    # -------------------------------------------------------------------------
    # Game Loop & Hand Start
    # -------------------------------------------------------------------------
    def start_hand(self) -> tuple[bool, str]:
        if self.stage not in (GameStage.WAITING, GameStage.HAND_ENDED):
            return False, "Роздача вже триває!"

        eligible = [p for p in self.seats if p is not None and p.chips >= self.config.big_blind]
        if len(eligible) < self.config.min_players_to_start:
            return False, f"Потрібно мінімум {self.config.min_players_to_start} гравців зі стеком!"

        # Reset state
        self._cancel_turn_timer()
        self.deck.reset()
        self.community_cards = []
        self.winners = []
        self.pot = 0
        self.highest_bet = 0
        self.min_raise_step = self.config.big_blind

        for p in self.seats:
            if p:
                p.reset_for_hand()

        # Advance dealer button to next occupied seat
        self.dealer_seat = self._next_occupied_seat(self.dealer_seat, eligible)

        # Post Blinds
        if len(eligible) == 2:
            # Heads-up: Dealer is SB, other player is BB
            sb_seat = self.dealer_seat
            bb_seat = self._next_occupied_seat(sb_seat, eligible)
            first_to_act = sb_seat
        else:
            sb_seat = self._next_occupied_seat(self.dealer_seat, eligible)
            bb_seat = self._next_occupied_seat(sb_seat, eligible)
            first_to_act = self._next_occupied_seat(bb_seat, eligible)

        sb_player = self.seats[sb_seat]
        bb_player = self.seats[bb_seat]
        assert sb_player and bb_player

        self._commit_bet(sb_player, self.config.small_blind)
        self._commit_bet(bb_player, self.config.big_blind)
        self.highest_bet = self.config.big_blind

        # Deal 2 hole cards to each eligible player
        for p in eligible:
            p.hole_cards = self.deck.draw(2)

        self.stage = GameStage.PREFLOP
        self.last_action_desc = (
            f"Нова роздача! Блайнди: {sb_player.display_name} ({self.config.small_blind}), "
            f"{bb_player.display_name} ({self.config.big_blind})."
        )
        self._set_turn(first_to_act)
        return True, "Роздачу розпочато!"

    def _next_occupied_seat(self, from_seat: int, subset: list[Player] | None = None) -> int:
        pool_ids = (
            {p.seat_index for p in subset}
            if subset is not None
            else {p.seat_index for p in self.seats if p}
        )
        for offset in range(1, self.config.max_seats + 1):
            idx = (from_seat + offset) % self.config.max_seats
            if idx in pool_ids:
                return idx
        return from_seat

    def _commit_bet(self, player: Player, amount: int) -> int:
        """Deduct chips from player and add to pot. Handles all-in if amount >= chips."""
        actual = min(player.chips, amount)
        player.chips -= actual
        player.current_bet += actual
        player.total_hand_bet += actual
        self.pot += actual
        if player.chips == 0:
            player.status = PlayerStatus.ALL_IN
        return actual

    # -------------------------------------------------------------------------
    # Turn Execution & Betting Engine
    # -------------------------------------------------------------------------
    def player_move(
        self, user_id: int, move: MoveType | str, raise_amount: int | None = None
    ) -> tuple[bool, str]:
        if self.stage in (GameStage.WAITING, GameStage.HAND_ENDED, GameStage.SHOWDOWN):
            return False, "Зараз не час для ходів!"

        if self.current_turn_seat is None:
            return False, "Немає активного гравця!"

        current_player = self.seats[self.current_turn_seat]
        if not current_player or current_player.user_id != user_id:
            return False, "Зараз не ваш хід!"

        if isinstance(move, str):
            try:
                move = MoveType(move.lower())
            except ValueError:
                return False, f"Невідомий хід '{move}'"

        to_call = self.highest_bet - current_player.current_bet

        if move == MoveType.FOLD:
            return self._handle_fold(current_player)

        if move == MoveType.CHECK:
            if to_call > 0:
                return (
                    False,
                    f"Не можна чекати! Потрібно зрівняти {to_call} фішок або скинути карти.",
                )
            self.last_action_desc = f"{current_player.display_name}: Чек."
            return self._advance_turn()

        if move == MoveType.CALL:
            if to_call <= 0:
                # If nothing to call, treating call as check
                self.last_action_desc = f"{current_player.display_name}: Чек."
                return self._advance_turn()
            paid = self._commit_bet(current_player, to_call)
            status_note = " (All-In)" if current_player.status == PlayerStatus.ALL_IN else ""
            self.last_action_desc = f"{current_player.display_name}: Колл {paid}{status_note}."
            return self._advance_turn()

        if move in (MoveType.BET, MoveType.RAISE):
            # Target bet level (player's total bet this round)
            if raise_amount is None or raise_amount <= 0:
                min_target = self.highest_bet + self.min_raise_step
                raise_amount = min_target

            # Total chips needed to reach raise_amount
            additional_needed = raise_amount - current_player.current_bet
            if additional_needed <= to_call:
                return False, f"Рейз має бути більшим за колл ({to_call})!"

            if additional_needed > current_player.chips:
                # Treat as All-In
                return self.player_move(user_id, MoveType.ALL_IN)

            raise_delta = raise_amount - self.highest_bet
            if raise_delta < self.min_raise_step and (additional_needed < current_player.chips):
                return False, f"Мінімальний крок підвищення — {self.min_raise_step}!"

            self._commit_bet(current_player, additional_needed)
            if raise_delta > self.min_raise_step:
                self.min_raise_step = raise_delta
            self.highest_bet = raise_amount

            self.last_action_desc = f"{current_player.display_name}: Підвищення до {raise_amount}."
            return self._advance_turn(after_raise=True)

        if move == MoveType.ALL_IN:
            all_in_chips = current_player.chips
            if all_in_chips <= 0:
                return False, "У вас немає фішок для олл-іну!"
            new_bet_total = current_player.current_bet + all_in_chips
            self._commit_bet(current_player, all_in_chips)

            if new_bet_total > self.highest_bet:
                diff = new_bet_total - self.highest_bet
                if diff >= self.min_raise_step:
                    self.min_raise_step = diff
                self.highest_bet = new_bet_total
                self.last_action_desc = (
                    f"{current_player.display_name}: Ва-банк (All-In {new_bet_total})!"
                )
                return self._advance_turn(after_raise=True)
            else:
                self.last_action_desc = (
                    f"{current_player.display_name}: Ва-банк (All-In {new_bet_total})!"
                )
                return self._advance_turn(after_raise=False)

        return False, "Неприпустима дія!"

    def _handle_fold(self, player: Player) -> tuple[bool, str]:
        player.status = PlayerStatus.FOLDED
        self.last_action_desc = f"{player.display_name}: Пас (Фолд)."

        active = [
            p
            for p in self.seats
            if p is not None and p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
        ]
        if len(active) <= 1:
            self._resolve_single_remaining_winner()
            return True, "Ви скинули карти."

        return self._advance_turn()

    def _advance_turn(self, after_raise: bool = False) -> tuple[bool, str]:
        """Advance turn to the next player, or progress to the next street if round complete."""
        self._cancel_turn_timer()

        active = self.get_active_in_hand_players()
        if len(active) <= 1:
            self._resolve_single_remaining_winner()
            return True, "Роздача завершена."

        acting = self.get_acting_players()
        # If 0 or 1 acting player left (everyone else all-in), run out the board!
        if len(acting) <= 1:
            # Check if current bets are settled
            if all(
                p.current_bet == self.highest_bet or p.status == PlayerStatus.ALL_IN for p in active
            ):
                self._run_out_board()
                return True, "Усі ва-банк! Відкриваємо карти."

        next_seat = self._find_next_turn_seat()
        if next_seat is not None:
            self._set_turn(next_seat)
            return True, "Хід перейшов до наступного гравця."
        else:
            # Round of betting is complete
            self._next_stage()
            return True, f"Перехід до раунду {self.stage.value}."

    def _find_next_turn_seat(self) -> int | None:
        """
        Find next seat who still needs to act in this betting round.
        Returns None if round is complete.
        """
        if self.current_turn_seat is None:
            return None

        acting = self.get_acting_players()
        if not acting:
            return None

        acting_seats = {p.seat_index for p in acting}

        # Check in circular order starting after current seat
        curr = self.current_turn_seat
        for offset in range(1, self.config.max_seats + 1):
            s = (curr + offset) % self.config.max_seats
            if s in acting_seats:
                player = self.seats[s]
                assert player
                # Needs to act if bet < highest_bet OR has not had a chance to act yet
                if player.current_bet < self.highest_bet or not getattr(
                    player, "_acted_this_round", False
                ):
                    return s

        return None

    def _set_turn(self, seat_idx: int) -> None:
        self.current_turn_seat = seat_idx
        p = self.seats[seat_idx]
        if p:
            p._acted_this_round = True
        self.turn_deadline = asyncio.get_event_loop().time() + self.config.turn_timeout_seconds
        self._start_turn_timer()

    def _start_turn_timer(self) -> None:
        self._cancel_turn_timer()
        self._turn_timer_task = asyncio.create_task(self._turn_timeout_watcher())

    def _cancel_turn_timer(self) -> None:
        if self._turn_timer_task and not self._turn_timer_task.done():
            self._turn_timer_task.cancel()
        self._turn_timer_task = None

    async def _turn_timeout_watcher(self) -> None:
        try:
            await asyncio.sleep(self.config.turn_timeout_seconds)
            # Timeout reached: Auto-check or auto-fold
            if self.current_turn_seat is not None:
                p = self.seats[self.current_turn_seat]
                if p:
                    logger.info(
                        "Player %s (%s) timed out on turn, auto-resolving",
                        p.display_name,
                        p.user_id,
                    )
                    to_call = self.highest_bet - p.current_bet
                    if to_call == 0:
                        self.player_move(p.user_id, MoveType.CHECK)
                    else:
                        self.player_move(p.user_id, MoveType.FOLD)
                    if self._on_state_change:
                        await self._on_state_change(self)
        except asyncio.CancelledError:
            pass

    # -------------------------------------------------------------------------
    # Streets Progression & Showdown
    # -------------------------------------------------------------------------
    def _next_stage(self) -> None:
        """Reset street bets and deal community cards."""
        # Reset current_bet for next round
        for p in self.seats:
            if p:
                p.current_bet = 0
                p._acted_this_round = False

        self.highest_bet = 0
        self.min_raise_step = self.config.big_blind

        if self.stage == GameStage.PREFLOP:
            self.stage = GameStage.FLOP
            self.deck.draw(1)  # Burn 1 card
            self.community_cards = self.deck.draw(3)
        elif self.stage == GameStage.FLOP:
            self.stage = GameStage.TURN
            self.deck.draw(1)  # Burn 1
            self.community_cards.append(self.deck.draw_one())
        elif self.stage == GameStage.TURN:
            self.stage = GameStage.RIVER
            self.deck.draw(1)  # Burn 1
            self.community_cards.append(self.deck.draw_one())
        elif self.stage == GameStage.RIVER:
            self._showdown()
            return

        acting = self.get_acting_players()
        if len(acting) <= 1:
            self._run_out_board()
            return

        # Street starts from first active player to the left of dealer
        first_seat = self._next_occupied_seat(self.dealer_seat, subset=acting)
        self._set_turn(first_seat)

    def _run_out_board(self) -> None:
        """Run out the remaining community cards if players are all-in, then evaluate."""
        self._cancel_turn_timer()
        self.current_turn_seat = None

        while len(self.community_cards) < 5:
            if self.deck.remaining > 1:
                self.deck.draw(1)  # Burn
                self.community_cards.append(self.deck.draw_one())
            else:
                break

        self._showdown()

    def _showdown(self) -> None:
        self.stage = GameStage.SHOWDOWN
        self._cancel_turn_timer()
        self.current_turn_seat = None

        active = self.get_active_in_hand_players()
        if not active:
            self.stage = GameStage.HAND_ENDED
            return

        # Map total bets across hand
        player_bets = {p.user_id: p.total_hand_bet for p in self.seats if p}
        active_ids = {p.user_id for p in active}
        pots = calculate_pots(player_bets, active_ids)

        # Evaluate hands for active players
        evaluations = {
            p.user_id: evaluate_hand(p.hole_cards + self.community_cards) for p in active
        }

        self.winners = []
        winner_payouts: dict[int, int] = {}

        for pot in pots:
            if not pot.eligible_player_ids:
                continue
            # Find best evaluation among eligible players for this pot
            best_eval = max(evaluations[uid] for uid in pot.eligible_player_ids)
            pot_winners = [uid for uid in pot.eligible_player_ids if evaluations[uid] == best_eval]

            # Split pot evenly, remainder to first winner
            split_amount = pot.amount // len(pot_winners)
            remainder = pot.amount % len(pot_winners)

            for i, uid in enumerate(pot_winners):
                share = split_amount + (1 if i < remainder else 0)
                winner_payouts[uid] = winner_payouts.get(uid, 0) + share

        # Payout chips to winners and create report records
        for uid, payout in winner_payouts.items():
            p = self.get_player(uid)
            if p:
                p.chips += payout
                hand_eval = evaluations[uid]
                self.winners.append(
                    HandWinner(
                        player_id=uid,
                        display_name=p.display_name,
                        amount_won=payout,
                        hand_description_uk=hand_eval.description_uk,
                        best_cards=[c.display for c in hand_eval.best_cards],
                    )
                )

        winners_summary = ", ".join(
            f"{w.display_name} (+{w.amount_won} з {w.hand_description_uk})" for w in self.winners
        )
        self.last_action_desc = f"Розкриття карт! Переможці: {winners_summary}"
        self.stage = GameStage.HAND_ENDED

    def _resolve_single_remaining_winner(self) -> None:
        """When all other players have folded, the single remaining active player wins the pot."""
        self._cancel_turn_timer()
        self.current_turn_seat = None
        self.stage = GameStage.HAND_ENDED

        active = [
            p
            for p in self.seats
            if p is not None and p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
        ]
        if not active:
            return

        winner = active[0]
        winner.chips += self.pot
        self.winners = [
            HandWinner(
                player_id=winner.user_id,
                display_name=winner.display_name,
                amount_won=self.pot,
                hand_description_uk="Останній гравець за столом (усі інші спасували)",
            )
        ]
        self.last_action_desc = (
            f"{winner.display_name} перемагає і забирає банк {self.pot} фішок (усі інші спасували)!"
        )
        self.pot = 0

    # -------------------------------------------------------------------------
    # State Serializer for Telegram & WebApp
    # -------------------------------------------------------------------------
    def to_view_dict(self, viewer_user_id: int | None = None) -> dict[str, Any]:
        """
        Serialize table state into JSON representation.
        Protects private hole cards: only the viewer sees their own cards
        unless the stage is SHOWDOWN or HAND_ENDED.
        """
        now = asyncio.get_event_loop().time()
        time_left = max(0, int(self.turn_deadline - now)) if self.turn_deadline else 0

        # Calculate current viewer's available actions
        can_check = False
        can_call = False
        can_raise = False
        call_amount = 0
        min_raise = 0
        max_raise = 0

        if self.current_turn_seat is not None:
            active_p = self.seats[self.current_turn_seat]
            if active_p and active_p.user_id == viewer_user_id:
                diff = self.highest_bet - active_p.current_bet
                can_check = diff == 0
                can_call = diff > 0 and active_p.chips > 0
                call_amount = min(diff, active_p.chips)
                can_raise = active_p.chips > diff
                min_raise = min(
                    active_p.chips + active_p.current_bet, self.highest_bet + self.min_raise_step
                )
                max_raise = active_p.chips + active_p.current_bet

        reveal_all = self.stage in (GameStage.SHOWDOWN, GameStage.HAND_ENDED)

        seats_data = []
        for p in self.seats:
            if not p:
                seats_data.append(None)
            else:
                show_cards = reveal_all or (
                    viewer_user_id is not None and p.user_id == viewer_user_id
                )
                seats_data.append(p.to_dict(include_private_cards=show_cards))

        return {
            "chat_id": self.chat_id,
            "stage": self.stage.value,
            "pot": self.pot,
            "highest_bet": self.highest_bet,
            "dealer_seat": self.dealer_seat,
            "current_turn_seat": self.current_turn_seat,
            "turn_seconds_left": time_left,
            "community_cards": [c.display for c in self.community_cards],
            "seats": seats_data,
            "last_action": self.last_action_desc,
            "winners": [
                {
                    "user_id": w.player_id,
                    "display_name": w.display_name,
                    "amount_won": w.amount_won,
                    "hand_description": w.hand_description_uk,
                    "best_cards": w.best_cards,
                }
                for w in self.winners
            ],
            "viewer_actions": {
                "is_my_turn": (
                    self.current_turn_seat is not None
                    and self.seats[self.current_turn_seat] is not None
                    and self.seats[self.current_turn_seat].user_id == viewer_user_id
                ),
                "can_check": can_check,
                "can_call": can_call,
                "can_raise": can_raise,
                "call_amount": call_amount,
                "min_raise": min_raise,
                "max_raise": max_raise,
            },
        }
