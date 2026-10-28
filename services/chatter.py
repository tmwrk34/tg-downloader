"""
Tavern Chatter Service — ultra-lightweight Markov chain chatterbot
with per-chat SQLite configuration and vocabulary.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from services.db_base import SQLiteWALService, resolve_db_path

logger = logging.getLogger(__name__)

# Sentence boundary markers
_START1 = "__START1__"
_START2 = "__START2__"
_END = "__END__"

# Default emoji reaction pool for chatterbot
CHATTER_REACTIONS = [
    "🤡",
    "🎲",
    "🔥",
    "👀",
    "🍺",
    "🗿",
    "⚡",
    "🫡",
    "🤔",
    "🎉",
    "🧙‍♂️",
    "🍻",
    "🍗",
    "🌭",
    "🪙",
]

# URLs, commands, or mentions to filter from training
_URL_RE = re.compile(r"https?://\S+|t\.me/\S+", re.IGNORECASE)
_COMMAND_RE = re.compile(r"^/[a-zA-Z0-9_]+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_STRIP_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")


def resolve_default_chatter_db() -> str:
    """Resolve the default SQLite path for chatterbot, falling back to local dir if data/ is not writable."""
    return resolve_db_path("CHATTER_DB", "chatter.db", ".write_test_chatter")


def clean_and_tokenize(text: str) -> list[list[str]]:
    """
    Split text into clean sentences of words for Markov chain training.
    Filters out commands, URLs, and excessive noise.
    """
    if not text or not text.strip():
        return []

    # If message is a command or contains a link, ignore completely for training
    clean_text = text.strip()
    if clean_text.startswith("/") or _URL_RE.search(clean_text):
        return []

    sentences = _SENTENCE_SPLIT_RE.split(clean_text)
    result: list[list[str]] = []

    for s in sentences:
        words = s.split()
        cleaned_words: list[str] = []
        for w in words:
            # Skip mentions and bot tags
            if w.startswith("@"):
                continue
            token = _STRIP_PUNCT_RE.sub("", w)
            if token and len(token) <= 40:
                cleaned_words.append(token)

        # Require at least 2 words per sentence for meaningful Markov transitions
        if len(cleaned_words) >= 2:
            result.append(cleaned_words)

    return result


@dataclass
class ChatterConfig:
    chat_id: int
    enabled: bool = False
    reply_chance: float = 0.05
    reaction_chance: float = 0.10
    reply_to_mention: bool = True
    cooldown_seconds: int = 30
    learning_enabled: bool = True
    last_reply_timestamp: float = 0.0


@dataclass(frozen=True)
class ChatterStats:
    chat_id: int
    total_transitions: int
    unique_words: int


class ChatterService(SQLiteWALService):
    """
    Ultra-lightweight Markov chain generator and per-chat configuration engine
    backed by SQLite WAL storage.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        resolved = str(db_path) if db_path is not None else resolve_default_chatter_db()
        super().__init__(resolved)
        self._init_db()

        # Cache chat configs in memory for hot-path lookups: chat_id -> ChatterConfig
        self._config_cache: dict[int, ChatterConfig] = {}

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chatter_settings (
                    chat_id INTEGER PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    reply_chance REAL NOT NULL DEFAULT 0.05,
                    reaction_chance REAL NOT NULL DEFAULT 0.10,
                    reply_to_mention INTEGER NOT NULL DEFAULT 1,
                    cooldown_seconds INTEGER NOT NULL DEFAULT 30,
                    learning_enabled INTEGER NOT NULL DEFAULT 1,
                    last_reply_timestamp REAL NOT NULL DEFAULT 0.0
                )
                """
            )

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chatter_markov (
                    chat_id INTEGER NOT NULL,
                    w1 TEXT NOT NULL,
                    w2 TEXT NOT NULL,
                    next_word TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (chat_id, w1, w2, next_word)
                )
                """
            )

            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chatter_lookup ON chatter_markov(chat_id, w1, w2);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chatter_seed ON chatter_markov(chat_id, w1);"
            )
            self._conn.commit()

    def _on_close(self) -> None:
        self._config_cache.clear()

    def _sync_get_config(self, chat_id: int) -> ChatterConfig:
        with self._lock:
            if chat_id in self._config_cache:
                return self._config_cache[chat_id]

            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT enabled, reply_chance, reaction_chance, reply_to_mention,
                       cooldown_seconds, learning_enabled, last_reply_timestamp
                FROM chatter_settings
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
            row = cursor.fetchone()
            if row:
                cfg = ChatterConfig(
                    chat_id=chat_id,
                    enabled=bool(row[0]),
                    reply_chance=float(row[1]),
                    reaction_chance=float(row[2]),
                    reply_to_mention=bool(row[3]),
                    cooldown_seconds=int(row[4]),
                    learning_enabled=bool(row[5]),
                    last_reply_timestamp=float(row[6]),
                )
            else:
                cfg = ChatterConfig(chat_id=chat_id)
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO chatter_settings (
                        chat_id, enabled, reply_chance, reaction_chance,
                        reply_to_mention, cooldown_seconds, learning_enabled, last_reply_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        int(cfg.enabled),
                        cfg.reply_chance,
                        cfg.reaction_chance,
                        int(cfg.reply_to_mention),
                        cfg.cooldown_seconds,
                        int(cfg.learning_enabled),
                        cfg.last_reply_timestamp,
                    ),
                )
                self._conn.commit()

            self._config_cache[chat_id] = cfg
            return cfg

    def _sync_update_config(self, chat_id: int, **kwargs: Any) -> ChatterConfig:
        base_cfg = self._sync_get_config(chat_id)
        with self._lock:
            valid_updates = {k: v for k, v in kwargs.items() if hasattr(base_cfg, k)}
            cfg = replace(base_cfg, **valid_updates)

            self._conn.execute(
                """
                INSERT OR REPLACE INTO chatter_settings (
                    chat_id, enabled, reply_chance, reaction_chance,
                    reply_to_mention, cooldown_seconds, learning_enabled, last_reply_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    int(cfg.enabled),
                    cfg.reply_chance,
                    cfg.reaction_chance,
                    int(cfg.reply_to_mention),
                    cfg.cooldown_seconds,
                    int(cfg.learning_enabled),
                    cfg.last_reply_timestamp,
                ),
            )
            self._conn.commit()
            self._config_cache[chat_id] = cfg
            return cfg

    def _sync_train(self, chat_id: int, sentences: list[list[str]]) -> int:
        if not sentences:
            return 0

        transitions_added = 0
        with self._lock:
            cursor = self._conn.cursor()
            for sentence in sentences:
                seq = [_START1, _START2] + sentence + [_END]
                for i in range(len(seq) - 2):
                    w1 = seq[i]
                    w2 = seq[i + 1]
                    next_word = seq[i + 2]
                    cursor.execute(
                        """
                        INSERT INTO chatter_markov (chat_id, w1, w2, next_word, count)
                        VALUES (?, ?, ?, ?, 1)
                        ON CONFLICT(chat_id, w1, w2, next_word)
                        DO UPDATE SET count = count + 1
                        """,
                        (chat_id, w1, w2, next_word),
                    )
                    transitions_added += 1
            self._conn.commit()

        return transitions_added

    def _sync_generate(
        self,
        chat_id: int,
        seed_words: list[str] | None = None,
        max_words: int = 35,
    ) -> str | None:
        with self._lock:
            cursor = self._conn.cursor()

            w1, w2 = _START1, _START2
            # Try to pick a seed prefix matching any candidate word from user message
            if seed_words:
                candidates = [w for w in seed_words if w and w not in {_START1, _START2, _END}]
                random.shuffle(candidates)
                for cand in candidates:
                    cursor.execute(
                        """
                        SELECT w1, w2 FROM chatter_markov
                        WHERE chat_id = ? AND (w1 = ? OR w2 = ?) AND w1 != ?
                        ORDER BY count DESC LIMIT 5
                        """,
                        (chat_id, cand, cand, _START1),
                    )
                    rows = cursor.fetchall()
                    if rows:
                        chosen = random.choice(rows)
                        w1, w2 = chosen[0], chosen[1]
                        break

            generated_words: list[str] = []
            if w1 not in {_START1, _START2, _END}:
                generated_words.append(w1)
            if w2 not in {_START1, _START2, _END}:
                generated_words.append(w2)

            visited_triples: set[tuple[str, str, str]] = set()

            while len(generated_words) < max_words:
                cursor.execute(
                    """
                    SELECT next_word, count FROM chatter_markov
                    WHERE chat_id = ? AND w1 = ? AND w2 = ?
                    """,
                    (chat_id, w1, w2),
                )
                choices = cursor.fetchall()
                if not choices:
                    break

                words = [c[0] for c in choices]
                weights = [c[1] for c in choices]
                next_word = random.choices(words, weights=weights, k=1)[0]

                if next_word == _END:
                    # If sentence is too short (< 3 words) and we have other paths, don't end prematurely
                    if len(generated_words) < 3 and len(choices) > 1:
                        non_end = [c for c in choices if c[0] != _END]
                        if non_end:
                            next_word = random.choices(
                                [c[0] for c in non_end],
                                weights=[c[1] for c in non_end],
                                k=1,
                            )[0]
                        else:
                            break
                    else:
                        break

                triple = (w1, w2, next_word)
                if triple in visited_triples:
                    # Break on cycle to avoid repetitive loops
                    break
                visited_triples.add(triple)

                generated_words.append(next_word)
                w1, w2 = w2, next_word

            if not generated_words:
                return None

            sentence = " ".join(generated_words).strip()
            # Capitalize first letter
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                if sentence[-1] not in ".!?":
                    sentence += "."

            return sentence

    def _sync_clear_vocabulary(self, chat_id: int) -> int:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM chatter_markov WHERE chat_id = ?", (chat_id,))
            self._conn.commit()
            return cursor.rowcount

    def _sync_get_stats(self, chat_id: int) -> ChatterStats:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT SUM(count) FROM chatter_markov WHERE chat_id = ?",
                (chat_id,),
            )
            count_row = cursor.fetchone()
            total_transitions = int(count_row[0]) if count_row and count_row[0] else 0

            cursor.execute(
                """
                SELECT COUNT(DISTINCT next_word) FROM chatter_markov
                WHERE chat_id = ? AND next_word != ?
                """,
                (chat_id, _END),
            )
            words_row = cursor.fetchone()
            unique_words = int(words_row[0]) if words_row and words_row[0] else 0

            return ChatterStats(
                chat_id=chat_id,
                total_transitions=total_transitions,
                unique_words=unique_words,
            )

    # ---------------- Async API Methods ----------------

    async def get_config(self, chat_id: int) -> ChatterConfig:
        return await asyncio.to_thread(self._sync_get_config, chat_id)

    async def update_config(self, chat_id: int, **kwargs: Any) -> ChatterConfig:
        return await asyncio.to_thread(self._sync_update_config, chat_id, **kwargs)

    async def record_reply(self, chat_id: int, timestamp: float | None = None) -> None:
        ts = time.time() if timestamp is None else timestamp
        await self.update_config(chat_id, last_reply_timestamp=ts)

    async def train(self, chat_id: int, text: str) -> int:
        cfg = await self.get_config(chat_id)
        if not cfg.learning_enabled:
            return 0
        sentences = clean_and_tokenize(text)
        if not sentences:
            return 0
        return await asyncio.to_thread(self._sync_train, chat_id, sentences)

    async def generate(
        self,
        chat_id: int,
        seed_text: str | None = None,
        max_words: int = 35,
    ) -> str | None:
        seed_words: list[str] = []
        if seed_text:
            cleaned_sents = clean_and_tokenize(seed_text)
            for s in cleaned_sents:
                seed_words.extend(s)
        return await asyncio.to_thread(self._sync_generate, chat_id, seed_words, max_words)

    async def clear_vocabulary(self, chat_id: int) -> int:
        return await asyncio.to_thread(self._sync_clear_vocabulary, chat_id)

    async def get_stats(self, chat_id: int) -> ChatterStats:
        return await asyncio.to_thread(self._sync_get_stats, chat_id)

    def should_reply(
        self,
        config: ChatterConfig,
        is_mention_or_reply: bool,
        current_time: float | None = None,
    ) -> bool:
        """
        Determine whether the chatterbot should respond to a message.
        """
        if not config.enabled:
            return False

        now = time.time() if current_time is None else current_time

        # Direct mention or reply to the bot bypasses cooldown if reply_to_mention is active
        if is_mention_or_reply:
            return config.reply_to_mention

        # Check cooldown
        if (now - config.last_reply_timestamp) < config.cooldown_seconds:
            return False

        # Random chance
        return random.random() < config.reply_chance

    def should_react(self, config: ChatterConfig) -> bool:
        """
        Determine whether the bot should set an emoji reaction to the message.
        """
        if not config.enabled or config.reaction_chance <= 0:
            return False
        return random.random() < config.reaction_chance

    @staticmethod
    def pick_reaction() -> str:
        """Pick a random reaction emoji from the tavern pool."""
        return random.choice(CHATTER_REACTIONS)
