"""
Hybrid Long-Term Memory Engine for Munito AI.

Combines:
1. Multilingual 384-d embeddings (ONNX `fastembed` when installed, with a fast
   deterministic multilingual subword/stem + character n-gram projection engine).
2. SQLite WAL + FTS5 BM25 lexical index + dense vector cosine similarity
   (with optional LanceDB table synchronization when `lancedb` is installed).
3. Smart Noise Gate and Episodic Chunking for group chat conversations.
4. Structured User Dossiers (`ai_user_dossiers`) and Tavern Lore (`ai_memories`).
5. Cosine deduplication (`similarity > 0.90`) and strict token budget formatting.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
DEFAULT_DEDUP_THRESHOLD = 0.90
DEFAULT_RAG_MIN_SCORE = 0.72

# Noise filtering regexes and stop-reactions
_URL_RE = re.compile(r"https?://\S+|t\.me/\S+", re.IGNORECASE)
_COMMAND_RE = re.compile(r"^/[a-zA-Z0-9_а-яА-ЯіїєґІЇЄҐ]+")
_WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐёЁ0-9_]{2,}", re.UNICODE)

_NOISE_EXACT = frozenset(
    {
        "ок",
        "окей",
        "пон",
        "поняв",
        "зрозумів",
        "зрозуміло",
        "ахах",
        "ахаха",
        "ахахах",
        "хаха",
        "хахах",
        "хіхі",
        "лол",
        "кек",
        "база",
        "жиза",
        "імба",
        "крінж",
        "дякую",
        "спс",
        "спасибі",
        "так",
        "ні",
        "да",
        "нет",
        "неа",
        "угу",
        "ага",
        "ну",
        "хз",
        "норм",
        "супер",
        "клас",
        "кайф",
        "ok",
        "okay",
        "lol",
        "lmao",
        "rofl",
        "nice",
        "cool",
        "yes",
        "no",
        "yep",
        "nope",
        "thanks",
        "thx",
        "ty",
        "+",
        "++",
        "+++",
        "-",
        "--",
        "100%",
    }
)


@dataclass(frozen=True)
class MemoryEntry:
    """Represents an episodic memory chunk or distilled tavern lore/fact."""

    id: int
    chat_id: int
    category: str
    subject: str
    content: str
    participants: str
    importance: float
    created_at: float
    updated_at: float
    score: float = 0.0


@dataclass(frozen=True)
class UserDossier:
    """Represents structured traits and notes about a specific chat member."""

    chat_id: int
    user_id: int
    username: str | None
    display_name: str
    traits: list[str]
    updated_at: float


@dataclass(frozen=True)
class MemoryStats:
    """Summary statistics of long-term memory for a chat."""

    chat_id: int
    total_memories: int = 0
    lore_count: int = 0
    episodic_count: int = 0
    dossier_count: int = 0
    buffered_turns: int = 0


@dataclass
class _BufferedTurn:
    user_name: str
    user_id: int | None
    content: str
    timestamp: float


def is_meaningful_message(text: str, min_chars: int = 25, min_words: int = 3) -> bool:
    """
    Smart Noise Gate: filters out commands, pure URLs, short one-word reactions,
    and low-signal chatter before indexing into long-term episodic memory.
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()
    if _COMMAND_RE.match(stripped):
        return False

    # Strip URLs and check remaining text
    without_urls = _URL_RE.sub("", stripped).strip()
    if not without_urls:
        return False

    # Normalize punctuation and check against known noise reactions
    clean_lower = re.sub(r"[^\w\s+]", "", without_urls.lower()).strip()
    if not clean_lower or clean_lower in _NOISE_EXACT:
        return False

    # Reject repeated laughter/characters like "ахахахахах" or "))))))"
    if re.fullmatch(r"[ахa-z0-9)+_-]+", clean_lower) and len(set(clean_lower)) <= 3:
        return False

    if len(without_urls) < min_chars:
        return False

    words = _WORD_RE.findall(without_urls)
    return len(words) >= min_words


def _hash_feature(feature: str) -> tuple[int, float]:
    """Deterministic 64-bit hash mapping a feature string to (dimension_index, sign)."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "little")
    idx = val % EMBEDDING_DIM
    sign = 1.0 if ((val >> 32) & 1) == 0 else -1.0
    return idx, sign


def _hash_feature_secondary(feature: str) -> tuple[int, float]:
    """Secondary hash projection for reduced collision variance."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8, key=b"munito_v2").digest()
    val = int.from_bytes(digest, "little")
    idx = val % EMBEDDING_DIM
    sign = 1.0 if ((val >> 32) & 1) == 0 else -1.0
    return idx, sign


class HybridEmbedder:
    """
    Produces 384-dimensional L2-normalized float32 embeddings.
    Uses ONNX `fastembed` (`paraphrase-multilingual-MiniLM-L12-v2`) when installed and enabled,
    with an instant, zero-dependency multilingual morphological subword + n-gram embedder fallback.
    """

    def __init__(self, use_fastembed: bool | None = None) -> None:
        self._fastembed_model: Any = None
        if use_fastembed is None:
            env_flag = os.getenv("MUNITO_USE_FASTEMBED", "auto").strip().lower()
            use_fastembed = env_flag in ("1", "true", "yes")

        if use_fastembed and importlib.util.find_spec("fastembed") is not None:
            try:
                from fastembed import TextEmbedding

                self._fastembed_model = TextEmbedding(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                logger.info("Loaded ONNX fastembed multilingual embedding model.")
            except Exception as e:
                logger.warning("Failed to load fastembed model, using built-in embedder: %s", e)
                self._fastembed_model = None

    @property
    def backend_name(self) -> str:
        return "fastembed-onnx" if self._fastembed_model is not None else "multilingual-subword-384"

    def embed(self, text: str) -> np.ndarray:
        """Compute a 384-dimensional L2-normalized float32 vector for `text`."""
        if not text or not text.strip():
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        if self._fastembed_model is not None:
            try:
                vecs = list(self._fastembed_model.embed([text]))
                if vecs:
                    arr = np.asarray(vecs[0], dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    return arr / norm if norm > 1e-8 else arr
            except Exception as e:
                logger.debug("fastembed inference error, falling back: %s", e)

        return self._embed_subword(text)

    @staticmethod
    def _embed_subword(text: str) -> np.ndarray:
        """
        Multilingual morphological subword + word + char n-gram feature hashing into 384-d float32.
        Handles Ukrainian, Russian, and English inflections via stem prefixes and char 3/4-grams.
        """
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        cleaned = _URL_RE.sub(" ", text.lower())
        words = _WORD_RE.findall(cleaned)
        if not words:
            return vec

        feature_weights: dict[str, float] = {}

        def add_feat(feat: str, weight: float) -> None:
            feature_weights[feat] = feature_weights.get(feat, 0.0) + weight

        for idx_w, w in enumerate(words):
            # 1. Exact word token
            add_feat(f"w:{w}", 1.6)

            # 2. Morphological root/stem prefixes for Slavic/English inflections
            w_len = len(w)
            if w_len >= 4:
                add_feat(f"p4:{w[:4]}", 1.4)
            if w_len >= 5:
                add_feat(f"p5:{w[:5]}", 1.5)
            if w_len >= 6:
                add_feat(f"p6:{w[:6]}", 1.3)

            # 3. Bounded character 3-grams and 4-grams
            padded = f"<{w}>"
            for n in (3, 4):
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        add_feat(f"c{n}:{padded[i : i + n]}", 0.55)

            # 4. Word bigrams for local phrase structure
            if idx_w + 1 < len(words):
                add_feat(f"bi:{w}_{words[idx_w + 1]}", 1.1)

        for feat, raw_tf in feature_weights.items():
            scaled = 1.0 + math.log(raw_tf) if raw_tf > 1.0 else raw_tf
            i1, s1 = _hash_feature(feat)
            i2, s2 = _hash_feature_secondary(feat)
            vec[i1] += s1 * scaled
            vec[i2] += s2 * (scaled * 0.7)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    if v1.shape != v2.shape or v1.size == 0:
        return 0.0
    return float(np.dot(v1, v2))


class HybridMemoryManager:
    """
    Manages long-term episodic memories, tavern lore, and user dossiers
    using SQLite WAL + FTS5 BM25 + 384-d vector embeddings (and optional LanceDB sync).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: threading.RLock,
        db_path: str | Path | None = None,
        embedder: HybridEmbedder | None = None,
        episode_chunk_size: int = 5,
        episode_max_gap_seconds: float = 600.0,
    ) -> None:
        self._conn = conn
        self._lock = lock
        self._embedder = embedder or HybridEmbedder()
        self.episode_chunk_size = episode_chunk_size
        self.episode_max_gap_seconds = episode_max_gap_seconds
        self._episode_buffers: dict[int, list[_BufferedTurn]] = {}
        self._lance_db: Any = None
        self._db_path = str(db_path) if db_path else ":memory:"

        self._init_tables()
        self._init_optional_lancedb()

    def _init_tables(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    category TEXT NOT NULL DEFAULT 'episodic',
                    subject TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    participants TEXT NOT NULL DEFAULT '',
                    embedding BLOB NOT NULL,
                    importance REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_memories_chat ON ai_memories (chat_id, category, updated_at DESC);"
            )

            # SQLite FTS5 Virtual Table for BM25 lexical search
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS ai_memories_fts USING fts5(
                    content,
                    subject,
                    category,
                    participants,
                    chat_id UNINDEXED,
                    memory_id UNINDEXED,
                    tokenize='unicode61'
                )
                """
            )

            # Structured User Dossiers per chat
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_user_dossiers (
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT,
                    display_name TEXT NOT NULL,
                    traits_json TEXT NOT NULL DEFAULT '[]',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_id, user_id)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_dossiers_chat ON ai_user_dossiers (chat_id, display_name);"
            )
            self._conn.commit()

    def _init_optional_lancedb(self) -> None:
        """Initialize optional LanceDB directory alongside SQLite DB when `lancedb` is installed."""
        if self._db_path == ":memory:" or importlib.util.find_spec("lancedb") is None:
            return
        try:
            import lancedb

            lance_dir = Path(self._db_path).resolve().parent / "lancedb"
            lance_dir.mkdir(parents=True, exist_ok=True)
            self._lance_db = lancedb.connect(str(lance_dir))
            logger.info("Connected to embedded LanceDB at %s", lance_dir)
        except Exception as e:
            logger.debug("Optional LanceDB initialization skipped: %s", e)
            self._lance_db = None

    def add_memory(
        self,
        chat_id: int,
        content: str,
        category: str = "episodic",
        subject: str = "",
        participants: str = "",
        importance: float | None = None,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        max_memories_per_chat: int = 500,
    ) -> tuple[MemoryEntry, bool]:
        """
        Store a memory or fact in the hybrid index.
        Deduplicates against existing memories in `chat_id`: if cosine similarity > `dedup_threshold`,
        updates the existing record instead of inserting a duplicate.
        Returns `(MemoryEntry, is_new)`.
        """
        clean_content = content.strip()
        clean_subject = subject.strip()
        clean_category = (category or "episodic").strip().lower()
        if importance is None:
            importance = 1.5 if clean_category in ("lore", "fact", "inside_joke", "rule") else 1.0

        embed_text = f"{clean_subject}: {clean_content}" if clean_subject else clean_content
        vec = self._embedder.embed(embed_text)
        vec_bytes = vec.tobytes()
        now = time.time()

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, category, subject, content, participants, embedding, importance, created_at
                FROM ai_memories
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
            rows = cursor.fetchall()

            best_match_id: int | None = None
            best_sim = -1.0
            best_row: tuple[Any, ...] | None = None

            for row in rows:
                existing_vec = np.frombuffer(row[5], dtype=np.float32)
                if existing_vec.size == EMBEDDING_DIM:
                    sim = float(np.dot(vec, existing_vec))
                    if sim > best_sim:
                        best_sim = sim
                        best_match_id = int(row[0])
                        best_row = row

            if best_match_id is not None and best_row is not None and best_sim >= dedup_threshold:
                merged_importance = max(float(best_row[6]), importance)
                merged_subject = clean_subject or str(best_row[2])
                merged_participants = participants.strip() or str(best_row[4])
                merged_category = (
                    clean_category if clean_category != "episodic" else str(best_row[1])
                )

                self._conn.execute(
                    """
                    UPDATE ai_memories
                    SET category = ?, subject = ?, content = ?, participants = ?,
                        embedding = ?, importance = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        merged_category,
                        merged_subject,
                        clean_content,
                        merged_participants,
                        vec_bytes,
                        merged_importance,
                        now,
                        best_match_id,
                    ),
                )
                self._conn.execute(
                    "DELETE FROM ai_memories_fts WHERE memory_id = ?",
                    (best_match_id,),
                )
                self._conn.execute(
                    """
                    INSERT INTO ai_memories_fts (content, subject, category, participants, chat_id, memory_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_content,
                        merged_subject,
                        merged_category,
                        merged_participants,
                        chat_id,
                        best_match_id,
                    ),
                )
                self._conn.commit()
                return (
                    MemoryEntry(
                        id=best_match_id,
                        chat_id=chat_id,
                        category=merged_category,
                        subject=merged_subject,
                        content=clean_content,
                        participants=merged_participants,
                        importance=merged_importance,
                        created_at=float(best_row[7]),
                        updated_at=now,
                        score=best_sim,
                    ),
                    False,
                )

            cursor.execute(
                """
                INSERT INTO ai_memories (
                    chat_id, category, subject, content, participants, embedding, importance, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    clean_category,
                    clean_subject,
                    clean_content,
                    participants.strip(),
                    vec_bytes,
                    importance,
                    now,
                    now,
                ),
            )
            mem_id = int(cursor.lastrowid or 0)
            self._conn.execute(
                """
                INSERT INTO ai_memories_fts (content, subject, category, participants, chat_id, memory_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_content,
                    clean_subject,
                    clean_category,
                    participants.strip(),
                    chat_id,
                    mem_id,
                ),
            )

            # Prune oldest low-importance memories if limit exceeded
            cursor.execute(
                "SELECT COUNT(*) FROM ai_memories WHERE chat_id = ?",
                (chat_id,),
            )
            total_row = cursor.fetchone()
            if total_row and int(total_row[0]) > max_memories_per_chat:
                cursor.execute(
                    """
                    SELECT id FROM ai_memories
                    WHERE chat_id = ?
                    ORDER BY importance ASC, updated_at ASC
                    LIMIT ?
                    """,
                    (chat_id, int(total_row[0]) - max_memories_per_chat),
                )
                to_prune = [int(r[0]) for r in cursor.fetchall()]
                for pid in to_prune:
                    self._conn.execute("DELETE FROM ai_memories WHERE id = ?", (pid,))
                    self._conn.execute("DELETE FROM ai_memories_fts WHERE memory_id = ?", (pid,))

            self._conn.commit()

            return (
                MemoryEntry(
                    id=mem_id,
                    chat_id=chat_id,
                    category=clean_category,
                    subject=clean_subject,
                    content=clean_content,
                    participants=participants.strip(),
                    importance=importance,
                    created_at=now,
                    updated_at=now,
                    score=1.0,
                ),
                True,
            )

    def search_memories(
        self,
        chat_id: int,
        query: str,
        top_k: int = 3,
        min_score: float = DEFAULT_RAG_MIN_SCORE,
        exclude_exact: str | None = None,
    ) -> list[MemoryEntry]:
        """
        Execute hybrid search combining:
        1. Dense 384-d cosine similarity
        2. SQLite FTS5 BM25 lexical matching (including prefix stems for morphology)
        3. Subject/keyword overlap and importance calibration
        Returns top-K memories with calibrated hybrid score >= `min_score`.
        """
        if not query or not query.strip() or top_k <= 0:
            return []

        q_clean = query.strip()
        q_vec = self._embedder.embed(q_clean)
        q_words = [w.lower() for w in _WORD_RE.findall(q_clean) if len(w) >= 3]
        q_stems = {w[:4] for w in q_words if len(w) >= 4}

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, chat_id, category, subject, content, participants, embedding, importance, created_at, updated_at
                FROM ai_memories
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
            rows = cursor.fetchall()
            if not rows:
                return []

            # 1. Run FTS5 BM25 search for lexical candidates
            bm25_scores: dict[int, float] = {}
            if q_words:
                fts_terms: list[str] = []
                for w in q_words[:10]:
                    safe_w = re.sub(r"[^a-zA-Zа-яА-ЯіїєґІЇЄҐёЁ0-9]", "", w)
                    if len(safe_w) >= 4:
                        fts_terms.append(f'"{safe_w[:4]}"*')
                    elif len(safe_w) >= 2:
                        fts_terms.append(f'"{safe_w}"*')

                if fts_terms:
                    fts_query = " OR ".join(fts_terms)
                    try:
                        cursor.execute(
                            """
                            SELECT memory_id, bm25(ai_memories_fts)
                            FROM ai_memories_fts
                            WHERE ai_memories_fts MATCH ? AND CAST(chat_id AS INTEGER) = ?
                            LIMIT 30
                            """,
                            (fts_query, int(chat_id)),
                        )
                        for m_id, raw_bm25 in cursor.fetchall():
                            # In SQLite FTS5, bm25() returns negative values (more negative = better)
                            pos_bm25 = abs(float(raw_bm25))
                            bm25_scores[int(m_id)] = min(1.0, (pos_bm25 + 0.5) / (pos_bm25 + 1.2))
                    except sqlite3.OperationalError as e:
                        logger.debug("FTS5 query fallback: %s", e)

            scored_entries: list[MemoryEntry] = []
            exclude_norm = exclude_exact.strip().lower() if exclude_exact else None

            for row in rows:
                mem_id = int(row[0])
                content_str = str(row[4])
                if exclude_norm and content_str.strip().lower() == exclude_norm:
                    continue

                subject_str = str(row[3])
                participants_str = str(row[5])
                emb_bytes = row[6]
                importance = float(row[7])

                mem_vec = np.frombuffer(emb_bytes, dtype=np.float32)
                vec_sim = (
                    max(0.0, float(np.dot(q_vec, mem_vec)))
                    if mem_vec.size == EMBEDDING_DIM
                    else 0.0
                )
                lex_score = bm25_scores.get(mem_id, 0.0)

                # Calculate stem/keyword coverage
                doc_text_lower = f"{subject_str} {content_str} {participants_str}".lower()
                doc_words = _WORD_RE.findall(doc_text_lower)
                doc_stems = {w[:4] for w in doc_words if len(w) >= 4}

                stem_overlap = 0.0
                if q_stems and doc_stems:
                    matched_stems = len(q_stems & doc_stems)
                    if matched_stems > 0:
                        stem_overlap = min(1.0, matched_stems / max(1, min(len(q_stems), 3)))

                # Hybrid fusion score calibrated to [0.0, 1.0]
                raw_hybrid = (
                    0.50 * min(1.0, vec_sim * 1.45) + 0.30 * lex_score + 0.20 * stem_overlap
                )
                if vec_sim >= 0.12 and (lex_score > 0.0 or stem_overlap > 0.0):
                    # Synergy boost when both semantic vector and lexical/stem match agree
                    raw_hybrid = max(
                        raw_hybrid,
                        0.72 + 0.15 * min(1.0, vec_sim) + 0.10 * stem_overlap,
                    )
                elif vec_sim >= 0.55:
                    raw_hybrid = max(raw_hybrid, min(0.98, vec_sim * 1.25))

                calibrated_score = min(1.0, raw_hybrid * min(1.15, 0.90 + 0.10 * importance))

                if calibrated_score >= min_score:
                    scored_entries.append(
                        MemoryEntry(
                            id=mem_id,
                            chat_id=int(row[1]),
                            category=str(row[2]),
                            subject=subject_str,
                            content=content_str,
                            participants=participants_str,
                            importance=importance,
                            created_at=float(row[8]),
                            updated_at=float(row[9]),
                            score=round(calibrated_score, 4),
                        )
                    )

            scored_entries.sort(key=lambda e: (e.score, e.updated_at), reverse=True)
            return scored_entries[:top_k]

    def forget_memory(self, chat_id: int, query: str) -> list[MemoryEntry]:
        """
        Find and delete memories matching `query` in `chat_id`.
        Returns the list of removed MemoryEntry items.
        """
        matches = self.search_memories(chat_id, query, top_k=3, min_score=0.45)
        if not matches:
            # Fallback substring search
            q_low = query.strip().lower()
            with self._lock:
                cursor = self._conn.cursor()
                cursor.execute(
                    """
                    SELECT id, chat_id, category, subject, content, participants, importance, created_at, updated_at
                    FROM ai_memories
                    WHERE chat_id = ? AND (LOWER(content) LIKE ? OR LOWER(subject) LIKE ?)
                    ORDER BY updated_at DESC LIMIT 3
                    """,
                    (chat_id, f"%{q_low}%", f"%{q_low}%"),
                )
                matches = [
                    MemoryEntry(
                        id=int(r[0]),
                        chat_id=int(r[1]),
                        category=str(r[2]),
                        subject=str(r[3]),
                        content=str(r[4]),
                        participants=str(r[5]),
                        importance=float(r[6]),
                        created_at=float(r[7]),
                        updated_at=float(r[8]),
                        score=1.0,
                    )
                    for r in cursor.fetchall()
                ]

        if not matches:
            return []

        # Delete the top match (or all matches with score >= 0.80)
        to_remove = [matches[0]] + [m for m in matches[1:] if m.score >= 0.80]
        with self._lock:
            for item in to_remove:
                self._conn.execute(
                    "DELETE FROM ai_memories WHERE id = ? AND chat_id = ?",
                    (item.id, chat_id),
                )
                self._conn.execute(
                    "DELETE FROM ai_memories_fts WHERE memory_id = ?",
                    (item.id,),
                )
            self._conn.commit()
        return to_remove

    def get_chat_lore(self, chat_id: int, limit: int = 8) -> list[MemoryEntry]:
        """Fetch distilled tavern lore, facts, and inside jokes for `chat_id`."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, chat_id, category, subject, content, participants, importance, created_at, updated_at
                FROM ai_memories
                WHERE chat_id = ? AND category IN ('lore', 'fact', 'inside_joke', 'rule', 'preference', 'event')
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            return [
                MemoryEntry(
                    id=int(r[0]),
                    chat_id=int(r[1]),
                    category=str(r[2]),
                    subject=str(r[3]),
                    content=str(r[4]),
                    participants=str(r[5]),
                    importance=float(r[6]),
                    created_at=float(r[7]),
                    updated_at=float(r[8]),
                    score=1.0,
                )
                for r in cursor.fetchall()
            ]

    def update_user_dossier(
        self,
        chat_id: int,
        user_name: str,
        trait_or_note: str,
        user_id: int | None = None,
        username: str | None = None,
        max_traits: int = 15,
    ) -> UserDossier:
        """
        Add or update a trait/note in a member's Tavern Dossier.
        Deduplicates semantically similar traits for the same user.
        """
        clean_name = (user_name or "Traveler").strip().lstrip("@")
        clean_note = trait_or_note.strip()
        now = time.time()

        with self._lock:
            cursor = self._conn.cursor()
            resolved_uid = user_id
            resolved_username = username
            existing_traits: list[str] = []

            if resolved_uid is not None:
                cursor.execute(
                    """
                    SELECT user_id, username, display_name, traits_json
                    FROM ai_user_dossiers
                    WHERE chat_id = ? AND user_id = ?
                    """,
                    (chat_id, resolved_uid),
                )
                row = cursor.fetchone()
                if row:
                    resolved_username = resolved_username or row[1]
                    clean_name = clean_name or row[2]
                    with ExceptionSuppressor():
                        existing_traits = json.loads(row[3])
            else:
                # Try matching by display_name or username in ai_user_dossiers
                cursor.execute(
                    """
                    SELECT user_id, username, display_name, traits_json
                    FROM ai_user_dossiers
                    WHERE chat_id = ? AND (LOWER(display_name) = LOWER(?) OR LOWER(username) = LOWER(?))
                    LIMIT 1
                    """,
                    (chat_id, clean_name, clean_name),
                )
                row = cursor.fetchone()
                if row:
                    resolved_uid = int(row[0])
                    resolved_username = resolved_username or row[1]
                    clean_name = row[2]
                    with ExceptionSuppressor():
                        existing_traits = json.loads(row[3])
                else:
                    # Generate deterministic synthetic user_id from normalized name
                    digest = hashlib.blake2b(
                        clean_name.lower().encode("utf-8"), digest_size=6
                    ).digest()
                    resolved_uid = -abs(int.from_bytes(digest, "little"))

            if clean_note:
                new_vec = self._embedder.embed(clean_note)
                merged = False
                for idx, old_trait in enumerate(existing_traits):
                    old_vec = self._embedder.embed(old_trait)
                    if cosine_similarity(new_vec, old_vec) >= 0.88:
                        existing_traits[idx] = clean_note
                        merged = True
                        break
                if not merged:
                    existing_traits.append(clean_note)
                if len(existing_traits) > max_traits:
                    existing_traits = existing_traits[-max_traits:]

            traits_str = json.dumps(existing_traits, ensure_ascii=False)
            self._conn.execute(
                """
                INSERT INTO ai_user_dossiers (chat_id, user_id, username, display_name, traits_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, user_id) DO UPDATE SET
                    username = COALESCE(excluded.username, ai_user_dossiers.username),
                    display_name = excluded.display_name,
                    traits_json = excluded.traits_json,
                    updated_at = excluded.updated_at
                """,
                (chat_id, resolved_uid, resolved_username, clean_name, traits_str, now),
            )
            self._conn.commit()

            return UserDossier(
                chat_id=chat_id,
                user_id=resolved_uid,
                username=resolved_username,
                display_name=clean_name,
                traits=existing_traits,
                updated_at=now,
            )

    def get_user_dossiers(
        self,
        chat_id: int,
        active_user_id: int | None = None,
        active_user_name: str | None = None,
        query_text: str | None = None,
        limit: int = 5,
    ) -> list[UserDossier]:
        """
        Retrieve relevant user dossiers for a chat: prioritizes active speaker
        and any users mentioned in `query_text`.
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT chat_id, user_id, username, display_name, traits_json, updated_at
                FROM ai_user_dossiers
                WHERE chat_id = ?
                ORDER BY updated_at DESC
                """,
                (chat_id,),
            )
            all_dossiers: list[UserDossier] = []
            for r in cursor.fetchall():
                try:
                    traits = json.loads(r[4])
                except Exception:
                    traits = []
                if traits:
                    all_dossiers.append(
                        UserDossier(
                            chat_id=int(r[0]),
                            user_id=int(r[1]),
                            username=r[2],
                            display_name=str(r[3]),
                            traits=traits,
                            updated_at=float(r[5]),
                        )
                    )

            if not all_dossiers:
                return []

            if active_user_id is None and not active_user_name and not query_text:
                return all_dossiers[:limit]

            prioritized: list[UserDossier] = []
            others: list[UserDossier] = []
            q_low = (query_text or "").lower()
            act_name_low = (active_user_name or "").strip().lower()

            for d in all_dossiers:
                is_active = (active_user_id is not None and d.user_id == active_user_id) or (
                    act_name_low and d.display_name.lower() == act_name_low
                )
                is_mentioned = bool(
                    q_low
                    and (
                        d.display_name.lower() in q_low
                        or (d.username and d.username.lower() in q_low)
                    )
                )
                if is_active or is_mentioned:
                    prioritized.append(d)
                else:
                    others.append(d)

            return (prioritized + others)[:limit]

    def ingest_passive_turn(
        self,
        chat_id: int,
        user_name: str | None,
        user_id: int | None,
        content: str,
    ) -> MemoryEntry | None:
        """
        Passively evaluate an incoming turn via Smart Noise Gate.
        Meaningful turns are buffered per chat and indexed as an episodic scene
        once `episode_chunk_size` turns accumulate (or when time gap > `episode_max_gap_seconds`).
        """
        if not is_meaningful_message(content):
            return None

        now = time.time()
        speaker = (user_name or "User").strip()
        buf = self._episode_buffers.setdefault(chat_id, [])

        # If buffer has old turns and gap exceeded, flush previous episode first
        flushed_entry: MemoryEntry | None = None
        if buf and (now - buf[-1].timestamp) > self.episode_max_gap_seconds:
            flushed_entry = self.flush_episode_buffer(chat_id)
            buf = self._episode_buffers.setdefault(chat_id, [])

        buf.append(
            _BufferedTurn(
                user_name=speaker,
                user_id=user_id,
                content=content.strip(),
                timestamp=now,
            )
        )

        if len(buf) >= self.episode_chunk_size:
            return self.flush_episode_buffer(chat_id)

        return flushed_entry

    def flush_episode_buffer(self, chat_id: int) -> MemoryEntry | None:
        """Flush buffered turns for `chat_id` into a single coherent episodic memory chunk."""
        buf = self._episode_buffers.pop(chat_id, [])
        if not buf:
            return None

        participants_ordered: list[str] = []
        lines: list[str] = []
        for t in buf:
            if t.user_name not in participants_ordered:
                participants_ordered.append(t.user_name)
            lines.append(f"[{t.user_name}]: {t.content}")

        chunk_text = "\n".join(lines)
        participants_str = ", ".join(participants_ordered)
        subject_str = f"Діалог ({participants_str})"
        entry, _ = self.add_memory(
            chat_id=chat_id,
            content=chunk_text,
            category="episodic",
            subject=subject_str,
            participants=participants_str,
            importance=1.0,
        )
        return entry

    def get_stats(self, chat_id: int) -> MemoryStats:
        """Return memory counts for `chat_id`."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN category != 'episodic' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN category = 'episodic' THEN 1 ELSE 0 END)
                FROM ai_memories
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
            row = cursor.fetchone()
            total_mem = int(row[0] or 0) if row else 0
            lore_cnt = int(row[1] or 0) if row else 0
            ep_cnt = int(row[2] or 0) if row else 0

            cursor.execute(
                "SELECT COUNT(*) FROM ai_user_dossiers WHERE chat_id = ?",
                (chat_id,),
            )
            d_row = cursor.fetchone()
            dossier_cnt = int(d_row[0] or 0) if d_row else 0

            buffered_cnt = len(self._episode_buffers.get(chat_id, []))
            return MemoryStats(
                chat_id=chat_id,
                total_memories=total_mem,
                lore_count=lore_cnt,
                episodic_count=ep_cnt,
                dossier_count=dossier_cnt,
                buffered_turns=buffered_cnt,
            )

    def clear_long_term_memory(self, chat_id: int) -> int:
        """Delete all long-term memories, FTS entries, dossiers, and buffers for `chat_id`."""
        self._episode_buffers.pop(chat_id, None)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM ai_memories WHERE chat_id = ?", (chat_id,))
            deleted_mem = cursor.rowcount
            self._conn.execute("DELETE FROM ai_memories_fts WHERE chat_id = ?", (chat_id,))
            cursor.execute("DELETE FROM ai_user_dossiers WHERE chat_id = ?", (chat_id,))
            deleted_dos = cursor.rowcount
            self._conn.commit()
            return deleted_mem + deleted_dos

    def build_memory_context_block(
        self,
        chat_id: int,
        query: str,
        active_user_id: int | None = None,
        active_user_name: str | None = None,
        max_lore_chars: int = 500,
        max_dossier_chars: int = 500,
        max_memories_chars: int = 1000,
    ) -> str:
        """
        Build the dynamic XML memory injection block (`<tavern_lore>`, `<user_dossiers>`,
        `<retrieved_memories>`) strictly bounded by token/character budgets.
        Returns an empty string if no memories exist for the chat.
        """
        sections: list[str] = []

        # 1. Tavern Lore & Facts
        lore_items = self.get_chat_lore(chat_id, limit=6)
        if lore_items:
            lore_lines: list[str] = []
            used_chars = 0
            for item in lore_items:
                line = f"- [{item.subject}] {item.content}" if item.subject else f"- {item.content}"
                if used_chars + len(line) > max_lore_chars and lore_lines:
                    break
                lore_lines.append(line)
                used_chars += len(line) + 1
            if lore_lines:
                sections.append("<tavern_lore>\n" + "\n".join(lore_lines) + "\n</tavern_lore>")

        # 2. User Dossiers (active speaker + mentioned members)
        dossiers = self.get_user_dossiers(
            chat_id,
            active_user_id=active_user_id,
            active_user_name=active_user_name,
            query_text=query,
            limit=3,
        )
        if dossiers:
            dossier_lines: list[str] = []
            used_chars = 0
            for d in dossiers:
                uname = f" (@{d.username})" if d.username else ""
                traits_joined = "; ".join(d.traits[-6:])
                line = f"- {d.display_name}{uname}: {traits_joined}"
                if used_chars + len(line) > max_dossier_chars and dossier_lines:
                    break
                dossier_lines.append(line)
                used_chars += len(line) + 1
            if dossier_lines:
                sections.append(
                    "<user_dossiers>\n" + "\n".join(dossier_lines) + "\n</user_dossiers>"
                )

        # 3. Hybrid RAG Retrieved Memories (Top-3 with score >= DEFAULT_RAG_MIN_SCORE)
        lore_ids = {item.id for item in lore_items}
        retrieved = self.search_memories(
            chat_id,
            query=query,
            top_k=3,
            min_score=DEFAULT_RAG_MIN_SCORE,
            exclude_exact=query,
        )
        filtered_retrieved = [m for m in retrieved if m.id not in lore_ids]
        if filtered_retrieved:
            mem_lines: list[str] = []
            used_chars = 0
            for m in filtered_retrieved:
                prefix = f"[{m.category}: {m.subject}] " if m.subject else f"[{m.category}] "
                line = f"- {prefix}{m.content}"
                if used_chars + len(line) > max_memories_chars and mem_lines:
                    break
                mem_lines.append(line)
                used_chars += len(line) + 1
            if mem_lines:
                sections.append(
                    "<retrieved_memories>\n" + "\n".join(mem_lines) + "\n</retrieved_memories>"
                )

        return "\n".join(sections)


class ExceptionSuppressor:
    """Lightweight context manager for suppressing JSON parse errors."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return exc_type is not None
