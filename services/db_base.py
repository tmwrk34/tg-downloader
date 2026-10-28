"""
Shared SQLite WAL base class and database path resolution utilities.
Eliminates boilerplate across MediaCache, ChatterService, and MunitoAIService.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from pathlib import Path


def resolve_db_path(
    env_var: str,
    default_filename: str,
    test_marker: str = ".write_test",
) -> str:
    """
    Resolve the default SQLite path from an environment variable or `data/<default_filename>`,
    falling back to `./<default_filename>` if `data/` is not writable.
    """
    env_path = os.getenv(env_var)
    if env_path:
        return env_path
    data_dir = Path("data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / test_marker
        test_file.touch()
        test_file.unlink()
        return f"data/{default_filename}"
    except (PermissionError, OSError):
        return default_filename


class SQLiteWALService:
    """
    Thread-safe base class for SQLite-backed services with WAL mode,
    in-memory / URI connection handling, and deterministic cleanup.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        self._lock = threading.Lock()
        self._conn = self._create_connection()
        self._apply_wal_pragmas()

    def _create_connection(self) -> sqlite3.Connection:
        if not self._is_memory and not self.db_path.startswith("file:"):
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.startswith("file:"):
            return sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _apply_wal_pragmas(self) -> None:
        with self._lock:
            if not self._is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
                self._conn.execute("PRAGMA temp_store=MEMORY;")
                self._conn.execute("PRAGMA cache_size=-4000;")

    def _on_close(self) -> None:
        """Hook for subclasses to clear in-memory caches before closing SQLite connection."""

    def close(self) -> None:
        """Close the underlying SQLite connection and clear in-memory caches."""
        with self._lock:
            self._on_close()
            with contextlib.suppress(Exception):
                self._conn.close()
