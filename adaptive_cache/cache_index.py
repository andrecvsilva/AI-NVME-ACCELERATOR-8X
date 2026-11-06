"""Small SQLite index for content-addressed NVMe cache blobs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class CacheIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize()
        except sqlite3.DatabaseError:
            damaged = self.path.with_name(self.path.name + ".corrupt")
            suffix = 1
            while damaged.exists():
                damaged = self.path.with_name(self.path.name + f".corrupt-{suffix}")
                suffix += 1
            if self.path.exists():
                self.path.replace(damaged)
            for sidecar in (Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
                if sidecar.exists() and sidecar.is_file():
                    sidecar.unlink()
            self._initialize()
            self.recovered_from = str(damaged)
        else:
            self.recovered_from = None

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=2.0)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            connection.close()
            raise
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            check = connection.execute("PRAGMA quick_check").fetchone()
            if check and check[0] != "ok":
                raise sqlite3.DatabaseError(str(check[0]))
            connection.execute("""CREATE TABLE IF NOT EXISTS entries (
                key_hash TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                frequency INTEGER NOT NULL DEFAULT 0
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS entries_expiry ON entries(expires_at)")

    def get(self, key_hash: str, now: float):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT content_hash,size_bytes,expires_at,frequency FROM entries WHERE key_hash=?", (key_hash,)
            ).fetchone()
            if row is None:
                return None
            if row[2] <= now:
                connection.execute("DELETE FROM entries WHERE key_hash=?", (key_hash,))
                return {"expired": True, "content_hash": row[0], "size_bytes": row[1]}
            connection.execute("UPDATE entries SET accessed_at=?,frequency=frequency+1 WHERE key_hash=?", (now, key_hash))
            return {"content_hash": row[0], "size_bytes": row[1], "frequency": row[3] + 1}

    def put(self, key_hash: str, content_hash: str, size: int, now: float, expires_at: float) -> None:
        with self._connection() as connection:
            connection.execute("""INSERT INTO entries(key_hash,content_hash,size_bytes,created_at,accessed_at,expires_at,frequency)
                VALUES(?,?,?,?,?,?,0) ON CONFLICT(key_hash) DO UPDATE SET
                content_hash=excluded.content_hash,size_bytes=excluded.size_bytes,accessed_at=excluded.accessed_at,
                expires_at=excluded.expires_at""", (key_hash, content_hash, size, now, now, expires_at))

    def remove(self, key_hash: str):
        with self._connection() as connection:
            row = connection.execute("SELECT content_hash,size_bytes FROM entries WHERE key_hash=?", (key_hash,)).fetchone()
            connection.execute("DELETE FROM entries WHERE key_hash=?", (key_hash,))
            return row

    def remove_expired(self, now: float) -> list[tuple[str, int]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT content_hash,size_bytes FROM entries WHERE expires_at<=?", (now,)).fetchall()
            connection.execute("DELETE FROM entries WHERE expires_at<=?", (now,))
            return rows

    def eviction_candidates(self, policy: str) -> list[tuple[str, str, int]]:
        order = {"LRU": "accessed_at ASC", "LFU": "frequency ASC, accessed_at ASC",
                 "HYBRID": "(frequency * 1.0 / (1.0 + MAX(0, CAST(strftime('%s','now') AS REAL) - accessed_at))) ASC, accessed_at ASC"}[policy.upper()]
        with self._connection() as connection:
            return connection.execute(f"SELECT key_hash,content_hash,size_bytes FROM entries ORDER BY {order}").fetchall()

    def total_size(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COALESCE(SUM(size_bytes),0) FROM (SELECT content_hash,MAX(size_bytes) AS size_bytes FROM entries GROUP BY content_hash)").fetchone()[0])

    def count(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def content_referenced(self, content_hash: str) -> bool:
        with self._connection() as connection:
            return connection.execute("SELECT 1 FROM entries WHERE content_hash=? LIMIT 1", (content_hash,)).fetchone() is not None

    def frequency(self, key_hash: str) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT frequency FROM entries WHERE key_hash=?", (key_hash,)).fetchone()
            return int(row[0]) if row else 0
