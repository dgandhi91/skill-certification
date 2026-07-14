from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import settings
from app.core.embeddings import compute_similarity, text_to_embedding
from app.core.models import (
    EvaluationResult,
    OverlapResult,
    RegistryEntry,
    SkillDefinition,
)

if TYPE_CHECKING:
    from app.evaluation.providers.base import JudgeProvider

logger = logging.getLogger(__name__)

HookFn = Callable[[str, dict[str, Any]], None]


class InMemoryRegistryStore:
    """In-memory skill registry with monitoring hooks."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._hooks: list[HookFn] = []

    def upsert(self, entry: RegistryEntry) -> None:
        version = entry.metadata.get("version", "latest")
        key = f"{entry.skill_name}:{version}"
        self._entries[key] = entry
        self.emit_hook(
            "skill_registered",
            {
                "skill": entry.skill_name,
                "version": version,
            },
        )

    def get(self, name: str, version: str | None = None) -> RegistryEntry | None:
        if version:
            return self._entries.get(f"{name}:{version}")
        for entry in reversed(list(self._entries.values())):
            if entry.skill_name == name:
                return entry
        return None

    def list_entries(self) -> list[RegistryEntry]:
        return list(self._entries.values())

    def register_hook(self, hook: HookFn) -> None:
        self._hooks.append(hook)

    def emit_hook(self, event: str, data: dict[str, Any]) -> None:
        for hook in self._hooks:
            try:
                hook(event, data)
            except Exception:
                logger.exception("Hook error for event %s", event)

    def clear(self) -> None:
        self._entries.clear()


RegistryStore = InMemoryRegistryStore


class SQLiteRegistryStore:
    """Persistent skill registry backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = str(db_path or settings.registry_db_path)
        self._hooks: list[HookFn] = []
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS skills ("
                "  key TEXT PRIMARY KEY,"
                "  data TEXT NOT NULL,"
                "  created_at TEXT NOT NULL"
                ")"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def upsert(self, entry: RegistryEntry) -> None:
        version = entry.metadata.get("version", "latest")
        key = f"{entry.skill_name}:{version}"
        data = entry.model_dump_json()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO skills (key, data, created_at) VALUES (?, ?, ?)",
                (key, data, now),
            )
        logger.info("Persisted registry entry: %s", key)
        self.emit_hook(
            "skill_registered",
            {"skill": entry.skill_name, "version": version},
        )

    def get(self, name: str, version: str | None = None) -> RegistryEntry | None:
        with self._connect() as conn:
            if version:
                row = conn.execute(
                    "SELECT data FROM skills WHERE key = ?", (f"{name}:{version}",)
                ).fetchone()
                if row:
                    return RegistryEntry.model_validate_json(row[0])
                return None
            rows = conn.execute(
                "SELECT data FROM skills WHERE key LIKE ? ORDER BY created_at DESC",
                (f"{name}:%",),
            ).fetchall()
            for row in rows:
                return RegistryEntry.model_validate_json(row[0])
            return None

    def list_entries(self) -> list[RegistryEntry]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM skills ORDER BY key").fetchall()
        return [RegistryEntry.model_validate_json(row[0]) for row in rows]

    def register_hook(self, hook: HookFn) -> None:
        self._hooks.append(hook)

    def emit_hook(self, event: str, data: dict[str, Any]) -> None:
        for hook in self._hooks:
            try:
                hook(event, data)
            except Exception:
                logger.exception("Hook error for event %s", event)

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM skills")
        logger.info("Cleared all registry entries")


async def check_overlap(
    skill: SkillDefinition,
    store: InMemoryRegistryStore | SQLiteRegistryStore,
    judge: JudgeProvider | None = None,
) -> OverlapResult:
    embedding = text_to_embedding(skill.description, judge=judge)
    max_score = 0.0
    conflicts: list[str] = []

    for entry in store.list_entries():
        if not entry.embedding:
            continue
        score = compute_similarity(embedding, entry.embedding)
        if score > max_score:
            max_score = score
        if score >= settings.similarity_threshold:
            conflicts.append(entry.skill_name)

    return OverlapResult(
        overlap=len(conflicts) > 0,
        similarity_score=round(max_score, 4),
        conflicts_with=conflicts,
    )
