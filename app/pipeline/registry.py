from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.core.config import settings
from app.core.embeddings import compute_similarity, text_to_embedding
from app.core.models import (
    EvaluationResult,
    OverlapResult,
    RegistryEntry,
    SkillDefinition,
)

logger = logging.getLogger(__name__)

HookFn = Callable[[str, dict[str, Any]], None]


class RegistryStore:
    """In-memory skill registry with monitoring hooks."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._hooks: list[HookFn] = []

    def upsert(self, entry: RegistryEntry) -> None:
        version = entry.metadata.get("version", "latest")
        key = f"{entry.skill_name}:{version}"
        self._entries[key] = entry
        self.emit_hook("skill_registered", {
            "skill": entry.skill_name,
            "version": version,
        })

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


async def check_overlap(
    skill: SkillDefinition,
    store: RegistryStore,
) -> OverlapResult:
    embedding = text_to_embedding(skill.description)
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
