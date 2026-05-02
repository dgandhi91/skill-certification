from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.models import (
    CertificationDecision,
    EvaluationResult,
    OverlapResult,
    SkillDefinition,
    ValidationResult,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


@dataclass
class PipelineResults:
    validation: ValidationResult
    overlap: OverlapResult | None = None
    evaluation: EvaluationResult | None = None
    decision: CertificationDecision | None = None
    logs: list[str] | None = None
    skill: SkillDefinition | None = None


def save_results(
    skill_name: str, iteration: int, results: PipelineResults
) -> Path:
    skill_dir = RESULTS_DIR / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / f"iteration-{iteration}.json"

    data: dict = {
        "_metadata": {
            "skill_name": skill_name,
            "iteration": iteration,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "validation": results.validation.model_dump(mode="json"),
    }

    if results.overlap is not None:
        data["overlap"] = results.overlap.model_dump(mode="json")
    if results.evaluation is not None:
        data["evaluation"] = results.evaluation.model_dump(mode="json")
    if results.decision is not None:
        data["decision"] = results.decision.model_dump(mode="json")
    if results.logs is not None:
        data["logs"] = results.logs
    if results.skill is not None:
        data["skill"] = results.skill.model_dump(mode="json")

    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved results to %s", path)
    return path


def load_results(skill_name: str, iteration: int) -> PipelineResults | None:
    path = RESULTS_DIR / skill_name / f"iteration-{iteration}.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text())

    validation = ValidationResult.model_validate(data["validation"])
    overlap = (
        OverlapResult.model_validate(data["overlap"])
        if "overlap" in data
        else None
    )
    evaluation = (
        EvaluationResult.model_validate(data["evaluation"])
        if "evaluation" in data
        else None
    )
    decision = (
        CertificationDecision.model_validate(data["decision"])
        if "decision" in data
        else None
    )
    logs = data.get("logs")
    skill = (
        SkillDefinition.model_validate(data["skill"])
        if "skill" in data
        else None
    )

    logger.info("Loaded cached results from %s", path)
    return PipelineResults(
        validation=validation,
        overlap=overlap,
        evaluation=evaluation,
        decision=decision,
        logs=logs,
        skill=skill,
    )


def list_cached_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []

    entries = []
    for skill_dir in sorted(RESULTS_DIR.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        for result_file in sorted(skill_dir.glob("iteration-*.json")):
            stem = result_file.stem
            try:
                iteration = int(stem.split("-", 1)[1])
            except (IndexError, ValueError):
                continue

            data = json.loads(result_file.read_text())
            meta = data.get("_metadata", {})

            entries.append({
                "skill": skill_dir.name,
                "iteration": iteration,
                "path": result_file,
                "saved_at": meta.get("saved_at", ""),
            })
    return entries


def delete_results(skill_name: str, iteration: int) -> bool:
    path = RESULTS_DIR / skill_name / f"iteration-{iteration}.json"
    if path.exists():
        path.unlink()
        logger.info("Deleted cached results: %s", path)
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return True
    return False
