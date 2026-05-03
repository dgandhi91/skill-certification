from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.core.models import (
    BenchmarkResult,
    EvalCase,
    EvalDataset,
    EvalRunOutput,
    GradingResult,
    SkillDefinition,
    TimingData,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceData:
    skill: SkillDefinition
    dataset: EvalDataset
    with_runs: list[EvalRunOutput]
    without_runs: list[EvalRunOutput] | None
    benchmark: BenchmarkResult | None


def load_workspace(
    skill_dir: Path,
    workspace_dir: Path,
    iteration: int = 1,
) -> WorkspaceData:
    skill = _parse_skill_md(skill_dir)
    dataset = _load_evals(skill_dir)
    slug_map = _build_slug_map(dataset)

    iteration_dir = workspace_dir / f"iteration-{iteration}"
    if not iteration_dir.is_dir():
        raise FileNotFoundError(f"Iteration directory not found: {iteration_dir}")

    with_runs = _load_all_runs(iteration_dir, slug_map, dataset, "with_skill")
    without_runs = _load_all_runs(iteration_dir, slug_map, dataset, "without_skill")

    benchmark = _load_benchmark(iteration_dir)

    return WorkspaceData(
        skill=skill,
        dataset=dataset,
        with_runs=with_runs,
        without_runs=without_runs if without_runs else None,
        benchmark=benchmark,
    )


def _parse_skill_md(skill_dir: Path) -> SkillDefinition:
    skill_path = skill_dir / "SKILL.md"
    if not skill_path.is_file():
        raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

    content = skill_path.read_text()
    parts = re.split(r"^---\s*$", content, maxsplit=2, flags=re.MULTILINE)

    if len(parts) < 3:
        raise ValueError(f"Invalid SKILL.md frontmatter in {skill_path}")

    frontmatter = yaml.safe_load(parts[1])
    if not isinstance(frontmatter, dict):
        raise ValueError(f"SKILL.md frontmatter is not a YAML mapping in {skill_path}")

    body = parts[2].strip()
    if "description" not in frontmatter and body:
        frontmatter["description"] = body[:1024]

    if "allowed-tools" in frontmatter:
        frontmatter["allowed_tools"] = frontmatter.pop("allowed-tools")

    return SkillDefinition(**frontmatter)


def _load_evals(skill_dir: Path) -> EvalDataset:
    evals_path = skill_dir / "evals" / "evals.json"
    if not evals_path.is_file():
        raise FileNotFoundError(f"evals.json not found at {evals_path}")
    return EvalDataset(**json.loads(evals_path.read_text()))


def _build_slug_map(dataset: EvalDataset) -> dict[str, int]:
    slug_map: dict[str, int] = {}
    for case in dataset.evals:
        slug = case.slug if case.slug else _derive_slug(case.prompt)
        if slug in slug_map:
            raise ValueError(
                f"Slug collision: '{slug}' maps to eval IDs "
                f"{slug_map[slug]} and {case.id}"
            )
        slug_map[slug] = case.id
    return slug_map


def _derive_slug(prompt: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s-]", "", prompt.lower())
    tokens = cleaned.split()[:5]
    return "-".join(tokens)[:64]


def _load_eval_run(run_dir: Path, eval_id: int, prompt: str) -> EvalRunOutput:
    timing = TimingData()
    timing_path = run_dir / "timing.json"
    if timing_path.is_file():
        timing = TimingData(**json.loads(timing_path.read_text()))
    else:
        logger.warning("Missing timing.json in %s", run_dir)

    grading = GradingResult()
    grading_path = run_dir / "grading.json"
    if grading_path.is_file():
        grading = GradingResult(**json.loads(grading_path.read_text()))
    else:
        logger.warning("Missing grading.json in %s", run_dir)

    actual_output = ""
    outputs_dir = run_dir / "outputs"
    if outputs_dir.is_dir():
        parts = []
        for f in sorted(outputs_dir.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md", ".csv", ".json"):
                parts.append(f.read_text())
        actual_output = "\n".join(parts)

    return EvalRunOutput(
        eval_id=eval_id,
        prompt=prompt,
        actual_output=actual_output,
        grading=grading,
        timing=timing,
    )


def _load_all_runs(
    iteration_dir: Path,
    slug_map: dict[str, int],
    dataset: EvalDataset,
    condition: str,
) -> list[EvalRunOutput]:
    case_map = {c.id: c for c in dataset.evals}
    runs: list[EvalRunOutput] = []

    for eval_dir in sorted(iteration_dir.iterdir()):
        if not eval_dir.is_dir() or not eval_dir.name.startswith("eval-"):
            continue

        slug = eval_dir.name[5:]
        eval_id = slug_map.get(slug)
        if eval_id is None:
            logger.warning(
                "No eval case matches slug '%s', skipping %s", slug, eval_dir
            )
            continue

        run_dir = eval_dir / condition
        if not run_dir.is_dir():
            continue

        case = case_map[eval_id]
        runs.append(_load_eval_run(run_dir, eval_id, case.prompt))

    return runs


def _load_benchmark(iteration_dir: Path) -> BenchmarkResult | None:
    path = iteration_dir / "benchmark.json"
    if not path.is_file():
        return None
    return BenchmarkResult(**json.loads(path.read_text()))
