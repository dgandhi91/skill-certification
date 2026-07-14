from __future__ import annotations

import logging

from app.core.embeddings import text_to_embedding
from app.core.models import (
    CertificationDecision,
    EvalCase,
    EvalDataset,
    EvalRunOutput,
    RegistryEntry,
    SkillDefinition,
    Tier,
)
from app.evaluation.metrics import run_full_evaluation
from app.evaluation.providers import JudgeProvider, create_judge
from app.evaluation.scoring import compute_final_score, determine_tier
from app.evaluation.validation import validate_skill
from app.pipeline.registry import RegistryStore, check_overlap

logger = logging.getLogger(__name__)


class CertificationPipeline:
    def __init__(self, store: RegistryStore, judge: JudgeProvider):
        self.store = store
        self.judge = judge

    async def run(
        self,
        skill: SkillDefinition,
        dataset: EvalDataset,
        with_runs: list[EvalRunOutput],
        without_runs: list[EvalRunOutput] | None = None,
        validation_cases: list[EvalCase] | None = None,
        validation_runs: list[EvalRunOutput] | None = None,
    ) -> CertificationDecision:
        logger.info("Starting certification pipeline for '%s'", skill.name)

        logger.info("Stage: Contract validation")
        validation = await validate_skill(skill, dataset, self.judge)
        if not validation.passed:
            logger.warning("Validation failed: %s", validation.flags)
            return CertificationDecision(
                certified=False,
                tier=Tier.FAIL,
                reasons=[f"Validation failed: {f}" for f in validation.flags],
            )

        logger.info("Stage: Registry overlap check")
        overlap = await check_overlap(skill, self.store, judge=self.judge)
        logger.info(
            "Overlap result: overlap=%s score=%.4f",
            overlap.overlap,
            overlap.similarity_score,
        )

        cases = dataset.evals

        logger.info(
            "Stage: Evaluation engine (%d cases, %d runs)", len(cases), len(with_runs)
        )
        eval_result = await run_full_evaluation(
            skill=skill,
            cases=cases,
            with_runs=with_runs,
            judge=self.judge,
            without_runs=without_runs,
        )

        if overlap.overlap:
            eval_result.flags.append(
                f"Overlaps with: {', '.join(overlap.conflicts_with)} "
                f"(score={overlap.similarity_score})"
            )

        logger.info("Stage: Tier determination")
        scores = compute_final_score(
            skill=skill, validation=validation, overlap=overlap, ev=eval_result
        )
        tier, reasons = determine_tier(eval_result, final_score=scores["final"])
        certified = tier != Tier.FAIL
        logger.info("Result: tier=%s certified=%s", tier.value, certified)

        decision = CertificationDecision(
            certified=certified,
            tier=tier,
            reasons=reasons,
            evaluation=eval_result,
        )

        embedding = text_to_embedding(skill.description, judge=self.judge)
        entry = RegistryEntry(
            skill_name=skill.name,
            description=skill.description,
            metadata=skill.metadata.model_dump(),
            embedding=embedding,
            certification=decision,
            evaluation=eval_result,
        )
        logger.info("Registering skill '%s' in registry", skill.name)
        self.store.upsert(entry)
        self.store.emit_hook(
            "certification_complete",
            {
                "skill": skill.name,
                "tier": tier.value,
                "certified": certified,
            },
        )

        return decision
