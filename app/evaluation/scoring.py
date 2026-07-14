from __future__ import annotations

import logging

from app.core.config import settings
from app.core.models import (
    EvaluationResult,
    OverlapResult,
    SkillDefinition,
    Tier,
    ValidationResult,
)

logger = logging.getLogger(__name__)


def _build_tier_thresholds() -> dict[Tier, dict[str, float]]:
    """Build tier thresholds from settings."""
    s = settings
    return {
        Tier.PREMIUM: {
            "recall": s.tier_premium_min_score,
            "false_trigger_rate_max": 1.0 - s.tier_premium_min_score,
            "avg_pass_rate": s.tier_premium_min_score,
            "min_final_score": s.tier_premium_min_score,
        },
        Tier.GOLD: {
            "recall": s.tier_gold_min_score,
            "false_trigger_rate_max": 1.0 - s.tier_gold_min_score,
            "avg_pass_rate": s.tier_gold_min_score,
            "min_final_score": s.tier_gold_min_score,
        },
        Tier.SILVER: {
            "recall": s.tier_silver_min_score,
            "false_trigger_rate_max": 1.0 - s.tier_silver_min_score,
            "avg_pass_rate": s.tier_silver_min_score,
            "min_final_score": s.tier_silver_min_score,
        },
    }


def _build_stage_weights() -> dict[str, float]:
    """Build stage weights from settings."""
    return {
        "stage_1": settings.weight_stage_1,
        "stage_2": settings.weight_stage_2,
        "stage_3": settings.weight_stage_3,
        "stage_4": settings.weight_stage_4,
    }


def _build_stage_4_weights() -> dict[str, float]:
    """Build stage 4 sub-weights from settings."""
    return {
        "routing": settings.weight_s4_routing,
        "output_quality": settings.weight_s4_output_quality,
        "assertion_grading": settings.weight_s4_assertion_grading,
        "benchmark": settings.weight_s4_benchmark,
    }


TIER_THRESHOLDS: dict[Tier, dict[str, float]] = _build_tier_thresholds()
STAGE_WEIGHTS: dict[str, float] = _build_stage_weights()
STAGE_4_WEIGHTS: dict[str, float] = _build_stage_4_weights()


def determine_tier(
    result: EvaluationResult, final_score: float | None = None
) -> tuple[Tier, list[str]]:
    reasons: list[str] = []

    has_gradings = bool(result.grading)
    avg_pass_rate = 1.0
    if has_gradings:
        avg_pass_rate = sum(g.summary.pass_rate for g in result.grading) / len(
            result.grading
        )

    logger.info(
        "Determining tier: recall=%.3f FTR=%.3f avg_pass_rate=%.3f final_score=%s",
        result.routing.recall,
        result.routing.false_trigger_rate,
        avg_pass_rate,
        f"{final_score:.3f}" if final_score is not None else "N/A",
    )

    for tier in [Tier.PREMIUM, Tier.GOLD, Tier.SILVER]:
        thresholds = TIER_THRESHOLDS[tier]
        meets = True

        if result.routing.recall < thresholds["recall"]:
            meets = False
            reasons.append(
                f"recall {result.routing.recall:.2f} "
                f"< {thresholds['recall']} ({tier.value})"
            )

        if result.routing.false_trigger_rate > thresholds["false_trigger_rate_max"]:
            meets = False
            reasons.append(
                f"false_trigger_rate {result.routing.false_trigger_rate:.2f} "
                f"> {thresholds['false_trigger_rate_max']} ({tier.value})"
            )

        if has_gradings and avg_pass_rate < thresholds["avg_pass_rate"]:
            meets = False
            reasons.append(
                f"avg_pass_rate {avg_pass_rate:.2f} "
                f"< {thresholds['avg_pass_rate']} ({tier.value})"
            )

        if final_score is not None and final_score < thresholds["min_final_score"]:
            meets = False
            reasons.append(
                f"final_score {final_score:.3f} "
                f"< {thresholds['min_final_score']} ({tier.value})"
            )

        if meets:
            logger.info("Tier determined: %s", tier.value)
            return tier, []
        logger.debug("Does not meet %s: %s", tier.value, reasons[-1] if reasons else "")

    logger.info("Tier determined: FAIL — %d reasons", len(reasons))
    return Tier.FAIL, reasons


def _score_stage_1(skill: SkillDefinition | None) -> dict:
    if not skill:
        return {
            "score": 0.0,
            "license": 0.0,
            "compatibility": 0.0,
            "author": 0.0,
            "version": 0.0,
        }
    fields = {
        "license": 1.0 if skill.license else 0.0,
        "compatibility": 1.0 if skill.compatibility else 0.0,
        "author": 1.0 if skill.metadata.author else 0.0,
        "version": 1.0 if skill.metadata.version else 0.0,
    }
    fields["score"] = sum(fields.values()) / len(fields)
    return fields


def _score_stage_2(validation: ValidationResult | None) -> dict:
    if not validation:
        return {"score": 0.0, "checks_passed": 0, "checks_total": 0}
    scorable_flags = [f for f in validation.flags if not f.startswith("Security risk:")]
    total = len(scorable_flags)
    failed = sum(
        1
        for f in scorable_flags
        if f.startswith("Red flag:")
        or f.startswith("Provenance:")
        or f.startswith("Permission scope:")
        or f.startswith("Prompt injection")
    )
    passed = total - failed
    score = passed / total if total > 0 else 1.0
    return {"score": score, "checks_passed": passed, "checks_total": total}


def _score_stage_3(overlap: OverlapResult | None) -> dict:
    if not overlap:
        return {"score": 1.0, "similarity": 0.0}
    return {
        "score": 1.0 - overlap.similarity_score,
        "similarity": overlap.similarity_score,
    }


def _score_stage_4(ev: EvaluationResult | None) -> dict:
    if not ev:
        return {
            "score": 0.0,
            "routing": 0.0,
            "output_quality": 0.0,
            "assertion_grading": 0.0,
            "benchmark": 0.0,
        }

    routing = (
        ev.routing.recall + ev.routing.precision + (1.0 - ev.routing.false_trigger_rate)
    ) / 3
    output_quality = (
        ev.output_quality.answer_relevance + ev.output_quality.faithfulness
    ) / 2

    if ev.grading:
        assertion_grading = sum(g.summary.pass_rate for g in ev.grading) / len(
            ev.grading
        )
    else:
        assertion_grading = 0.0

    benchmark = min(ev.benchmark.with_skill.pass_rate, 1.0) if ev.benchmark else 0.0

    w = STAGE_4_WEIGHTS
    score = (
        w["routing"] * routing
        + w["output_quality"] * output_quality
        + w["assertion_grading"] * assertion_grading
        + w["benchmark"] * benchmark
    )

    return {
        "score": score,
        "routing": routing,
        "output_quality": output_quality,
        "assertion_grading": assertion_grading,
        "benchmark": benchmark,
    }


def compute_final_score(
    skill: SkillDefinition | None = None,
    validation: ValidationResult | None = None,
    overlap: OverlapResult | None = None,
    ev: EvaluationResult | None = None,
) -> dict:
    s1 = _score_stage_1(skill)
    s2 = _score_stage_2(validation)
    s3 = _score_stage_3(overlap)
    s4 = _score_stage_4(ev)

    w = STAGE_WEIGHTS
    final = (
        w["stage_1"] * s1["score"]
        + w["stage_2"] * s2["score"]
        + w["stage_3"] * s3["score"]
        + w["stage_4"] * s4["score"]
    )

    return {
        "stage_1": s1,
        "stage_2": s2,
        "stage_3": s3,
        "stage_4": s4,
        "final": final,
    }
