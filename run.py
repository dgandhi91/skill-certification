from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from app.core.models import (
    CertificationDecision,
    EvaluationResult,
    OverlapResult,
    SkillDefinition,
    Tier,
    ValidationResult,
)
from app.evaluation.judge import GeminiJudge
from app.evaluation.metrics import run_full_evaluation
from app.evaluation.scoring import (
    STAGE_4_WEIGHTS,
    STAGE_WEIGHTS,
    TIER_THRESHOLDS,
    compute_final_score,
    determine_tier,
)
from app.evaluation.validation import validate_skill
from app.pipeline.loader import load_workspace
from app.pipeline.registry import RegistryStore, check_overlap
from app.pipeline.results_store import PipelineResults, save_results

logger = logging.getLogger(__name__)


async def run_pipeline(
    skill_dir: str,
    workspace_dir: str,
    iteration: int = 1,
    save_cache: bool = True,
) -> None:
    logger.info("Stage 1: Loading workspace: skill=%s workspace=%s iteration=%d", skill_dir, workspace_dir, iteration)
    data = load_workspace(Path(skill_dir), Path(workspace_dir), iteration)
    logger.info("Loaded skill '%s' with %d evals, %d with_runs, %d without_runs",
                data.skill.name, len(data.dataset.evals), len(data.with_runs),
                len(data.without_runs) if data.without_runs else 0)

    store = RegistryStore()
    judge = GeminiJudge()

    try:
        logger.info("Stage 2: Contract validation")
        validation = await validate_skill(data.skill, data.dataset, judge)
        if not validation.passed:
            logger.warning("Validation FAILED")
            for flag in validation.flags:
                logger.warning("  Flag: %s", flag)
            decision = CertificationDecision(
                certified=False,
                tier=Tier.FAIL,
                reasons=[f"Validation failed: {f}" for f in validation.flags],
            )
            print_decision(decision, skill=data.skill, validation=validation)
            if save_cache:
                results = PipelineResults(validation=validation, decision=decision, skill=data.skill)
                path = save_results(data.skill.name, iteration, results)
                print(f"\nResults saved to: {path}")
            return

        logger.info("Validation PASSED")

        logger.info("Stage 3: Registry overlap check")
        overlap = await check_overlap(data.skill, store)
        logger.info("Overlap: similarity=%.4f overlap=%s", overlap.similarity_score, overlap.overlap)

        cases = data.dataset.evals

        logger.info("Stage 4: Evaluation engine (%d cases)", len(cases))
        eval_result = await run_full_evaluation(
            skill=data.skill,
            cases=cases,
            with_runs=data.with_runs,
            judge=judge,
            without_runs=data.without_runs,
        )

        if overlap.overlap:
            eval_result.flags.append(
                f"Overlaps with: {', '.join(overlap.conflicts_with)} "
                f"(score={overlap.similarity_score})"
            )

        logger.info("Stage 5: Scoring & certification decision")
        scores = compute_final_score(skill=data.skill, validation=validation, overlap=overlap, ev=eval_result)
        tier, reasons = determine_tier(eval_result, final_score=scores["final"])
        decision = CertificationDecision(
            certified=tier != Tier.FAIL,
            tier=tier,
            reasons=reasons,
            evaluation=eval_result,
        )
        logger.info("Pipeline complete: tier=%s certified=%s", tier.value, decision.certified)

        print_results(data.skill, validation, overlap, eval_result, decision)

        if save_cache:
            results = PipelineResults(
                validation=validation,
                overlap=overlap,
                evaluation=eval_result,
                decision=decision,
                skill=data.skill,
            )
            path = save_results(data.skill.name, iteration, results)
            print(f"\nResults saved to: {path}")

    finally:
        await judge.close()


def print_results(
    skill: SkillDefinition,
    validation: ValidationResult,
    overlap: OverlapResult,
    evaluation: EvaluationResult,
    decision: CertificationDecision,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Skill Certification Report: {skill.name}")
    print(f"{'='*60}")

    print(f"\n--- Validation ---")
    print(f"  Passed: {validation.passed}")
    if validation.flags:
        for f in validation.flags:
            print(f"  Flag: {f}")

    print(f"\n--- Overlap ---")
    print(f"  Overlap: {overlap.overlap}")
    print(f"  Max Similarity: {overlap.similarity_score:.4f}")
    if overlap.conflicts_with:
        print(f"  Conflicts: {', '.join(overlap.conflicts_with)}")

    print(f"\n--- Routing ---")
    r = evaluation.routing
    print(f"  Recall:             {r.recall:.3f}")
    print(f"  Precision:          {r.precision:.3f}")
    print(f"  False Trigger Rate: {r.false_trigger_rate:.3f}")

    print(f"\n--- Output Quality ---")
    oq = evaluation.output_quality
    print(f"  Answer Relevance:   {oq.answer_relevance:.3f}")
    print(f"  Faithfulness:       {oq.faithfulness:.3f}")

    if evaluation.grading:
        print(f"\n--- Assertion Grading ---")
        for i, g in enumerate(evaluation.grading):
            print(f"  Eval {i+1}: {g.summary.passed}/{g.summary.total} passed ({g.summary.pass_rate:.0%})")
            for ar in g.assertion_results:
                status = "PASS" if ar.passed else "FAIL"
                print(f"    [{status}] {ar.text}")
                if ar.evidence:
                    print(f"            {ar.evidence}")

    if evaluation.benchmark:
        b = evaluation.benchmark
        print(f"\n--- Benchmark ---")
        print(f"  {'':18s} {'Pass Rate':>10s} {'Time (s)':>10s} {'Tokens':>8s}")
        print(f"  {'With Skill':18s} {b.with_skill.pass_rate:10.2f} {b.with_skill.time_seconds:10.1f} {b.with_skill.tokens:8d}")
        print(f"  {'Without Skill':18s} {b.without_skill.pass_rate:10.2f} {b.without_skill.time_seconds:10.1f} {b.without_skill.tokens:8d}")
        print(f"  {'Delta':18s} {b.delta.pass_rate:+10.2f} {b.delta.time_seconds:+10.1f} {b.delta.tokens:+8d}")

    if evaluation.flags:
        print(f"\n--- Flags ---")
        for f in evaluation.flags:
            print(f"  {f}")

    print_decision(decision, skill=skill, validation=validation, overlap=overlap, evaluation=evaluation)


def print_decision(
    decision: CertificationDecision,
    skill: SkillDefinition | None = None,
    validation: ValidationResult | None = None,
    overlap: OverlapResult | None = None,
    evaluation: EvaluationResult | None = None,
) -> None:
    scores = compute_final_score(skill=skill, validation=validation, overlap=overlap, ev=evaluation)

    print(f"\n--- Final Score ---")
    sw = STAGE_WEIGHTS
    s4w = STAGE_4_WEIGHTS
    s4 = scores["stage_4"]

    print(f"  {'Stage':25s} {'Weight':>8s} {'Score':>8s} {'Weighted':>10s}")
    print(f"  {'1. Skill Definition':25s} {sw['stage_1']:8.2f} {scores['stage_1']['score']:8.3f} {sw['stage_1'] * scores['stage_1']['score']:10.3f}")
    print(f"  {'2. Contract Validation':25s} {sw['stage_2']:8.2f} {scores['stage_2']['score']:8.3f} {sw['stage_2'] * scores['stage_2']['score']:10.3f}")
    print(f"  {'3. Registry Overlap':25s} {sw['stage_3']:8.2f} {scores['stage_3']['score']:8.3f} {sw['stage_3'] * scores['stage_3']['score']:10.3f}")
    print(f"  {'4. Evaluation Engine':25s} {sw['stage_4']:8.2f} {s4['score']:8.3f} {sw['stage_4'] * s4['score']:10.3f}")
    print(f"    {'4.1 Routing':23s} {s4w['routing']:8.2f} {s4['routing']:8.3f} {s4w['routing'] * s4['routing']:10.3f}")
    print(f"    {'4.2 Output Quality':23s} {s4w['output_quality']:8.2f} {s4['output_quality']:8.3f} {s4w['output_quality'] * s4['output_quality']:10.3f}")
    print(f"    {'4.3 Assertion Grading':23s} {s4w['assertion_grading']:8.2f} {s4['assertion_grading']:8.3f} {s4w['assertion_grading'] * s4['assertion_grading']:10.3f}")
    print(f"    {'4.4 Benchmark':23s} {s4w['benchmark']:8.2f} {s4['benchmark']:8.3f} {s4w['benchmark'] * s4['benchmark']:10.3f}")
    print(f"  {'-'*53}")
    print(f"  {'FINAL':25s} {'1.00':>8s} {'':>8s} {scores['final']:10.3f}")

    print(f"\n{'='*60}")
    print(f"  TIER:      {decision.tier.value}")
    print(f"  CERTIFIED: {decision.certified}")
    if decision.reasons:
        print(f"\n  Reasons:")
        for r in decision.reasons:
            print(f"    - {r}")

    print(f"\n  Thresholds (out of 1.0):")
    print(f"  {'Tier':10s} {'Range':>20s}")
    thresholds_table = [
        (Tier.PREMIUM, ">= 0.9"),
        (Tier.GOLD, ">= 0.8 and < 0.9"),
        (Tier.SILVER, ">= 0.7 and < 0.8"),
        (Tier.FAIL, "< 0.7"),
    ]
    for tier, range_str in thresholds_table:
        marker = " <<<" if tier == decision.tier else ""
        print(f"  {tier.value:10s} {range_str:>20s}{marker}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run skill certification pipeline")
    parser.add_argument("skill_dir", help="Path to skill directory containing SKILL.md and evals/")
    parser.add_argument("workspace_dir", help="Path to workspace directory containing iteration-N/ dirs")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number to evaluate (default: 1)")
    parser.add_argument("--no-cache", action="store_true", help="Skip saving results to cache")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(run_pipeline(args.skill_dir, args.workspace_dir, args.iteration, save_cache=not args.no_cache))


if __name__ == "__main__":
    main()
