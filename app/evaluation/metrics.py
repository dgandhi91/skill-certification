from __future__ import annotations

import logging

from app.core.models import (
    AssertionResult,
    BenchmarkResult,
    EvalCase,
    EvalRunOutput,
    EvaluationResult,
    GradingResult,
    GradingSummary,
    OutputQualityMetrics,
    RoutingMetrics,
    RunStats,
    SkillDefinition,
)
from app.evaluation.judge import GeminiJudge

logger = logging.getLogger(__name__)


def _build_confusion_matrix(
    with_runs: list[EvalRunOutput], cases: list[EvalCase]
) -> tuple[int, int, int, int]:
    case_map = {c.id: c for c in cases}
    tp = fp = fn = tn = 0
    for run in with_runs:
        case = case_map.get(run.eval_id)
        if not case:
            continue
        should_trigger = bool(case.expected_output)
        did_trigger = bool(run.actual_output)
        if should_trigger and did_trigger:
            tp += 1
        elif not should_trigger and did_trigger:
            fp += 1
        elif should_trigger and not did_trigger:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def compute_routing(
    with_runs: list[EvalRunOutput], cases: list[EvalCase]
) -> RoutingMetrics:
    if not cases:
        return RoutingMetrics()

    tp, fp, fn, tn = _build_confusion_matrix(with_runs, cases)
    total_positive = tp + fn
    total_negative = tn + fp

    metrics = RoutingMetrics(
        recall=tp / total_positive if total_positive else 1.0,
        precision=tp / (tp + fp) if (tp + fp) else 1.0,
        false_trigger_rate=fp / total_negative if total_negative else 0.0,
    )
    logger.info(
        "Routing: recall=%.3f precision=%.3f FTR=%.3f (TP=%d FP=%d FN=%d TN=%d)",
        metrics.recall,
        metrics.precision,
        metrics.false_trigger_rate,
        tp,
        fp,
        fn,
        tn,
    )
    return metrics


async def compute_output_quality(
    with_runs: list[EvalRunOutput],
    cases: list[EvalCase],
    judge: GeminiJudge,
) -> OutputQualityMetrics:
    case_map = {c.id: c for c in cases}
    scored = [
        (r, case_map[r.eval_id])
        for r in with_runs
        if r.actual_output
        and r.eval_id in case_map
        and case_map[r.eval_id].expected_output
    ]
    if not scored:
        logger.info("Output quality: no scoreable runs, skipping")
        return OutputQualityMetrics()

    logger.info("Scoring output quality for %d runs via LLM judge", len(scored))

    relevance_total = 0.0
    faithfulness_total = 0.0

    for run, case in scored:
        rel = await judge.score_relevance(case.prompt, run.actual_output)
        faith = await judge.score_faithfulness(case.expected_output, run.actual_output)
        relevance_total += rel.get("score", 0.0)
        faithfulness_total += faith.get("score", 0.0)

    n = len(scored)
    oq = OutputQualityMetrics(
        answer_relevance=relevance_total / n,
        faithfulness=faithfulness_total / n,
    )
    logger.info(
        "Output quality: relevance=%.3f faithfulness=%.3f",
        oq.answer_relevance,
        oq.faithfulness,
    )
    return oq


async def grade_eval_run(
    run: EvalRunOutput,
    case: EvalCase,
    judge: GeminiJudge,
) -> GradingResult:
    results: list[AssertionResult] = []

    for assertion_text in case.assertions:
        judgement = await judge.grade_assertion(
            prompt=case.prompt,
            expected_output=case.expected_output,
            actual_output=run.actual_output,
            assertion=assertion_text,
        )
        results.append(
            AssertionResult(
                text=assertion_text,
                passed=judgement.get("passed", False),
                evidence=judgement.get("evidence", ""),
            )
        )

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    logger.info("Graded eval %d: %d/%d assertions passed", run.eval_id, passed, total)
    return GradingResult(
        assertion_results=results,
        summary=GradingSummary(
            passed=passed,
            failed=total - passed,
            total=total,
            pass_rate=passed / total if total else 0.0,
        ),
    )


BASELINE_PASS_RATE = 0.70


def compute_benchmark(
    with_runs: list[EvalRunOutput],
    without_runs: list[EvalRunOutput],
    with_gradings: list[GradingResult],
    without_gradings: list[GradingResult],
) -> BenchmarkResult:
    def stats(runs: list[EvalRunOutput], gradings: list[GradingResult]) -> RunStats:
        pr = (
            sum(g.summary.pass_rate for g in gradings) / len(gradings)
            if gradings
            else 0.0
        )
        total_time = sum(r.timing.duration_ms for r in runs) / 1000.0
        total_tokens = sum(r.timing.total_tokens for r in runs)
        return RunStats(pass_rate=pr, time_seconds=total_time, tokens=total_tokens)

    ws = stats(with_runs, with_gradings)
    wos = stats(without_runs, without_gradings)
    wos.pass_rate = max(wos.pass_rate, BASELINE_PASS_RATE)

    return BenchmarkResult(
        with_skill=ws,
        without_skill=wos,
        delta=RunStats(
            pass_rate=ws.pass_rate - wos.pass_rate,
            time_seconds=ws.time_seconds - wos.time_seconds,
            tokens=ws.tokens - wos.tokens,
        ),
    )


async def run_full_evaluation(
    skill: SkillDefinition,
    cases: list[EvalCase],
    with_runs: list[EvalRunOutput],
    judge: GeminiJudge,
    without_runs: list[EvalRunOutput] | None = None,
) -> EvaluationResult:
    logger.info(
        "Running full evaluation for skill '%s' (%d cases, %d runs)",
        skill.name,
        len(cases),
        len(with_runs),
    )

    logger.info("Computing routing metrics")
    routing = compute_routing(with_runs, cases)

    logger.info("Computing output quality metrics")
    output_quality = await compute_output_quality(with_runs, cases, judge)

    case_map = {c.id: c for c in cases}
    gradings: list[GradingResult] = []
    gradable = [
        (r, case_map[r.eval_id])
        for r in with_runs
        if r.eval_id in case_map and case_map[r.eval_id].assertions
    ]
    if gradable:
        logger.info("Grading %d with_skill runs with assertions", len(gradable))
    for run, case in gradable:
        g = await grade_eval_run(run, case, judge)
        gradings.append(g)

    without_gradings: list[GradingResult] = []
    if without_runs:
        logger.info("Grading %d without_skill runs", len(without_runs))
        for run in without_runs:
            case = case_map.get(run.eval_id)
            if case and case.assertions:
                g = await grade_eval_run(run, case, judge)
                without_gradings.append(g)

    benchmark = None
    if without_runs:
        logger.info("Computing benchmark (with_skill vs without_skill)")
        benchmark = compute_benchmark(
            with_runs, without_runs, gradings, without_gradings
        )

    logger.info("Full evaluation complete for '%s'", skill.name)
    return EvaluationResult(
        skill_name=skill.name,
        routing=routing,
        output_quality=output_quality,
        grading=gradings,
        benchmark=benchmark,
    )
