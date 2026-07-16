from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.embeddings import text_to_embedding
from app.core.models import CertificationDecision, RegistryEntry, Tier
from app.evaluation.metrics import run_full_evaluation
from app.evaluation.providers import create_judge
from app.evaluation.scoring import (
    STAGE_4_WEIGHTS,
    STAGE_WEIGHTS,
    TIER_THRESHOLDS,
    compute_final_score,
    determine_tier,
)
from app.evaluation.validation import validate_skill
from app.pipeline.loader import WorkspaceData, load_workspace
from app.pipeline.registry import check_overlap, create_registry_store
from app.pipeline.results_store import (
    PipelineResults,
    delete_results,
    list_cached_results,
    load_results,
    save_results,
)

logger = logging.getLogger(__name__)


class LogCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


async def run_pipeline(data: WorkspaceData) -> PipelineResults:
    handler = LogCaptureHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S"
        )
    )
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(handler)

    store = create_registry_store()
    judge = create_judge()

    try:
        logger.info(
            "Stage 1: Loading complete — skill '%s', %d evals, %d with_runs",
            data.skill.name,
            len(data.dataset.evals),
            len(data.with_runs),
        )

        logger.info("Stage 2: Contract validation")
        validation = await validate_skill(data.skill, data.dataset, judge)
        if not validation.passed:
            logger.warning("Validation failed with %d flags", len(validation.flags))
            return PipelineResults(
                validation=validation,
                decision=CertificationDecision(
                    certified=False,
                    tier=Tier.FAIL,
                    reasons=[f"Validation failed: {f}" for f in validation.flags],
                ),
                logs=handler.records,
                skill=data.skill,
            )

        logger.info("Validation passed")

        logger.info("Stage 3: Registry overlap check")
        overlap = await check_overlap(data.skill, store, judge=judge)
        logger.info(
            "Overlap check done — similarity=%.4f overlap=%s",
            overlap.similarity_score,
            overlap.overlap,
        )

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
        scores = compute_final_score(
            skill=data.skill, validation=validation, overlap=overlap, ev=eval_result
        )
        tier, reasons = determine_tier(eval_result, final_score=scores["final"])
        logger.info(
            "Certification result: tier=%s certified=%s", tier.value, tier != Tier.FAIL
        )
        decision = CertificationDecision(
            certified=tier != Tier.FAIL,
            tier=tier,
            reasons=reasons,
            evaluation=eval_result,
        )

        embedding = text_to_embedding(data.skill.description, judge=judge)
        entry = RegistryEntry(
            skill_name=data.skill.name,
            description=data.skill.description,
            metadata=data.skill.metadata.model_dump(),
            embedding=embedding,
            certification=decision,
            evaluation=eval_result,
        )
        store.upsert(entry)

        return PipelineResults(
            validation=validation,
            overlap=overlap,
            evaluation=eval_result,
            decision=decision,
            logs=handler.records,
            skill=data.skill,
        )
    finally:
        await judge.close()
        app_logger.removeHandler(handler)


# --- Rendering Helpers ---


def metric_color(
    value: float, thresholds: list[float], higher_is_better: bool = True
) -> str:
    if higher_is_better:
        if value >= thresholds[0]:
            return "green"
        if value >= thresholds[1]:
            return "orange"
        return "red"
    else:
        if value <= thresholds[0]:
            return "green"
        if value <= thresholds[1]:
            return "orange"
        return "red"


def render_metric(
    label: str,
    value: float,
    fmt: str = ".3f",
    thresholds: list[float] | None = None,
    higher_is_better: bool = True,
):
    color = "gray"
    if thresholds:
        color = metric_color(value, thresholds, higher_is_better)
    st.markdown(f"**{label}:** :{color}[{value:{fmt}}]")


TIER_ICONS = {
    Tier.PREMIUM: "star",
    Tier.GOLD: "circle",
    Tier.SILVER: "square",
    Tier.FAIL: "x",
}

TIER_COLORS = {
    Tier.PREMIUM: "violet",
    Tier.GOLD: "orange",
    Tier.SILVER: "gray",
    Tier.FAIL: "red",
}


def stage_badge(label: str, passed: bool) -> str:
    icon = "white_check_mark" if passed else "x"
    return f":{icon}: **{label}**"


# --- Discovery Helpers ---


SKILLS_DIR = Path("skills")


def discover_skills() -> list[str]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(str(p.parent) for p in SKILLS_DIR.rglob("SKILL.md"))


def discover_workspaces() -> list[str]:
    if not SKILLS_DIR.is_dir():
        return []
    seen: set[str] = set()
    for p in SKILLS_DIR.rglob("iteration-*"):
        if p.is_dir():
            seen.add(str(p.parent))
    return sorted(seen)


def discover_iterations(workspace: str) -> list[int]:
    ws = Path(workspace)
    if not ws.is_dir():
        return [1]
    iters = []
    for p in sorted(ws.iterdir()):
        if p.is_dir() and p.name.startswith("iteration-"):
            try:
                iters.append(int(p.name.split("-", 1)[1]))
            except ValueError:
                pass
    return iters or [1]


def page_pipeline(skill_options: list[str], workspace_options: list[str]):
    st.title("Skill Certification")

    with st.sidebar:
        st.subheader("Configuration")

        if skill_options:
            skill_dir = st.selectbox("Skill Directory", skill_options)
        else:
            skill_dir = st.text_input("Skill Directory", value="skills/csv-analyzer")

        if workspace_options:
            workspace_dir = st.selectbox("Workspace Directory", workspace_options)
        else:
            workspace_dir = st.text_input(
                "Workspace Directory", value="skills/csv-analyzer-workspace"
            )

        iteration_options = discover_iterations(workspace_dir)
        if len(iteration_options) > 1:
            iteration = st.selectbox("Iteration", iteration_options)
        else:
            iteration = iteration_options[0]

        # Derive skill name for cache lookups
        try:
            skill_name = Path(skill_dir).name
        except Exception:
            skill_name = ""

        cached = load_results(skill_name, iteration) if skill_name else None

        st.divider()

        if cached:
            st.success("Cached results available")
            col_load, col_run = st.columns(2)
            with col_load:
                load_btn = st.button("Load Cached", use_container_width=True)
            with col_run:
                run_btn = st.button("Re-run", type="primary", use_container_width=True)
            clear_btn = st.button("Clear Cache", use_container_width=True)

            if clear_btn:
                delete_results(skill_name, iteration)
                st.rerun()
        else:
            load_btn = False
            clear_btn = False
            run_btn = st.button(
                "Run Certification", type="primary", use_container_width=True
            )

    if not run_btn and not load_btn:
        st.info(
            "Configure the skill and workspace paths in the sidebar, then click **Run Certification**."
        )
        st.stop()

    # --- Load workspace data (always needed for Skill Overview) ---
    try:
        data = load_workspace(Path(skill_dir), Path(workspace_dir), iteration)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Failed to load workspace: {exc}")
        st.stop()

    # --- Run or load ---
    if load_btn and cached:
        results = cached
        st.toast("Loaded cached results")
    else:
        with st.spinner("Running certification pipeline..."):
            results = asyncio.run(run_pipeline(data))
        save_results(skill_name, iteration, results)
        st.toast(f"Results saved to results/{skill_name}/iteration-{iteration}.json")

    # --- Certification Logs in Sidebar ---
    if results.logs:
        with st.sidebar:
            with st.expander("Pipeline Logs", expanded=False):
                st.code("\n".join(results.logs), language="log")

    _render_pipeline_stages(results, data)


def _render_pipeline_stages(
    results: PipelineResults, data: WorkspaceData | None = None
):
    # ================================================================
    # Stage 1 — Skill Definition
    # ================================================================
    with st.expander("Stage 1 — Skill Definition", expanded=True):
        st.caption(
            "Loads the skill definition from SKILL.md and the eval dataset from evals/evals.json. "
            "Displays the skill metadata and a breakdown of the evaluation cases by category "
            "(positive, negative, adversarial) along with the number of with-skill and without-skill runs."
        )
        st.divider()
        if data:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Name:** `{data.skill.name}`")
                st.markdown(f"**Description:** {data.skill.description}")
                if data.skill.license:
                    st.markdown(f"**License:** {data.skill.license}")
                if data.skill.compatibility:
                    st.markdown(f"**Compatibility:** {data.skill.compatibility}")
            with col2:
                positive = sum(1 for e in data.dataset.evals if e.expected_output)
                negative = sum(
                    1
                    for e in data.dataset.evals
                    if not e.expected_output and e.category != "adversarial"
                )
                adversarial = sum(
                    1 for e in data.dataset.evals if e.category == "adversarial"
                )
                st.metric("Total Eval Cases", len(data.dataset.evals))
                c1, c2, c3 = st.columns(3)
                c1.metric("Positive", positive)
                c2.metric("Negative", negative)
                c3.metric("Adversarial", adversarial)
                st.markdown(f"**With-skill runs:** {len(data.with_runs)}")
                st.markdown(
                    f"**Without-skill runs:** {len(data.without_runs) if data.without_runs else 0}"
                )
        elif results.evaluation:
            st.markdown(f"**Name:** `{results.evaluation.skill_name}`")
        else:
            st.info("Skill definition not available from cached results.")

    # ================================================================
    # Stage 2 — Contract Validation
    # ================================================================
    with st.expander("Stage 2 — Contract Validation", expanded=True):
        st.caption(
            "Validates the skill contract against the AgentSkills specification. "
            "Checks include: ambiguity detection on skill description, "
            "security red-flag detection (network exfiltration, credential "
            "access, code injection, obfuscation, privilege escalation), author provenance "
            "verification, permission-scope analysis, and overall security risk classification "
            "(which also accounts for prompt injection patterns). "
            "If validation fails, the pipeline short-circuits and skips all subsequent stages."
        )
        st.divider()
        v = results.validation
        if v.passed:
            st.success("PASSED — All contract checks satisfied")
        else:
            st.error("FAILED — Contract validation failed")

        flag_text = " ".join(v.flags)
        no_ambiguity = "ambiguity" not in flag_text.lower()
        no_red_flags = "Red flag:" not in flag_text
        no_injection = "Prompt injection" not in flag_text
        provenance_ok = "Provenance:" not in flag_text
        permission_ok = "Permission scope:" not in flag_text

        risk_flag = next((f for f in v.flags if f.startswith("Security risk:")), None)
        risk_level = risk_flag.split(": ", 1)[1] if risk_flag else "Low"

        ambiguity_reasons = "; ".join(f for f in v.flags if "ambiguity" in f.lower())
        red_flag_reasons = "; ".join(
            f.split("Red flag: ", 1)[1] for f in v.flags if f.startswith("Red flag:")
        )
        provenance_reasons = "; ".join(
            f.split("Provenance: ", 1)[1]
            for f in v.flags
            if f.startswith("Provenance:")
        )
        injection_reasons = "; ".join(
            f.split("Prompt injection pattern detected: ", 1)[1]
            for f in v.flags
            if f.startswith("Prompt injection")
        )
        permission_reasons = "; ".join(
            f.split("Permission scope: ", 1)[1]
            for f in v.flags
            if f.startswith("Permission scope:")
        )

        checks_df = pd.DataFrame(
            [
                {
                    "Check": "Skill description clarity",
                    "Status": "PASS" if no_ambiguity else "WARN",
                    "Type": "Advisory",
                    "Reason": ambiguity_reasons or "—",
                },
                {
                    "Check": "Security risk level",
                    "Status": risk_level,
                    "Type": "Advisory",
                    "Reason": f"Based on {len([f for f in v.flags if f.startswith('Red flag:')])} red flag(s), {len([f for f in v.flags if f.startswith('Provenance:')])} provenance issue(s), {len([f for f in v.flags if f.startswith('Permission scope:')])} permission issue(s), {len([f for f in v.flags if f.startswith('Prompt injection')])} injection(s)",
                },
                {
                    "Check": "  Red flags",
                    "Status": "PASS" if no_red_flags else "FAIL",
                    "Type": "Required",
                    "Reason": red_flag_reasons or "—",
                },
                {
                    "Check": "  Author provenance",
                    "Status": "PASS" if provenance_ok else "WARN",
                    "Type": "Advisory",
                    "Reason": provenance_reasons or "—",
                },
                {
                    "Check": "  Prompt injection",
                    "Status": "PASS" if no_injection else "FAIL",
                    "Type": "Required",
                    "Reason": injection_reasons or "—",
                },
                {
                    "Check": "  Permission scope",
                    "Status": "PASS" if permission_ok else "WARN",
                    "Type": "Advisory",
                    "Reason": permission_reasons or "—",
                },
            ]
        )
        st.dataframe(checks_df, use_container_width=True, hide_index=True)

    # --- Early exit if validation failed ---
    if not results.validation.passed:
        with st.expander("Stages 3-4 — Skipped", expanded=False):
            st.warning(
                "Certification short-circuited because validation failed. Fix the issues above and re-run."
            )

        with st.expander("Stage 5 — Scoring & Certification Decision", expanded=True):
            if results.evaluation:
                tier, reasons = determine_tier(results.evaluation)
                st.error(f"**Tier:** {tier.value}")
                st.markdown(f"**Certified:** {tier != Tier.FAIL}")
                if reasons:
                    st.markdown("**Reasons:**")
                    for r in reasons:
                        st.markdown(f"- {r}")
            elif results.decision:
                st.error(f"**Tier:** {results.decision.tier.value}")
                st.markdown(f"**Certified:** {results.decision.certified}")
            else:
                st.error("**Tier:** Fail")
                st.markdown("**Certified:** False")
        return

    # ================================================================
    # Stage 3 — Registry Overlap Check
    # ================================================================
    with st.expander("Stage 3 — Registry Overlap Check", expanded=False):
        st.caption(
            "Checks whether this skill overlaps with any already-registered skill in the registry. "
            "Computes a cosine similarity between the skill's description embedding and all existing "
            "registry entries. If the similarity score exceeds the threshold (default 0.8), the skill "
            "is flagged as overlapping. Overlap does not block certification but is surfaced as a flag."
        )
        st.divider()
        o = results.overlap
        if o and o.overlap:
            st.warning(f"Overlap detected (similarity: {o.similarity_score:.4f})")
            st.markdown(f"**Conflicts with:** {', '.join(o.conflicts_with)}")
        elif o:
            st.success(
                f"No overlap detected (max similarity: {o.similarity_score:.4f})"
            )
        else:
            st.info("Overlap check not available")

    # ================================================================
    # Stage 4 — Evaluation Engine (core)
    # ================================================================
    e = results.evaluation

    with st.expander("Stage 4 — Evaluation Engine", expanded=True):
        st.caption(
            "The core evaluation engine runs the skill through multiple sub-stages: "
            "routing accuracy, output quality scoring via an LLM judge, "
            "assertion-level grading, and benchmark comparison against a no-skill baseline. "
            "Each sub-stage produces metrics that feed into the final weighted score."
        )
        st.divider()

        # 4.1 — Routing
        with st.expander("4.1 — Routing", expanded=True):
            st.caption(
                "Measures how accurately the skill activates when it should and stays silent when it shouldn't. "
                "Built from a confusion matrix (TP/FP/FN/TN) over all eval cases. "
                "**Recall** = TP/(TP+FN) — fraction of positive cases correctly triggered. "
                "**Precision** = TP/(TP+FP) — fraction of triggered cases that were correct. "
                "**False Trigger Rate** = FP/(FP+TN) — fraction of negative cases incorrectly triggered."
            )
            if e:
                r = e.routing
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_metric("Recall", r.recall, thresholds=[0.9, 0.7])
                with col2:
                    render_metric("Precision", r.precision, thresholds=[0.9, 0.7])
                with col3:
                    render_metric(
                        "False Trigger Rate",
                        r.false_trigger_rate,
                        thresholds=[0.1, 0.3],
                        higher_is_better=False,
                    )

        # 4.2 — Output Quality
        with st.expander("4.2 — Output Quality (LLM Judge)", expanded=False):
            st.caption(
                "Uses Gemini 2.5 Flash as an LLM judge to score the quality of skill outputs. "
                "Each with-skill run that produced output is scored on two dimensions: "
                "**Answer Relevance** (0-1) — how directly the output addresses the original prompt. "
                "**Faithfulness** (0-1) — how closely the output matches the expected output. "
                "Scores are averaged across all scoreable runs."
            )
            if e:
                oq = e.output_quality
                col1, col2 = st.columns(2)
                with col1:
                    render_metric(
                        "Answer Relevance", oq.answer_relevance, thresholds=[0.8, 0.5]
                    )
                with col2:
                    render_metric(
                        "Faithfulness", oq.faithfulness, thresholds=[0.8, 0.5]
                    )
                if oq.answer_relevance == 0.0 and oq.faithfulness == 0.0:
                    st.caption(
                        "Scores are 0.0 — this typically means the Gemini judge API key is not configured."
                    )

        # 4.3 — Assertion Grading
        with st.expander("4.3 — Assertion Grading", expanded=False):
            st.caption(
                "Grades each eval case's output against its defined assertions using the LLM judge. "
                "Each assertion is a natural-language statement that the output should satisfy "
                "(e.g., 'The output includes a bar chart image file'). The judge determines PASS or FAIL "
                "for each assertion and provides specific evidence from the output. "
                "The pass rate per eval is used in the Evaluation category of the final score."
            )
            if e and e.grading:
                for i, g in enumerate(e.grading):
                    s = g.summary
                    header = (
                        f"Eval {i+1}: {s.passed}/{s.total} passed ({s.pass_rate:.0%})"
                    )
                    if s.pass_rate == 1.0:
                        st.success(header)
                    elif s.pass_rate > 0:
                        st.warning(header)
                    else:
                        st.error(header)
                    for ar in g.assertion_results:
                        icon = "white_check_mark" if ar.passed else "x"
                        st.markdown(f"  :{icon}: **{ar.text}**")
                        if ar.evidence:
                            st.caption(f"    Evidence: {ar.evidence}")
                    st.divider()
            elif e:
                st.info(
                    "No assertions to grade (eval cases have no assertions defined)"
                )

        # 4.4 — Benchmark
        with st.expander(
            "4.4 — Benchmark (with_skill vs without_skill)", expanded=False
        ):
            st.caption(
                "Compares performance between with-skill and without-skill runs across three dimensions: "
                "**Pass Rate** — average assertion pass rate across graded runs. "
                "**Time** — total execution time in seconds. "
                "**Tokens** — total token usage. "
                "A positive delta in pass rate means the skill improves output quality. "
                "Higher time and tokens are expected as the skill adds processing overhead."
            )
            if e and e.benchmark:
                b = e.benchmark
                bench_df = pd.DataFrame(
                    [
                        {
                            "": "With Skill",
                            "Pass Rate": f"{b.with_skill.pass_rate:.2f}",
                            "Time (s)": f"{b.with_skill.time_seconds:.1f}",
                            "Tokens": b.with_skill.tokens,
                        },
                        {
                            "": "Without Skill",
                            "Pass Rate": f"{b.without_skill.pass_rate:.2f}",
                            "Time (s)": f"{b.without_skill.time_seconds:.1f}",
                            "Tokens": b.without_skill.tokens,
                        },
                        {
                            "": "Delta",
                            "Pass Rate": f"{b.delta.pass_rate:+.2f}",
                            "Time (s)": f"{b.delta.time_seconds:+.1f}",
                            "Tokens": f"{b.delta.tokens:+d}",
                        },
                    ]
                )
                st.dataframe(bench_df, use_container_width=True, hide_index=True)
            elif e:
                st.info("No benchmark data — without_skill runs not provided")

    # ================================================================
    # Stage 5 — Scoring & Certification Decision
    # ================================================================
    with st.expander("Stage 5 — Scoring & Certification Decision", expanded=True):
        st.caption(
            "Computes the final weighted score across all pipeline stages and assigns a certification tier. "
            "**Stage 1 — Skill Definition** (10%) — completeness of optional metadata fields. "
            "**Stage 2 — Contract Validation** (20%) — fraction of validation checks passed. "
            "**Stage 3 — Registry Overlap** (10%) — uniqueness score (1 - similarity). "
            "**Stage 4 — Evaluation Engine** (60%) — weighted average of routing, output quality, assertion grading, and benchmark. "
            "The final score determines the tier: Premium >= 0.9, Gold >= 0.8, Silver >= 0.7, Fail < 0.7."
        )
        st.divider()
        if not e:
            st.warning(
                "Certification decision not available — pipeline may have encountered an error."
            )
        else:
            skill_def = data.skill if data else None
            v = results.validation
            o = results.overlap
            scores = compute_final_score(skill=skill_def, validation=v, overlap=o, ev=e)
            sw = STAGE_WEIGHTS
            s4w = STAGE_4_WEIGHTS
            s4 = scores["stage_4"]

            tier, tier_reasons = determine_tier(e, final_score=scores["final"])
            certified = tier != Tier.FAIL
            tier_color = TIER_COLORS.get(tier, "gray")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.metric("Final Score", f"{scores['final']:.3f}")
            with col2:
                st.markdown(f"### :{tier_color}[{tier.value}]")
            with col3:
                if certified:
                    st.success("Certified")
                else:
                    st.error("Not Certified")

            st.divider()

            left, right = st.columns(2)
            with left:
                st.markdown("**Tier Thresholds** (out of 1.0)")
                thresholds_df = pd.DataFrame(
                    [
                        {"Tier": "Premium", "Score": ">= 0.9"},
                        {"Tier": "Gold", "Score": ">= 0.8 and < 0.9"},
                        {"Tier": "Silver", "Score": ">= 0.7 and < 0.8"},
                        {"Tier": "Fail", "Score": "< 0.7"},
                    ]
                )
                st.dataframe(thresholds_df, use_container_width=True, hide_index=True)
            with right:
                st.markdown("**Stage Scores**")
                stage_scores_df = pd.DataFrame(
                    {
                        "Stage": [
                            "1. Skill Definition",
                            "2. Contract Validation",
                            "3. Registry Overlap",
                            "4. Evaluation Engine",
                        ],
                        "Score": [
                            scores["stage_1"]["score"],
                            scores["stage_2"]["score"],
                            scores["stage_3"]["score"],
                            s4["score"],
                        ],
                    }
                )
                st.bar_chart(stage_scores_df.set_index("Stage"))

            st.divider()

            st.markdown("**Final Score Breakdown**")
            breakdown_rows = [
                {
                    "Stage": "1. Skill Definition",
                    "Weight": sw["stage_1"],
                    "Score": round(scores["stage_1"]["score"], 3),
                    "Weighted": round(sw["stage_1"] * scores["stage_1"]["score"], 3),
                    "Components": "License, Compatibility, Author, Version",
                },
                {
                    "Stage": "2. Contract Validation",
                    "Weight": sw["stage_2"],
                    "Score": round(scores["stage_2"]["score"], 3),
                    "Weighted": round(sw["stage_2"] * scores["stage_2"]["score"], 3),
                    "Components": f"{scores['stage_2']['checks_passed']}/{scores['stage_2']['checks_total']} checks passed",
                },
                {
                    "Stage": "3. Registry Overlap",
                    "Weight": sw["stage_3"],
                    "Score": round(scores["stage_3"]["score"], 3),
                    "Weighted": round(sw["stage_3"] * scores["stage_3"]["score"], 3),
                    "Components": f"1 - similarity ({scores['stage_3']['similarity']:.3f})",
                },
                {
                    "Stage": "4. Evaluation Engine",
                    "Weight": sw["stage_4"],
                    "Score": round(s4["score"], 3),
                    "Weighted": round(sw["stage_4"] * s4["score"], 3),
                    "Components": "Weighted avg of 4.1-4.4",
                },
                {
                    "Stage": "  4.1 Routing",
                    "Weight": s4w["routing"],
                    "Score": round(s4["routing"], 3),
                    "Weighted": round(s4w["routing"] * s4["routing"], 3),
                    "Components": "Recall, Precision, 1 - FTR",
                },
                {
                    "Stage": "  4.2 Output Quality",
                    "Weight": s4w["output_quality"],
                    "Score": round(s4["output_quality"], 3),
                    "Weighted": round(s4w["output_quality"] * s4["output_quality"], 3),
                    "Components": "Answer relevance, Faithfulness",
                },
                {
                    "Stage": "  4.3 Assertion Grading",
                    "Weight": s4w["assertion_grading"],
                    "Score": round(s4["assertion_grading"], 3),
                    "Weighted": round(
                        s4w["assertion_grading"] * s4["assertion_grading"], 3
                    ),
                    "Components": "Avg assertion pass rate",
                },
                {
                    "Stage": "  4.4 Benchmark",
                    "Weight": s4w["benchmark"],
                    "Score": round(s4["benchmark"], 3),
                    "Weighted": round(s4w["benchmark"] * s4["benchmark"], 3),
                    "Components": "With-skill pass rate",
                },
                {
                    "Stage": "Final",
                    "Weight": 1.00,
                    "Score": "",
                    "Weighted": round(scores["final"], 3),
                    "Components": "",
                },
            ]
            breakdown_df = pd.DataFrame(breakdown_rows)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            if e and not e.grading:
                st.caption(
                    "Assertion grading was skipped (no Gemini API key or no assertions). Tier is based on routing metrics only."
                )

            if tier_reasons:
                st.divider()
                st.markdown("**Reasons for tier assignment:**")
                for r in tier_reasons:
                    st.markdown(f"- {r}")

            if e and e.flags:
                st.divider()
                st.markdown("**Evaluation Flags:**")
                for f in e.flags:
                    st.warning(f)


# ============================================================
# Dashboard Page
# ============================================================


def page_dashboard():
    st.title("Skill Bench")

    cached = list_cached_results()
    if not cached:
        st.info(
            "No cached results found. Run the certification from the **Certification** page first to populate the Skill Bench."
        )
        st.stop()

    # Load all cached results
    all_results: list[dict] = []
    for entry in cached:
        res = load_results(entry["skill"], entry["iteration"])
        if res and res.evaluation:
            all_results.append(
                {
                    "skill": entry["skill"],
                    "iteration": entry["iteration"],
                    "saved_at": entry["saved_at"],
                    "results": res,
                }
            )

    if not all_results:
        st.info(
            "Cached results exist but have no evaluation data. Re-run the pipeline to generate full results."
        )
        st.stop()

    # --- Skills Summary Table ---
    st.subheader("Skills Summary")

    rows = []
    for item in all_results:
        res: PipelineResults = item["results"]
        ev = res.evaluation
        scores = compute_final_score(
            skill=res.skill, validation=res.validation, overlap=res.overlap, ev=ev
        )
        s4 = scores["stage_4"]
        tier, reasons = determine_tier(ev, final_score=scores["final"])
        certified = tier != Tier.FAIL
        rows.append(
            {
                "Skill": item["skill"],
                "Iteration": item["iteration"],
                "Final Score": round(scores["final"], 3),
                "Tier": tier.value,
                "Certified": certified,
                "1. Definition": round(scores["stage_1"]["score"], 3),
                "2. Validation": round(scores["stage_2"]["score"], 3),
                "3. Overlap": round(scores["stage_3"]["score"], 3),
                "4. Evaluation": round(s4["score"], 3),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Tier Distribution ---
    st.subheader("Tier Distribution")
    tier_counts = df["Tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    st.bar_chart(tier_counts.set_index("Tier"))

    # --- Score Comparison Charts ---
    st.subheader("Score Comparison")

    if len(df) > 1:
        skill_labels = df["Skill"] + " (iter " + df["Iteration"].astype(str) + ")"
    else:
        skill_labels = df["Skill"]

    compare_df = pd.DataFrame(
        {
            "Skill": skill_labels,
            "1. Definition": df["1. Definition"],
            "2. Validation": df["2. Validation"],
            "3. Overlap": df["3. Overlap"],
            "4. Evaluation": df["4. Evaluation"],
        }
    )
    st.bar_chart(compare_df.set_index("Skill"))

    # --- Benchmark Comparison ---
    benchmark_rows = []
    for item in all_results:
        res: PipelineResults = item["results"]
        ev = res.evaluation
        if ev and ev.benchmark:
            b = ev.benchmark
            benchmark_rows.append(
                {
                    "Skill": item["skill"],
                    "Pass Rate Delta": round(b.delta.pass_rate, 3),
                    "Time Delta (s)": round(b.delta.time_seconds, 1),
                    "Token Delta": b.delta.tokens,
                }
            )

    if benchmark_rows:
        st.subheader("Benchmark Comparison (with_skill vs without_skill)")
        bm_df = pd.DataFrame(benchmark_rows)
        st.dataframe(bm_df, use_container_width=True, hide_index=True)

    # --- Per-Skill Detail Expanders ---
    st.subheader("Per-Skill Details")

    for item in all_results:
        res: PipelineResults = item["results"]
        ev = res.evaluation
        scores = compute_final_score(
            skill=res.skill, validation=res.validation, overlap=res.overlap, ev=ev
        )
        tier, _reasons = determine_tier(ev, final_score=scores["final"])
        tier_icon = TIER_ICONS.get(tier, "question")
        label = f":{tier_icon}: {item['skill']} — iteration {item['iteration']} — {tier.value} ({scores['final']:.3f})"

        with st.expander(label, expanded=False):
            _render_pipeline_stages(res)


# ============================================================
# Main Entrypoint
# ============================================================

st.set_page_config(page_title="Skill Certification", layout="wide")

skill_options = discover_skills()
workspace_options = discover_workspaces()

with st.sidebar:
    st.header("Skill Certification")
    page = st.radio(
        "Navigate", ["Certification", "Skill Bench"], label_visibility="collapsed"
    )
    st.divider()

if page == "Certification":
    page_pipeline(skill_options, workspace_options)
elif page == "Skill Bench":
    page_dashboard()
