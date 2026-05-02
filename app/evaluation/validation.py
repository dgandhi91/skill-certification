from __future__ import annotations

import logging

from app.core.models import EvalDataset, SkillDefinition, ValidationResult
from app.core.security import (
    check_permission_scope,
    check_provenance,
    classify_risk,
    detect_ambiguity,
    scan_for_injection,
    scan_for_red_flags,
)
from app.evaluation.judge import GeminiJudge

logger = logging.getLogger(__name__)


async def validate_skill(skill: SkillDefinition, dataset: EvalDataset, judge: GeminiJudge) -> ValidationResult:
    flags: list[str] = []

    logger.info(
        "Validating skill '%s': %d evals",
        skill.name, len(dataset.evals),
    )

    non_adversarial = [e for e in dataset.evals if e.category != "adversarial"]
    texts_to_scan = [skill.description]
    for ev in non_adversarial:
        texts_to_scan.append(ev.prompt)
        if ev.expected_output:
            texts_to_scan.append(ev.expected_output)

    ambiguity_flags = await detect_ambiguity(skill.description, judge)
    flags.extend(ambiguity_flags)

    logger.info("Scanning %d texts for red flags via LLM", len(texts_to_scan))
    red_flag_findings = await scan_for_red_flags(texts_to_scan, judge)
    flags.extend(red_flag_findings)

    provenance_findings = check_provenance(skill.metadata.author)
    flags.extend(provenance_findings)

    logger.info("Scanning %d texts for dangerous tools via LLM", len(texts_to_scan))
    permission_findings = await check_permission_scope(texts_to_scan, judge)
    flags.extend(permission_findings)

    logger.info("Scanning %d texts for injection patterns via LLM", len(texts_to_scan))
    injection_findings = await scan_for_injection(texts_to_scan, judge)
    flags.extend(injection_findings)

    risk_level = classify_risk(
        red_flag_count=len(red_flag_findings),
        provenance_count=len(provenance_findings),
        permission_count=len(permission_findings),
        injection_count=len(injection_findings),
    )
    flags.append(f"Security risk: {risk_level}")

    passed = not any(
        f.startswith("Red flag:")
        for f in flags
    )

    if passed:
        logger.info("Validation PASSED for '%s' (%d informational flags)", skill.name, len(flags))
    else:
        logger.warning("Validation FAILED for '%s': %s", skill.name, flags)

    return ValidationResult(passed=passed, flags=flags)
