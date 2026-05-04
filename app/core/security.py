from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.evaluation.judge import GeminiJudge

logger = logging.getLogger(__name__)


async def scan_for_injection(texts: list[str], judge: GeminiJudge) -> list[str]:
    logger.info("Scanning %d texts for injection patterns via LLM", len(texts))
    result = await judge.scan_injection(texts)
    findings: list[str] = []
    if result.get("detected"):
        for pattern in result.get("patterns", []):
            logger.warning("Injection pattern detected: %s", pattern)
            findings.append(f"Prompt injection pattern detected: {pattern}")
    if not findings:
        logger.debug("No injection patterns found")
    return findings


async def detect_ambiguity(text: str, judge: GeminiJudge) -> list[str]:
    logger.info("Scanning for ambiguity via LLM")
    result = await judge.scan_ambiguity(text)
    if result.get("ambiguous"):
        indicators = result.get("indicators", [])
        reasoning = result.get("reasoning", "")
        logger.warning("Ambiguity detected: %s", reasoning)
        return [
            f"High ambiguity detected ({len(indicators)} indicators: "
            f"{', '.join(indicators[:5])}). Consider clarifying the skill description."
        ]
    logger.debug("No significant ambiguity found")
    return []


async def scan_for_red_flags(texts: list[str], judge: GeminiJudge) -> list[str]:
    logger.info("Scanning %d texts for red flags via LLM", len(texts))
    result = await judge.scan_red_flags(texts)
    findings: list[str] = []
    seen_categories: set[str] = set()
    for flag in result.get("red_flags", []):
        category = flag.get("category", "unknown")
        detail = flag.get("detail", "")
        if category not in seen_categories:
            logger.warning("Red flag: %s — %s", category, detail)
            findings.append(f"Red flag: {category} ({detail})")
            seen_categories.add(category)
    if not findings:
        logger.debug("No red flags found")
    return findings


async def check_permission_scope(texts: list[str], judge: GeminiJudge) -> list[str]:
    logger.info("Scanning %d texts for dangerous tools via LLM", len(texts))
    result = await judge.scan_dangerous_tools(texts)
    findings: list[str] = []
    for tool_info in result.get("dangerous_tools", []):
        tool = tool_info.get("tool", "unknown")
        invocation = tool_info.get("invocation", "unknown")
        detail = tool_info.get("detail", "")
        logger.warning("Dangerous tool: %s (%s) — %s", tool, invocation, detail)
        findings.append(
            f"Permission scope: dangerous tool '{tool}' ({invocation}) — {detail}"
        )
    if not findings:
        logger.debug("No dangerous tool invocations found")
    return findings


def check_provenance(metadata_author: str) -> list[str]:
    if not metadata_author or not metadata_author.strip():
        logger.warning("Provenance check failed: author is missing or empty")
        return ["Provenance: author is missing from skill metadata"]
    logger.debug("Provenance check passed: author='%s'", metadata_author)
    return []


def classify_risk(
    red_flag_count: int,
    provenance_count: int,
    permission_count: int,
    injection_count: int = 0,
) -> str:
    if (
        injection_count > 0
        or red_flag_count >= 5
        or (red_flag_count > 0 and permission_count > 0)
    ):
        level = "Extreme"
    elif red_flag_count > 0:
        level = "High"
    elif provenance_count > 0 or permission_count > 0:
        level = "Medium"
    else:
        level = "Low"
    logger.info(
        "Risk classification: %s (injections=%d, red_flags=%d, provenance=%d, permissions=%d)",
        level,
        injection_count,
        red_flag_count,
        provenance_count,
        permission_count,
    )
    return level
