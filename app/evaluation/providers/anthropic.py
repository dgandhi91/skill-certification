from __future__ import annotations

import json
import logging
import re

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.evaluation.prompts import (
    ASSERTION_GRADING_PROMPT,
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    SECURITY_AMBIGUITY_PROMPT,
    SECURITY_DANGEROUS_TOOLS_PROMPT,
    SECURITY_INJECTION_PROMPT,
    SECURITY_RED_FLAGS_PROMPT,
)
from app.evaluation.providers.base import JudgeProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(JudgeProvider):
    """Anthropic LLM judge provider using the official Anthropic SDK with prompt caching."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.anthropic_api_key
        self.base_url = base_url or settings.anthropic_base_url
        self.model = model or settings.anthropic_model
        self._client = AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)

    async def _call_anthropic(self, prompt: str, system: str | None = None) -> dict:
        """
        Call Anthropic Messages API and return parsed JSON response.

        Uses prompt caching on the system prompt for cost savings.
        Anthropic does not have native JSON mode, so we instruct via system prompt.
        """
        # Build system prompt with caching
        system_messages = []
        if system:
            system_messages.append(
                {
                    "type": "text",
                    "text": system,
                }
            )

        # Add JSON instruction with cache_control marker
        system_messages.append(
            {
                "type": "text",
                "text": "Return valid JSON only. No markdown, no code blocks.",
                "cache_control": {"type": "ephemeral"},
            }
        )

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.0,
            system=system_messages,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON from response content
        content = response.content[0].text

        # Try to parse directly first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: extract from markdown code blocks
            # Look for ```json ... ``` or ``` ... ```
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            # If that fails, try to parse the whole content
            raise

    async def score_relevance(self, question: str, answer: str) -> dict:
        logger.info("Scoring relevance via Anthropic %s", self.model)
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        try:
            result = await self._call_anthropic(prompt)
            logger.info("Relevance score: %.2f", result.get("score", 0.0))
            return result
        except Exception as e:
            logger.warning("Relevance scoring failed: %s", e)
            return {"score": 0.0, "reasoning": "Judge call failed"}

    async def score_faithfulness(self, expected: str, actual: str) -> dict:
        logger.info("Scoring faithfulness via Anthropic %s", self.model)
        prompt = FAITHFULNESS_PROMPT.format(expected=expected, actual=actual)
        try:
            result = await self._call_anthropic(prompt)
            logger.info("Faithfulness score: %.2f", result.get("score", 0.0))
            return result
        except Exception as e:
            logger.warning("Faithfulness scoring failed: %s", e)
            return {"score": 0.0, "reasoning": "Judge call failed"}

    async def grade_assertion(
        self,
        prompt: str,
        expected_output: str,
        actual_output: str,
        assertion: str,
    ) -> dict:
        logger.info("Grading assertion via Anthropic: %s", assertion[:80])
        filled = ASSERTION_GRADING_PROMPT.format(
            prompt=prompt,
            expected_output=expected_output,
            actual_output=actual_output,
            assertion=assertion,
        )
        try:
            result = await self._call_anthropic(filled)
            logger.info(
                "Assertion result: %s", "PASS" if result.get("passed") else "FAIL"
            )
            return result
        except Exception as e:
            logger.warning("Assertion grading failed: %s", e)
            return {"passed": False, "evidence": "Judge call failed"}

    async def scan_injection(self, texts: list[str]) -> dict:
        logger.info(
            "Scanning %d texts for injection via Anthropic %s", len(texts), self.model
        )
        joined = "\n---\n".join(texts)
        prompt = SECURITY_INJECTION_PROMPT.format(texts=joined)
        try:
            return await self._call_anthropic(prompt)
        except Exception as e:
            logger.warning("Injection scan failed: %s", e)
            return {
                "detected": False,
                "patterns": [],
                "reasoning": "Judge call failed",
            }

    async def scan_ambiguity(self, text: str) -> dict:
        logger.info("Scanning for ambiguity via Anthropic %s", self.model)
        prompt = SECURITY_AMBIGUITY_PROMPT.format(text=text)
        try:
            return await self._call_anthropic(prompt)
        except Exception as e:
            logger.warning("Ambiguity scan failed: %s", e)
            return {
                "ambiguous": False,
                "indicators": [],
                "reasoning": "Judge call failed",
            }

    async def scan_red_flags(self, texts: list[str]) -> dict:
        logger.info(
            "Scanning %d texts for red flags via Anthropic %s", len(texts), self.model
        )
        joined = "\n---\n".join(texts)
        prompt = SECURITY_RED_FLAGS_PROMPT.format(texts=joined)
        try:
            return await self._call_anthropic(prompt)
        except Exception as e:
            logger.warning("Red flag scan failed: %s", e)
            return {"red_flags": [], "reasoning": "Judge call failed"}

    async def scan_dangerous_tools(self, texts: list[str]) -> dict:
        logger.info(
            "Scanning %d texts for dangerous tools via Anthropic %s",
            len(texts),
            self.model,
        )
        joined = "\n---\n".join(texts)
        prompt = SECURITY_DANGEROUS_TOOLS_PROMPT.format(texts=joined)
        try:
            return await self._call_anthropic(prompt)
        except Exception as e:
            logger.warning("Dangerous tools scan failed: %s", e)
            return {"dangerous_tools": [], "reasoning": "Judge call failed"}

    async def close(self) -> None:
        await self._client.close()
