from __future__ import annotations

import asyncio
import json
import logging

from google import genai

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


class GeminiProvider(JudgeProvider):
    """Gemini LLM judge provider."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.gemini_api_key
        self.base_url = base_url or settings.gemini_base_url
        self.model = model or settings.gemini_model

        # Create the client with API key
        self._client = genai.Client(api_key=self.api_key)

    async def _call_gemini(self, prompt: str, system: str | None = None) -> dict:
        """Call Gemini API and return parsed JSON response."""

        def _generate_content():
            """Synchronous call to SDK wrapped for async execution."""
            # Prepare the config
            config = {"temperature": 0.0, "response_mime_type": "application/json"}
            if system:
                config["system_instruction"] = system

            # Generate content using the client
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )

            return json.loads(response.text)

        # Run the synchronous SDK call in a thread executor to make it async
        return await asyncio.to_thread(_generate_content)

    async def score_relevance(self, question: str, answer: str) -> dict:
        logger.info("Scoring relevance via %s", self.model)
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        try:
            result = await self._call_gemini(prompt)
            logger.info("Relevance score: %.2f", result.get("score", 0.0))
            return result
        except Exception:
            logger.warning("Relevance scoring failed — returning 0.0")
            return {"score": 0.0, "reasoning": "Judge call failed"}

    async def score_faithfulness(self, expected: str, actual: str) -> dict:
        logger.info("Scoring faithfulness via %s", self.model)
        prompt = FAITHFULNESS_PROMPT.format(expected=expected, actual=actual)
        try:
            result = await self._call_gemini(prompt)
            logger.info("Faithfulness score: %.2f", result.get("score", 0.0))
            return result
        except Exception:
            logger.warning("Faithfulness scoring failed — returning 0.0")
            return {"score": 0.0, "reasoning": "Judge call failed"}

    async def grade_assertion(
        self,
        prompt: str,
        expected_output: str,
        actual_output: str,
        assertion: str,
    ) -> dict:
        logger.info("Grading assertion: %s", assertion[:80])
        filled = ASSERTION_GRADING_PROMPT.format(
            prompt=prompt,
            expected_output=expected_output,
            actual_output=actual_output,
            assertion=assertion,
        )
        try:
            result = await self._call_gemini(filled)
            logger.info(
                "Assertion result: %s", "PASS" if result.get("passed") else "FAIL"
            )
            return result
        except Exception:
            logger.warning("Assertion grading failed for: %s", assertion[:80])
            return {"passed": False, "evidence": "Judge call failed"}

    async def scan_injection(self, texts: list[str]) -> dict:
        logger.info(
            "Scanning %d texts for injection patterns via %s", len(texts), self.model
        )
        joined = "\n---\n".join(texts)
        prompt = SECURITY_INJECTION_PROMPT.format(texts=joined)
        try:
            return await self._call_gemini(prompt)
        except Exception:
            logger.warning("Injection scan failed — returning empty")
            return {
                "detected": False,
                "patterns": [],
                "reasoning": "Judge call failed",
            }

    async def scan_ambiguity(self, text: str) -> dict:
        logger.info("Scanning for ambiguity via %s", self.model)
        prompt = SECURITY_AMBIGUITY_PROMPT.format(text=text)
        try:
            return await self._call_gemini(prompt)
        except Exception:
            logger.warning("Ambiguity scan failed — returning empty")
            return {
                "ambiguous": False,
                "indicators": [],
                "reasoning": "Judge call failed",
            }

    async def scan_red_flags(self, texts: list[str]) -> dict:
        logger.info("Scanning %d texts for red flags via %s", len(texts), self.model)
        joined = "\n---\n".join(texts)
        prompt = SECURITY_RED_FLAGS_PROMPT.format(texts=joined)
        try:
            return await self._call_gemini(prompt)
        except Exception:
            logger.warning("Red flag scan failed — returning empty")
            return {"red_flags": [], "reasoning": "Judge call failed"}

    async def scan_dangerous_tools(self, texts: list[str]) -> dict:
        logger.info(
            "Scanning %d texts for dangerous tools via %s", len(texts), self.model
        )
        joined = "\n---\n".join(texts)
        prompt = SECURITY_DANGEROUS_TOOLS_PROMPT.format(texts=joined)
        try:
            return await self._call_gemini(prompt)
        except Exception:
            logger.warning("Dangerous tools scan failed — returning empty")
            return {"dangerous_tools": [], "reasoning": "Judge call failed"}

    async def close(self) -> None:
        # No cleanup needed for the SDK-based implementation
        pass
