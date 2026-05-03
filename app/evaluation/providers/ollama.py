from __future__ import annotations

import json
import logging

import httpx

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


class OllamaProvider(JudgeProvider):
    """Ollama local LLM judge provider."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self._client = httpx.AsyncClient(timeout=60.0)

    async def _call_ollama(self, prompt: str, system: str | None = None) -> dict:
        """Call Ollama API and return parsed JSON response."""
        url = f"{self.base_url}/api/generate"

        # Ollama doesn't have native system instruction support
        # Prepend system message to prompt
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
        }

        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns JSON in different fields depending on the model:
        # - qwen models use "thinking" field for JSON output
        # - other models use "response" field
        # Try "thinking" first (qwen), fall back to "response"
        response_text = data.get("thinking") or data.get("response", "")

        if not response_text:
            logger.warning("Empty response from Ollama model %s", self.model)
            raise ValueError("Empty response from Ollama")

        return json.loads(response_text)

    async def score_relevance(self, question: str, answer: str) -> dict:
        logger.info("Scoring relevance via Ollama %s", self.model)
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        try:
            result = await self._call_ollama(prompt)
            logger.info("Relevance score: %.2f", result.get("score", 0.0))
            return result
        except Exception as e:
            logger.warning("Relevance scoring failed: %s", e)
            return {"score": 0.0, "reasoning": "Judge call failed"}

    async def score_faithfulness(self, expected: str, actual: str) -> dict:
        logger.info("Scoring faithfulness via Ollama %s", self.model)
        prompt = FAITHFULNESS_PROMPT.format(expected=expected, actual=actual)
        try:
            result = await self._call_ollama(prompt)
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
        logger.info("Grading assertion via Ollama: %s", assertion[:80])
        filled = ASSERTION_GRADING_PROMPT.format(
            prompt=prompt,
            expected_output=expected_output,
            actual_output=actual_output,
            assertion=assertion,
        )
        try:
            result = await self._call_ollama(filled)
            logger.info(
                "Assertion result: %s", "PASS" if result.get("passed") else "FAIL"
            )
            return result
        except Exception as e:
            logger.warning("Assertion grading failed: %s", e)
            return {"passed": False, "evidence": "Judge call failed"}

    async def scan_injection(self, texts: list[str]) -> dict:
        logger.info("Scanning %d texts for injection via Ollama", len(texts))
        joined = "\n---\n".join(texts)
        prompt = SECURITY_INJECTION_PROMPT.format(texts=joined)
        try:
            return await self._call_ollama(prompt)
        except Exception as e:
            logger.warning("Injection scan failed: %s", e)
            return {
                "detected": False,
                "patterns": [],
                "reasoning": "Judge call failed",
            }

    async def scan_ambiguity(self, text: str) -> dict:
        logger.info("Scanning for ambiguity via Ollama")
        prompt = SECURITY_AMBIGUITY_PROMPT.format(text=text)
        try:
            return await self._call_ollama(prompt)
        except Exception as e:
            logger.warning("Ambiguity scan failed: %s", e)
            return {
                "ambiguous": False,
                "indicators": [],
                "reasoning": "Judge call failed",
            }

    async def scan_red_flags(self, texts: list[str]) -> dict:
        logger.info("Scanning %d texts for red flags via Ollama", len(texts))
        joined = "\n---\n".join(texts)
        prompt = SECURITY_RED_FLAGS_PROMPT.format(texts=joined)
        try:
            return await self._call_ollama(prompt)
        except Exception as e:
            logger.warning("Red flag scan failed: %s", e)
            return {"red_flags": [], "reasoning": "Judge call failed"}

    async def scan_dangerous_tools(self, texts: list[str]) -> dict:
        logger.info("Scanning %d texts for dangerous tools via Ollama", len(texts))
        joined = "\n---\n".join(texts)
        prompt = SECURITY_DANGEROUS_TOOLS_PROMPT.format(texts=joined)
        try:
            return await self._call_ollama(prompt)
        except Exception as e:
            logger.warning("Dangerous tools scan failed: %s", e)
            return {"dangerous_tools": [], "reasoning": "Judge call failed"}

    async def close(self) -> None:
        await self._client.aclose()
