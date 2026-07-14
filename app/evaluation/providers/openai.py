from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

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


class OpenAIProvider(JudgeProvider):
    """OpenAI LLM judge provider using the official OpenAI SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url
        self.model = model or settings.openai_model
        self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def _call_openai(self, prompt: str, system: str | None = None) -> dict:
        """Call OpenAI Chat Completions API and return parsed JSON response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        # Extract JSON from response
        content = response.choices[0].message.content
        return json.loads(content)

    async def score_relevance(self, question: str, answer: str) -> dict:
        logger.info("Scoring relevance via OpenAI %s", self.model)
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        try:
            result = await self._call_openai(prompt)
            logger.info("Relevance score: %.2f", result.get("score", 0.0))
            return result
        except Exception as e:
            logger.warning("Relevance scoring failed: %s", e)
            return {"score": 0.0, "reasoning": "Judge call failed"}

    async def score_faithfulness(self, expected: str, actual: str) -> dict:
        logger.info("Scoring faithfulness via OpenAI %s", self.model)
        prompt = FAITHFULNESS_PROMPT.format(expected=expected, actual=actual)
        try:
            result = await self._call_openai(prompt)
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
        logger.info("Grading assertion via OpenAI: %s", assertion[:80])
        filled = ASSERTION_GRADING_PROMPT.format(
            prompt=prompt,
            expected_output=expected_output,
            actual_output=actual_output,
            assertion=assertion,
        )
        try:
            result = await self._call_openai(filled)
            logger.info(
                "Assertion result: %s", "PASS" if result.get("passed") else "FAIL"
            )
            return result
        except Exception as e:
            logger.warning("Assertion grading failed: %s", e)
            return {"passed": False, "evidence": "Judge call failed"}

    async def scan_injection(self, texts: list[str]) -> dict:
        logger.info(
            "Scanning %d texts for injection via OpenAI %s", len(texts), self.model
        )
        joined = "\n---\n".join(texts)
        prompt = SECURITY_INJECTION_PROMPT.format(texts=joined)
        try:
            return await self._call_openai(prompt)
        except Exception as e:
            logger.warning("Injection scan failed: %s", e)
            return {
                "detected": False,
                "patterns": [],
                "reasoning": "Judge call failed",
            }

    async def scan_ambiguity(self, text: str) -> dict:
        logger.info("Scanning for ambiguity via OpenAI %s", self.model)
        prompt = SECURITY_AMBIGUITY_PROMPT.format(text=text)
        try:
            return await self._call_openai(prompt)
        except Exception as e:
            logger.warning("Ambiguity scan failed: %s", e)
            return {
                "ambiguous": False,
                "indicators": [],
                "reasoning": "Judge call failed",
            }

    async def scan_red_flags(self, texts: list[str]) -> dict:
        logger.info(
            "Scanning %d texts for red flags via OpenAI %s", len(texts), self.model
        )
        joined = "\n---\n".join(texts)
        prompt = SECURITY_RED_FLAGS_PROMPT.format(texts=joined)
        try:
            return await self._call_openai(prompt)
        except Exception as e:
            logger.warning("Red flag scan failed: %s", e)
            return {"red_flags": [], "reasoning": "Judge call failed"}

    async def scan_dangerous_tools(self, texts: list[str]) -> dict:
        logger.info(
            "Scanning %d texts for dangerous tools via OpenAI %s",
            len(texts),
            self.model,
        )
        joined = "\n---\n".join(texts)
        prompt = SECURITY_DANGEROUS_TOOLS_PROMPT.format(texts=joined)
        try:
            return await self._call_openai(prompt)
        except Exception as e:
            logger.warning("Dangerous tools scan failed: %s", e)
            return {"dangerous_tools": [], "reasoning": "Judge call failed"}

    async def embed(self, text: str) -> list[float]:
        logger.info("Generating embedding via OpenAI text-embedding-3-small")
        try:
            response = await self._client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return response.data[0].embedding
        except Exception:
            logger.warning("OpenAI embedding failed — falling back to hash embedding")
            from app.core.embeddings import _hash_embedding

            return _hash_embedding(text)

    async def close(self) -> None:
        await self._client.close()
