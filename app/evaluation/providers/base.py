from __future__ import annotations

from abc import ABC, abstractmethod


class JudgeProvider(ABC):
    """Abstract base class for LLM judge providers."""

    @abstractmethod
    async def score_relevance(self, question: str, answer: str) -> dict:
        """
        Score how relevant the answer is to the question.

        Args:
            question: The input question/prompt
            answer: The generated answer

        Returns:
            dict: {"score": float (0.0-1.0), "reasoning": str}
        """
        pass

    @abstractmethod
    async def score_faithfulness(self, expected: str, actual: str) -> dict:
        """
        Score how faithful the actual output is to the expected output.

        Args:
            expected: The expected output
            actual: The actual output

        Returns:
            dict: {"score": float (0.0-1.0), "reasoning": str}
        """
        pass

    @abstractmethod
    async def grade_assertion(
        self,
        prompt: str,
        expected_output: str,
        actual_output: str,
        assertion: str,
    ) -> dict:
        """
        Grade whether an assertion passes given the prompt and outputs.

        Args:
            prompt: The original prompt
            expected_output: The expected output
            actual_output: The actual output
            assertion: The assertion to check

        Returns:
            dict: {"passed": bool, "evidence": str}
        """
        pass

    @abstractmethod
    async def scan_injection(self, texts: list[str]) -> dict:
        """
        Scan texts for prompt injection patterns.

        Args:
            texts: List of text strings to scan

        Returns:
            dict: {"detected": bool, "patterns": list[str], "reasoning": str}
        """
        pass

    @abstractmethod
    async def scan_ambiguity(self, text: str) -> dict:
        """
        Scan text for ambiguous instructions.

        Args:
            text: Text to scan

        Returns:
            dict: {"ambiguous": bool, "indicators": list[str], "reasoning": str}
        """
        pass

    @abstractmethod
    async def scan_red_flags(self, texts: list[str]) -> dict:
        """
        Scan texts for security red flags.

        Args:
            texts: List of text strings to scan

        Returns:
            dict: {"red_flags": list[str], "reasoning": str}
        """
        pass

    @abstractmethod
    async def scan_dangerous_tools(self, texts: list[str]) -> dict:
        """
        Scan texts for dangerous tool usage.

        Args:
            texts: List of text strings to scan

        Returns:
            dict: {"dangerous_tools": list[str], "reasoning": str}
        """
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate a semantic embedding vector for the given text.

        Used by the registry overlap check (Stage 3) to compute cosine
        similarity between skill descriptions.

        Args:
            text: The text to embed

        Returns:
            list[float]: Embedding vector
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources (close HTTP clients, etc.)."""
        pass
