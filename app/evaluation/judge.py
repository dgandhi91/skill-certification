"""
DEPRECATED: This module is deprecated. Use app.evaluation.providers instead.

For new code, use:
    from app.evaluation.providers import create_judge
    judge = create_judge()

This module is kept for backward compatibility only.
"""

import warnings

from app.evaluation.providers.gemini import GeminiProvider

__all__ = ["GeminiJudge"]


class GeminiJudge(GeminiProvider):
    """
    DEPRECATED: Use create_judge() from app.evaluation.providers instead.

    This class is kept for backward compatibility only. It will be removed
    in a future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GeminiJudge is deprecated. Use create_judge() from "
            "app.evaluation.providers instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
