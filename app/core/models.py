from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# --- Skill & Eval Dataset (AgentSkills spec) ---


class SkillMetadata(BaseModel):
    author: str = ""
    version: str = "1.0"


class SkillDefinition(BaseModel):
    """Mirrors the SKILL.md frontmatter from the AgentSkills spec."""

    name: str = Field(..., max_length=64, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    description: str = Field(..., min_length=1, max_length=1024)
    license: str = ""
    compatibility: str = ""
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)
    allowed_tools: str = ""


class EvalCase(BaseModel):
    """A single eval test case from evals/evals.json."""

    id: int
    prompt: str
    expected_output: str
    files: list[str] = Field(default_factory=list)
    assertions: list[str] = Field(default_factory=list)
    category: str = "standard"
    slug: str = ""


class EvalDataset(BaseModel):
    """Top-level evals/evals.json structure."""

    skill_name: str
    evals: list[EvalCase] = Field(default_factory=list)


# --- Grading & Timing ---


class AssertionResult(BaseModel):
    text: str
    passed: bool
    evidence: str


class GradingSummary(BaseModel):
    passed: int = 0
    failed: int = 0
    total: int = 0
    pass_rate: float = 0.0


class GradingResult(BaseModel):
    assertion_results: list[AssertionResult] = Field(default_factory=list)
    summary: GradingSummary = Field(default_factory=GradingSummary)


class TimingData(BaseModel):
    total_tokens: int = 0
    duration_ms: int = 0


class RunStats(BaseModel):
    pass_rate: float = 0.0
    time_seconds: float = 0.0
    tokens: int = 0


class BenchmarkResult(BaseModel):
    with_skill: RunStats = Field(default_factory=RunStats)
    without_skill: RunStats = Field(default_factory=RunStats)
    delta: RunStats = Field(default_factory=RunStats)


class EvalRunOutput(BaseModel):
    """Output of a single eval case run (with_skill or without_skill)."""

    eval_id: int
    prompt: str
    actual_output: str = ""
    grading: GradingResult = Field(default_factory=GradingResult)
    timing: TimingData = Field(default_factory=TimingData)


# --- Evaluation Metrics ---


class RoutingMetrics(BaseModel):
    recall: float = 0.0
    precision: float = 0.0
    false_trigger_rate: float = 0.0


class OutputQualityMetrics(BaseModel):
    answer_relevance: float = 0.0
    faithfulness: float = 0.0


class EvaluationResult(BaseModel):
    skill_name: str
    routing: RoutingMetrics = Field(default_factory=RoutingMetrics)
    output_quality: OutputQualityMetrics = Field(default_factory=OutputQualityMetrics)
    grading: list[GradingResult] = Field(default_factory=list)
    benchmark: BenchmarkResult | None = None
    flags: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    passed: bool = True
    flags: list[str] = Field(default_factory=list)


class OverlapResult(BaseModel):
    overlap: bool = False
    similarity_score: float = 0.0
    conflicts_with: list[str] = Field(default_factory=list)


# --- Certification ---


class Tier(str, Enum):
    PREMIUM = "Premium"
    GOLD = "Gold"
    SILVER = "Silver"
    FAIL = "Fail"


class CertificationDecision(BaseModel):
    certified: bool = False
    tier: Tier = Tier.FAIL
    reasons: list[str] = Field(default_factory=list)
    evaluation: EvaluationResult | None = None


class RegistryEntry(BaseModel):
    skill_name: str
    description: str
    metadata: dict[str, str] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    certification: CertificationDecision | None = None
    evaluation: EvaluationResult | None = None
