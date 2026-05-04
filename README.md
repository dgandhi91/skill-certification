# Skill Certification Pipeline

A multi-stage evaluation and certification system for AI skills. It validates skill definitions, evaluates performance using LLM-as-a-judge (Gemini 2.5 Flash), computes metrics across multiple dimensions, and assigns a certification tier (Premium / Gold / Silver / Fail).

## Quick Start

```bash
# Install dependencies
make install

# Configure Gemini API key
echo "GEMINI_API_KEY=your-key-here" > .env

# Run via CLI (example with bundled skills)
make run SKILL=skills/csv-analyzer WORKSPACE=skills/csv-analyzer-workspace

# Or launch the dashboard
make dashboard
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make help` | Show all available targets |
| `make install` | Create a virtual environment and install dependencies |
| `make run` | Run the CLI pipeline (`SKILL`, `WORKSPACE`, optional `ITERATION`) |
| `make dashboard` | Launch the Streamlit dashboard |
| `make clean` | Remove venv, caches, and results |

### Using with AI Assistants

**Claude Code:** Type `/certify-skill` to run the certification pipeline. **Cursor:** Ask "certify a skill" or "run the certification pipeline".

Both assistants will:

1. Discover available skills:
   ```bash
   find skills -name "SKILL.md" -exec dirname {} \;
   ```
2. Ask which skill to certify (or use the one you specify)
3. Resolve the workspace path automatically (`skills/{skill-name}-workspace`)
4. Find available iterations:
   ```bash
   ls skills/{skill-name}-workspace/
   ```
5. Run the pipeline:
   ```bash
   python run.py skills/{skill-name} skills/{skill-name}-workspace --iteration 1
   ```
6. Display the certification result (tier, score, flags)

**Directory convention:**
- `skills/{skill-name}/` — skill definition (`SKILL.md` + `evals/evals.json`)
- `skills/{skill-name}-workspace/` — workspace with evaluation runs (`iteration-N/` directories)

**CLI reference:**
```
python run.py <skill_dir> <workspace_dir> [--iteration N] [--no-cache]
```

Use `--no-cache` to skip saving results. Results are cached to `results/{skill-name}/iteration-{N}.json`.

## Project Structure

```
app/
  core/               # Foundation: config, models, utilities
    config.py         # Settings loaded from .env
    models.py         # All Pydantic data models
    embeddings.py     # Text-to-embedding and cosine similarity
    security.py       # LLM-based security scanning (injection, red flags, dangerous tools)
  evaluation/         # Judging and scoring logic
    judge.py          # GeminiJudge — LLM-as-a-judge via Gemini API
    prompts.py        # All LLM prompt templates (evaluation + security)
    metrics.py        # Routing, output quality, grading, benchmark
    scoring.py        # Stage-level scoring, tier thresholds, final score
    validation.py     # Contract validation (async, uses Gemini for security checks)
  pipeline/           # Orchestration and data loading
    pipeline.py       # CertificationPipeline — end-to-end runner
    loader.py         # Workspace directory parser (SKILL.md, evals, runs)
    registry.py       # In-memory skill registry and overlap detection
    results_store.py  # JSON-based results cache (save/load/list/delete)
skills/               # Bundled sample skills and workspaces
.claude/commands/     # Claude Code slash commands
  certify-skill.md    # /certify-skill — run certification pipeline
.cursor/rules/        # Cursor AI rules
  certify-skill.mdc   # Pipeline execution guidance
run.py                # CLI entrypoint
streamlit_app.py      # Streamlit dashboard
```

## Input Format

### Skill Directory

```
skill_dir/
  SKILL.md               # Skill definition with YAML frontmatter
  evals/
    evals.json           # Evaluation dataset
```

**SKILL.md** uses YAML frontmatter following the [AgentSkills spec](https://agentskills.io):

```yaml
---
name: csv-analyzer
description: Analyzes CSV files and generates charts
license: MIT
compatibility: claude-3
metadata:
  author: team
  version: "1.0"
allowed-tools: Bash, Read, Write
---
```

**evals/evals.json** defines the evaluation cases:

```json
{
  "skill_name": "csv-analyzer",
  "evals": [
    {
      "id": 1,
      "prompt": "Analyze the top months in sales.csv",
      "expected_output": "A bar chart showing top 5 months by revenue",
      "assertions": ["Output contains a chart", "Chart shows monthly data"],
      "category": "standard",
      "slug": "top-months-chart"
    }
  ]
}
```

Each eval case has:
- `id` — unique integer identifier
- `prompt` — the input to the skill
- `expected_output` — expected response (empty string for negative cases)
- `assertions` — list of conditions the output must satisfy
- `category` — `"standard"` or `"adversarial"`
- `slug` — kebab-case identifier mapping to the workspace directory name

### Workspace Directory

```
workspace_dir/
  iteration-1/
    eval-{slug}/
      with_skill/
        outputs/          # Skill output files (.txt, .md, .csv, .json)
        timing.json       # {"total_tokens": 150, "duration_ms": 2300}
        grading.json      # Pre-computed grading results
      without_skill/
        outputs/
        timing.json
        grading.json
    benchmark.json        # (Optional) pre-computed benchmark
```

---

## Pipeline Stages

The pipeline runs 5 stages sequentially. If Stage 2 (validation) fails, stages 3–4 are skipped.

### Stage 1 — Skill Definition

Loads and displays the skill metadata and dataset summary. Shows the skill name, description, license, compatibility, author, version, and a breakdown of eval cases by category (positive, negative, adversarial).

**Stage 1 Score (10% of final):** Fraction of optional metadata fields present (license, compatibility, author, version). Each contributes 0 or 1; score = count / 4.

### Stage 2 — Contract Validation

Validates the skill definition and dataset using Gemini-powered security analysis.

**Checks performed (all via LLM except provenance):**

| Check | Method | Type |
|-------|--------|------|
| **Skill description clarity** | LLM ambiguity analysis on skill description | Advisory |
| **Red flags** | LLM scan for network exfiltration, credential access, code execution, obfuscation, privilege escalation, hidden downloads | Required |
| **Author provenance** | Checks if author field is present in metadata | Advisory |
| **Prompt injection** | LLM scan for instruction override, role hijacking, system prompt manipulation | Feeds into risk |
| **Permission scope** | LLM analysis of texts for tools/commands invoked directly or indirectly | Advisory |
| **Security risk level** | Aggregate classification (Low / Medium / High / Extreme) based on all findings | Advisory |

**Security risk classification:**

| Level | Condition |
|-------|-----------|
| Extreme | Injection detected, or 5+ red flags, or red flags + dangerous tools |
| High | Red flags detected |
| Medium | Provenance or permission issues |
| Low | No findings |

**Pass/fail logic:** Fails only if red flags are detected. Ambiguity, provenance, injection, and permission issues are informational and feed into the security risk level.

**Stage 2 Score (20% of final):** Fraction of checks passed (passed / total flags).

### Stage 3 — Registry Overlap Check

Compares the new skill against all previously registered skills using embedding similarity.

- Computes a deterministic embedding of the skill description
- Calculates cosine similarity against every existing registry entry
- Flags overlap if any similarity score >= 0.8 (configurable via `SIMILARITY_THRESHOLD` in `.env`)

| Metric | Description |
|--------|-------------|
| **Similarity Score** | Maximum cosine similarity between the new skill and any registered skill (0.0–1.0) |
| **Conflicts With** | List of registered skill names exceeding the similarity threshold |

**Stage 3 Score (10% of final):** `1.0 - similarity_score`.

### Stage 4 — Evaluation Engine

Runs 4 sub-stages to evaluate skill performance:

#### 4.1 Routing (35% of Stage 4)

Measures how accurately the skill triggers when it should and stays silent when it shouldn't. Built from a binary confusion matrix where `should_trigger = bool(expected_output)` and `did_trigger = bool(actual_output)`.

| Metric | Formula | Description |
|--------|---------|-------------|
| **Recall** | `TP / (TP + FN)` | Proportion of positive cases where the skill produced output |
| **Precision** | `TP / (TP + FP)` | Proportion of triggered cases that were correct |
| **False Trigger Rate** | `FP / (TN + FP)` | Proportion of negative cases incorrectly triggered |

**Score:** Average of recall, precision, and (1 - false trigger rate).

#### 4.2 Output Quality (25% of Stage 4)

Uses Gemini as an LLM judge to score output quality on positive cases.

| Metric | Description |
|--------|-------------|
| **Answer Relevance** | LLM-judged relevance of actual output to the prompt (0.0–1.0) |
| **Faithfulness** | LLM-judged faithfulness of actual output vs expected output (0.0–1.0) |

**Score:** Average of answer relevance and faithfulness.

#### 4.3 Assertion Grading (25% of Stage 4)

LLM judge evaluates custom assertions per eval case with evidence-based pass/fail decisions.

| Metric | Description |
|--------|-------------|
| **Pass Rate** (per eval) | `passed_assertions / total_assertions` |
| **Avg Pass Rate** | Average of all per-eval pass rates |

**Score:** Average pass rate across all graded evals.

#### 4.4 Benchmark (15% of Stage 4)

Compares skill-assisted performance against a baseline (without the skill). Only available if `without_skill` runs exist.

| Metric | Description |
|--------|-------------|
| **Pass Rate** | Average assertion pass rate, per condition |
| **Time (seconds)** | Total execution time per condition |
| **Tokens** | Total token consumption per condition |
| **Delta** | Difference (with_skill - without_skill) |

**Score:** With-skill pass rate (capped at 1.0).

**Stage 4 Score (60% of final):** Weighted average: `0.35 * routing + 0.25 * output_quality + 0.25 * assertion_grading + 0.15 * benchmark`.

### Stage 5 — Scoring & Certification Decision

Computes the final weighted score across all stages and assigns a certification tier.

**Final Score:** `0.10 * stage_1 + 0.20 * stage_2 + 0.10 * stage_3 + 0.60 * stage_4`

**Tier Thresholds:**

| Tier | Min Recall | Max FTR | Min Avg Pass Rate | Min Final Score | Certified |
|------|-----------|---------|-------------------|-----------------|-----------|
| **Premium** | >= 0.9 | <= 0.1 | >= 0.9 | >= 0.9 | Yes |
| **Gold** | >= 0.8 | <= 0.2 | >= 0.8 | >= 0.8 | Yes |
| **Silver** | >= 0.7 | <= 0.3 | >= 0.7 | >= 0.7 | Yes |
| **Fail** | < 0.7 | > 0.3 | < 0.7 | < 0.7 | No |

Tiers are evaluated top-down (Premium first). The first tier where all conditions are met is assigned.

---

## Configuration

Settings are loaded from a `.env` file at the project root:

### Provider Selection

The system supports multiple LLM providers for evaluation. Choose one by setting `JUDGE_PROVIDER`:

```bash
# Copy example config
cp .env.example .env

# Edit .env and set your provider
JUDGE_PROVIDER=gemini  # or ollama, openai, anthropic
```

### Provider-Specific Configuration

**Gemini (Default)**
```bash
JUDGE_PROVIDER=gemini
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-2.5-flash
```

**Ollama (Local)**
```bash
JUDGE_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

Make sure Ollama is running:
```bash
ollama serve
ollama pull llama3.2
```

**OpenAI**
```bash
JUDGE_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o
```

**Anthropic (Claude)**
```bash
JUDGE_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-api-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5
```

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGE_PROVIDER` | `gemini` | LLM provider (gemini, ollama, openai, anthropic) |
| `GEMINI_API_KEY` | `""` | API key for Gemini (required when provider=gemini) |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | Gemini API endpoint |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model to use |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OPENAI_API_KEY` | `""` | API key for OpenAI (required when provider=openai) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API endpoint |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model to use |
| `ANTHROPIC_API_KEY` | `""` | API key for Anthropic (required when provider=anthropic) |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic API endpoint |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Anthropic model to use |
| `SIMILARITY_THRESHOLD` | `0.8` | Cosine similarity threshold for overlap detection |

## References

- [Agent Skills Specification](https://agentskills.io/home)
