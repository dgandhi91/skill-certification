Run the skill certification pipeline end-to-end.

## How to discover skills

Skills live in the `skills/` directory. Each skill has two directories:
- `skills/{skill-name}/` — skill definition (contains `SKILL.md` and `evals/evals.json`)
- `skills/{skill-name}-workspace/` — workspace with evaluation runs (contains `iteration-N/` directories)

Find available skills:
```bash
find skills -name "SKILL.md" -exec dirname {} \;
```

Find available workspaces and iterations:
```bash
find skills -type d -name "iteration-*" | sort
```

## How to run

The CLI command is:
```bash
python run.py <skill_dir> <workspace_dir> [--iteration N] [--no-cache]
```

For example:
```bash
python run.py skills/csv-analyzer skills/csv-analyzer-workspace --iteration 1
```

The workspace directory is always `skills/{skill-name}-workspace`. The iteration defaults to 1.

## Steps

1. Run `find skills -name "SKILL.md" -exec dirname {} \;` to list available skills
2. If the user provided a skill name as `$ARGUMENTS`, use that. Otherwise ask which skill to certify.
3. Determine the workspace path: replace the skill directory name with `{skill-name}-workspace` (e.g., `skills/csv-analyzer` -> `skills/csv-analyzer-workspace`)
4. Check which iterations exist: `ls skills/{skill-name}-workspace/` and look for `iteration-N` directories
5. Run: `python run.py skills/{skill-name} skills/{skill-name}-workspace --iteration {N}`
6. Show the certification result (tier, score, and any flags)

## Alternative: Streamlit dashboard

For an interactive UI with cached results and visualizations:
```bash
streamlit run streamlit_app.py
```

## Pipeline stages

The pipeline runs 5 stages:
1. **Skill Definition** — loads SKILL.md and evals.json
2. **Contract Validation** — LLM-based security checks (injection, red flags, dangerous tools)
3. **Registry Overlap** — embedding similarity against registered skills
4. **Evaluation Engine** — routing, output quality, assertion grading, benchmark
5. **Scoring & Certification** — weighted final score, tier assignment (Premium/Gold/Silver/Fail)

Results are cached to `results/{skill-name}/iteration-{N}.json`. Use `--no-cache` to skip caching.
