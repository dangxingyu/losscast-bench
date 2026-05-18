# Publish Plan

How this dev repo maps to the public release.

## Current State (Dev)

Everything is in one repo, all tracked:

```
losscast-bench/              # ← this repo, dev mode
├── losscast_bench/          # Python package (published)
├── scripts/                 # CLI tools (published)
├── data/                    # 5k+ runs (published train/val, private test)
│   ├── train/               # 3,977 NCPL runs (≤430M params)
│   ├── val/                 # 1,109 NCPL runs (>430M params, NCPL OOD-val)
│   ├── staging/             # Intermediate conversion output
│   └── test/                # Private runs (future)
├── raw_data/                # WandB exports, not in benchmark format
├── losscast-dev/            # Dev scripts, conversion, exploration, agents
├── .claude/skills/          # Claude Code skills
├── tests/                   # Test suite
├── templates/               # Contribution templates
└── docs/                    # Specs
```

## Published Version

When we release, split into:

### Public repo: `losscast-bench`

What users and agents clone:

```
losscast-bench/
├── losscast_bench/          # Package
├── scripts/                 # eval_cli, run_baseline, validate_contribution, nanochat_to_bench
├── data/
│   ├── train/               # Public training data
│   └── val/                 # Public validation data
├── .claude/skills/          # schema-creator + code-rewriter
├── tests/
├── templates/
├── docs/
├── README.md, PREDICT.md, CONTRIBUTE.md, DESIGN.md
└── pyproject.toml
```

**Excluded from public:**
- `data/staging/` — intermediate, not needed
- `data/test/` — private, lives on eval server only
- `raw_data/` — WandB dumps, not in benchmark format
- `losscast-dev/` — dev scripts, exploration, conversion pipelines, agent tests

### What goes where

| Content | Public repo | Eval server | Dev-only |
|---------|-------------|-------------|----------|
| `losscast_bench/` package | ✓ | ✓ | |
| `scripts/` CLI tools | ✓ | ✓ | |
| `data/train/` + `data/val/` | ✓ | ✓ | |
| `data/test/` | | ✓ | |
| `.claude/skills/` | ✓ | | |
| `raw_data/` | | | ✓ |
| `losscast-dev/` | | | ✓ |
| `data/staging/` | | | ✓ |

### Publish checklist

- [ ] Clean up `data/staging/` (or just don't include)
- [ ] Verify `data/train/` and `data/val/` are complete and valid
- [ ] Populate `data/test/` with private nanochat runs
- [ ] Set up eval server (GitHub Actions? hosted service?)
- [ ] Update git clone URL in README
- [ ] Tag v0.1.0
- [ ] Add `.gitignore` entries for dev-only content
- [ ] Consider: separate `losscast-dev` into its own private repo?
