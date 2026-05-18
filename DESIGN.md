# Design Notes

This document captures the design philosophy, open questions, and rationale behind key decisions in LossCast-Bench. It's meant to give contributors and collaborators (human or agent) the context to make good judgment calls when the schema or scope needs to evolve.

## Core Idea

Scaling laws (Chinchilla, etc.) model loss as a function of two variables: parameter count N and data size D. But practitioners know loss depends on much more — optimizer, learning rate, schedule, architecture details, initialization, data mix. There's no good model for the full picture.

LossCast-Bench treats this as a supervised learning problem: given a fully specified training recipe, predict the loss at every eval checkpoint. The dataset is a growing collection of (recipe, loss_curve) pairs from real experiments.

The key insight is that **the recipe itself is the input, not just (N, D)**. This means the predictor has to learn how optimizer choice, LR schedule shape, warmup length, weight decay, and all the other knobs interact with scale to determine the loss trajectory.

## Two Layers of Representation

Each run has a JSON layer (config.json + losses.json) and a code layer (model.py + optimizer.py).

The JSON layer is structured and machine-readable — it's what most predictors will consume. But it can't capture everything: per-layer init scaling, custom attention patterns, non-standard activations, subtle differences between optimizer implementations. The code layer preserves these details for agent-based approaches that can reason about source code.

This dual-layer design is deliberate. We expect the first wave of predictors to use only the JSON features. But the long-term bet is that code-reading agents will extract richer signal from the implementation details.

## Why Loss Curves, Not Just Final Loss

Most scaling law papers predict final loss only. We require the full curve because:

1. The curve shape contains information about the training dynamics (warmup behavior, mid-training instabilities, cooldown effects) that final loss alone doesn't capture.
2. A predictor that gets the curve right is more useful in practice — you can predict early stopping points, detect anomalies mid-training, and estimate how much additional compute would help.
3. It's a harder task, which means the benchmark has more room to differentiate methods.

## The eval_dataset Problem

Different data sources measure loss on different things. Marin runs report eval loss on C4 English (a held-out validation set from the Paloma benchmark). StepLaw runs report pretraining loss (effectively training loss, no separate eval set). These numbers are not directly comparable.

We handle this with the `eval_dataset` field in config.json:
- `eval_dataset: "c4_en"` — loss measured on C4 English validation set (Marin)
- `eval_dataset: "train"` — loss measured on the training distribution (StepLaw)
- Other values as needed: `"fineweb"`, `"pile"`, `"c4"`, etc.

Losses are only comparable across runs with the same `tokenizer` + `eval_dataset` combination. A predictor should learn that runs with different eval_datasets live on different scales.

We considered recording both train and eval loss for every run, but decided against it for now. The benchmark's job is to predict the target loss curve — which loss that is depends on the source. Adding a separate `train_losses.json` would complicate the schema for questionable benefit at this stage. This can be revisited if someone has a concrete use case.

## Tokenizer Matters

Two runs with identical configs but different tokenizers will have different losses, because cross-entropy loss in nats depends on the vocabulary size and token distribution. The `tokenizer` field is required so predictors can learn this. For Marin data, it's `"EleutherAI/gpt-neox-20b"` (GPT-NeoX tokenizer). For nanochat-style runs, it's typically `"gpt2"`.

## Data Sourcing Strategy

### Phase 1: Seed with existing data (current)
We're bootstrapping from two large published datasets:

**Marin (Fantastic Optimizers paper):** 2,549 runs, 130M–1.2B params, multiple optimizers (AdamW, Muon, Mudam, Shampoo, etc.). Rich optimizer diversity. Eval loss on C4 English via Paloma benchmark. WandB project: `stanford-mercury/optimizer-scaling`.

**StepLaw:** 2,581 runs, 215M–1B params, AdamW only, fine-grained LR and batch-size sweeps. Good for learning how LR and batch size affect loss independently. Pretraining loss only. WandB project: `billzid/predictable-scale`.

**NCPL (Zhang, Wen, Ma 2026):** Published a curated dataset at `huggingface.co/datasets/zhqwqwq/NCPL-Pretraining-Logs` combining both sources (2,549 Marin + 2,581 StepLaw). We used this as our seed data, converting 5,086 runs to benchmark format and splitting by NCPL's published OOD rule: 3,977 train (≤430M params) + 1,109 val (>430M params, matches NCPL's OOD-val count exactly). NCPL's own train/ID-val split indices are not published, so our train corresponds to NCPL's (train ∪ ID-val) pool rather than their exact train split.

### Phase 2: Community contributions
The CONTRIBUTE.md pipeline is designed for agents. The ideal flow: an agent runs a pretraining experiment, then follows CONTRIBUTE.md to format and submit the data via PR. The code-rewriter skill (`.claude/skills/code-rewriter/`) helps convert framework-specific code to the benchmark's self-contained format.

### Phase 3: Autoresearch integration
Projects like Karpathy's autoresearch run hundreds of experiments automatically. If those systems output data in losscast-bench format as a side effect, the dataset grows with zero marginal cost.

## Splits and Evaluation

- `train/` and `val/` are public (configs + ground truth). Seeded from NCPL's published dataset (2,549 Marin + 2,537 StepLaw runs). Predictors are built on train, tuned on val.
- `test/` contains private runs (our own experiments, community-contributed private data). Ground truth is never published. Participants submit their predictor function, and the eval server runs it.

The train/val split follows NCPL's OOD-val rule verbatim: val = runs with n_params > 430M (1,109 runs, matching NCPL's count exactly), train = runs with n_params ≤ 430M (3,977 runs). This is a pure scale-extrapolation setup and is leakage-proof against any predictor trained on NCPL's own train + ID-val pool. A legacy `group-stratified` mode (80/20 random split over (source, optimizer, N_bucket, D_bucket) groups) is still available via `losscast-dev/splits/build_splits.py --mode group-stratified`, but should not be used — it re-randomizes NCPL's split and leaks.

The test set is hidden because the **data itself is private** — not just the configs. Public data (like NCPL) cannot go into test because anyone could download the ground truth. Test data must come from experiments whose loss curves have never been published.

## Schema Decisions Still Open

- **loss_type / loss_smoothing**: Not yet in the schema. StepLaw's "pretraining loss" may be smoothed; Marin's eval loss is not. If this becomes a problem for predictors, we'll add fields to distinguish.
- **Multi-eval datasets**: Some runs (like Marin) have losses on many eval sets (C4, WikiText, RedPajama, etc.). Currently we pick one. We could add a `multi_eval_losses.json` later for runs that have richer eval data.
- **Hardware / throughput metadata**: We have `hardware` and `precision` fields but they don't affect loss mathematically (only speed). Keeping them for provenance but not expecting predictors to use them.
- **Data mix details**: The `mix` field in data config is currently free-form (null or a description). A structured format for data mixtures would help, but the right schema depends on what data sources we end up with.

## Relationship to NCPL

NCPL (Neural Configuration-to-Performance Law) by Zhang, Wen, Ma (2025) is the closest prior work. They train Qwen3-1.7B to predict loss from a text description of the training config. Key differences from losscast-bench:

1. NCPL predicts residuals relative to a Chinchilla baseline. We predict raw loss values (but Chinchilla is available as a baseline).
2. NCPL uses text input (natural language config description). We provide structured JSON + source code.
3. NCPL's dataset mixes Marin eval loss and StepLaw training loss without distinguishing them. We make this explicit via `eval_dataset`.
4. We include the code layer (model.py, optimizer.py) which NCPL doesn't use.
5. We're a living benchmark with community contributions. NCPL is a fixed dataset.

NCPL's published dataset (`zhqwqwq/NCPL-Pretraining-Logs`) is used as our seed data. We converted all 5,130 runs, applying the `eval_dataset` distinction (Marin → `c4_en`, StepLaw → `train`), parsing StepLaw architecture from run names (separating d_model from d_ff), and interpreting the epsilon field correctly (stored as -log10 in NCPL).
