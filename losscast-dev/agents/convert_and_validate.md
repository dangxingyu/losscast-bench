# Conversion Agent Prompt

Use this prompt with Claude Code (or Agent SDK) to test the code-rewriter skill
on real nanochat source code.

## Usage

### Via Claude Code CLI
```bash
cd /path/to/losscast-bench
claude "$(cat losscast-dev/agents/convert_and_validate.md)"
```

### Via Agent SDK
```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    cwd="/path/to/losscast-bench",
    setting_sources=["user", "project"],  # needed for skills
    allowed_tools=["Skill", "Read", "Write", "Bash", "Edit", "Glob", "Grep"],
)

prompt = open("losscast-dev/agents/convert_and_validate.md").read()
async for msg in query(prompt=prompt, options=options):
    print(msg)
```

---

## Agent Task

You are testing the LossCast-Bench code-rewriter conversion pipeline. Your job is
to convert real nanochat training code into the benchmark's standard format, then
validate the results.

### Step 1: Understand the source code

Read the nanochat source files in `losscast-dev/examples/nanochat_d12/source/`:
- `gpt.py` — the transformer model (GPTConfig, GPT class)
- `optim.py` — MuonAdamW hybrid optimizer
- `base_train.py` — training script with CLI args

Pay attention to:
- Model architecture: RoPE, ReLU², QK norm, value embeddings, smear gate
- Optimizer: hybrid Muon (for matrix params) + AdamW (for embeddings/scalars)
- Training config: how depth/aspect_ratio determine model dimensions

### Step 2: Convert using the code-rewriter skill

Use the code-rewriter skill to convert the nanochat d12 source into LossCast-Bench
format. The conversion should produce three files in `losscast-dev/examples/nanochat_d12/converted/`:

1. **config.json** — extract all hyperparameters for a d12 run (depth=12, aspect_ratio=64)
2. **model.py** — standalone PyTorch model, no flash_attn or nanochat imports
3. **optimizer.py** — standalone MuonAdamW optimizer

IMPORTANT: The conversion **preserves all custom architectural features** exactly.
Do NOT simplify the model. Keep: ReLU² activation, QK norm with 1.2x sharpening,
value embeddings with gates on alternating layers, smear gate, backout lambda,
per-layer resid_lambdas and x0_lambdas, sliding window pattern, logit softcap.

Only remove/replace:
- `nanochat.*` imports → inline the code
- `flash_attn` → `F.scaled_dot_product_attention` (same math)
- Distributed code → single-device
- Inference paths (KV cache, generate) → not needed
- FP8 paths → not architecture

Simulate this training config:
- depth=12, aspect_ratio=64 → d_model=768, n_heads=6
- 8x H100 GPUs, device_batch_size=32, seq_len=2048
- matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.008
- warmup_steps=40, warmdown_ratio=0.65, final_lr_frac=0.05
- eval_every=250

### Step 3: Validate

After conversion, run validation:

```bash
# model.py should be runnable
python losscast-dev/examples/nanochat_d12/converted/model.py

# optimizer.py should show decreasing loss
python losscast-dev/examples/nanochat_d12/converted/optimizer.py

# Schema validation
python scripts/validate_contribution.py --run-dir losscast-dev/examples/nanochat_d12/converted/
```

### Step 4: Compare against expected output

Read `losscast-dev/examples/nanochat_d12/expected/config.json` and compare your
converted config against it. Flag any differences and explain whether they're
correct or need fixing.

### Step 5: Report

Summarize:
1. What went well in the conversion
2. What was tricky or required manual intervention
3. What the code-rewriter skill got wrong (if anything)
4. Suggestions for improving the skill or the schema
