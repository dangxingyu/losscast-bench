---
name: code-rewriter
description: |
  Convert arbitrary LLM training code into LossCast-Bench standard format. Use this skill whenever the user wants to: extract a model architecture from a training repo and rewrite it as a standalone model.py; extract optimizer logic into optimizer.py; generate a config.json from training hyperparameters; convert code from repos like nanochat, nanogpt, llm.c, or any custom training codebase into the benchmark's template format. Trigger on phrases like "convert this code", "rewrite for losscast", "extract the model", "make a config from this", or when the user provides training code and wants it formatted for benchmark contribution.
---

# Code Rewriter for LossCast-Bench

You are converting training code from an external source into LossCast-Bench's standard three-file format: `model.py`, `optimizer.py` (if needed), and `config.json`. The goal is to make any training recipe machine-readable for the benchmark while preserving exact numerical behavior.

## The Three Output Files

### 1. `model.py` — Architecture + Initialization

Read the template at `references/model_template.py` and the complete example at `references/example_model.py` before writing anything. Your output must follow this exact structure:

```
Components (RMSNorm, RoPE, Attention, FFN, etc.)
    ↓
class Model(nn.Module)
    __init__(self, config: dict)    — build the architecture
    _init_weights(self, module)     — exact weight initialization
    forward(self, input_ids)        — input_ids → logits
    ↓
MODEL_CONFIG = { ... }
    ↓
Verification block (copy from template, do not modify)
```

Key rules:
- **Preserve ALL architectural features exactly.** Custom activations (ReLU², SwiGLU), custom attention patterns (sliding window, QK norm), residual stream modifications (value embeddings, smear gates, backout), per-layer scalars — all of these ARE the architecture and must be kept. The goal is to decouple the model from the training framework, not to simplify it.
- **Pure PyTorch only.** No HuggingFace, no flash-attn, no xformers, no triton kernels. Replace any such dependencies with vanilla PyTorch equivalents. Flash attention → `F.scaled_dot_product_attention` (same math, portable). Custom CUDA kernels → pure Python/PyTorch math.
- **Self-contained.** Every component (RMSNorm, RoPE, GQA, SwiGLU, etc.) must be defined in this single file.
- **Remove only framework coupling.** Remove: framework imports, distributed code, inference paths (KV cache, generate), FP8/mixed-precision paths, dropout. Keep: everything that affects the forward pass computation.
- **Exact init matters.** The `_init_weights` method must reproduce the source code's initialization exactly. Pay close attention to: per-layer scaling (e.g., `1/sqrt(2*n_layers)` on residual projections), special treatment of embeddings vs linear layers, any muP-style scaling, and which layers get zero-init.
- **Mark residual projections.** Tag output projections with `module._is_residual_proj = True` so the init can apply residual scaling to them specifically.
- **`MODEL_CONFIG` dict** at the bottom must contain all fields from the JSON schema's `model` section, with values matching the source code.

### 2. `optimizer.py` — Custom Optimizer (only if non-standard)

Read the template at `references/optimizer_template.py`. Only create this file if the source uses an optimizer that isn't fully described by `{name, lr, weight_decay, beta1, beta2, eps, grad_clip}` — i.e., anything beyond standard AdamW/Adam/SGD/Lion.

Common cases that need optimizer.py:
- **Muon** — orthogonalized momentum updates on weight matrices
- **SOAP** — periodic preconditioner updates
- **Hybrid optimizers** (e.g., Muon for matrices + AdamW for embeddings)
- **Schedule-Free** optimizers
- Any optimizer with periodic operations or custom update rules

Structure:
```
class CustomOptimizer(Optimizer)
    __init__(self, params, lr, **kwargs)
    step(self, closure=None, step_idx: int = 0)   ← receives step index!
    ↓
OPTIMIZER_DEFAULTS = { ... }
    ↓
Verification block (copy from template, do not modify)
```

The `step_idx` parameter is the key design choice: all periodic logic (preconditioner updates, warm-up inside the optimizer, etc.) goes inside `step()` using `step_idx`, keeping the training loop standardized.

For **hybrid optimizers** (different optimizers for different parameter groups), implement them as a single `CustomOptimizer` class that internally dispatches based on parameter shape or metadata. The constructor should accept all parameter groups and route them appropriately.

### 3. `config.json` — Structured Hyperparameters

Read `references/config_template.json` for the full schema. Extract every field from the source code:

**model**: `arch`, `n_layers`, `d_model`, `n_heads`, `head_dim`, `d_ff`, `vocab_size`, `activation`, `norm_type`, `rope`, `n_kv_heads`, `tied_embeddings`

**optimizer**: `name`, `lr`, `weight_decay`, `beta1`, `beta2`, `eps`, `grad_clip`, `mup`
- For hybrid optimizers, use a descriptive name like `"muon_adamw"` and put the primary (matrix) LR as `lr`.

**data**: `dataset`, `tokenizer`, `seq_len`, `batch_tokens`, `tokens_total`, `mix`, `eval_dataset`
- `batch_tokens` = per-device batch size × seq_len × dp_size (total tokens per step)

**schedule**: `lr_schedule`, `warmup_steps`, `total_steps`, `cooldown_steps`, `final_lr_ratio`
- `final_lr_ratio` = final_lr / peak_lr (e.g., if cosine decays to 0, this is 0.0)

**top-level**: `run_id`, `eval_interval`, `hardware`, `precision`, `dp_size`, `tp_size`

## Conversion Process

Follow this order:

1. **Read and understand the source code.** Identify: model class, all subcomponents, init logic, optimizer setup, LR schedule, data config, batch size computation. Don't start writing until you have a clear picture.

2. **Identify what's standard vs custom.** Standard AdamW with cosine schedule? No optimizer.py needed. Muon + AdamW hybrid with linear weight decay? That needs optimizer.py.

3. **Write `config.json` first.** This forces you to extract all numeric hyperparameters before getting into the code. Cross-check values against the source: is that LR the peak LR? Is batch size in tokens or samples? How many warmup steps?

4. **Write `model.py`.** Start with the components (norm, position encoding, attention, FFN), then the block, then the full model. Replace any non-PyTorch dependencies. Preserve the exact same tensor shapes and operations — don't "improve" the architecture, reproduce it faithfully.

5. **Write `optimizer.py` if needed.** Focus on getting the update rule exactly right. For hybrid optimizers, test that the parameter routing matches the source.

6. **Verify.** Run `python model.py` and `python optimizer.py` (if present). Both have built-in verification that checks construction, forward pass shapes, numerical stability, and (for optimizers) that loss decreases over 200 steps.

## Source-Specific Guidance

### HuggingFace `transformers` Models

HuggingFace's `modeling_*.py` files are well-structured PyTorch underneath, but wrapped in several layers of abstraction that must be stripped. Here's the mapping:

**Model conversion (`PreTrainedModel` → `Model(nn.Module)`):**
- Ignore the `PreTrainedModel` / `PretrainedConfig` base classes entirely. Read only the model-specific code (e.g., `LlamaForCausalLM`, `LlamaModel`, `LlamaAttention`, `LlamaMLP`).
- The HF config class (e.g., `LlamaConfig`) maps directly to our `MODEL_CONFIG` dict. Extract: `hidden_size` → `d_model`, `num_hidden_layers` → `n_layers`, `num_attention_heads` → `n_heads`, `num_key_value_heads` → `n_kv_heads`, `intermediate_size` → `d_ff`, `vocab_size` → `vocab_size`, `rms_norm_eps` → (embed in norm implementation), `rope_theta` → (embed in RoPE implementation), `max_position_embeddings` → `seq_len`.
- **Strip these HF-only patterns:**
  - `past_key_value` / KV cache handling in attention → remove, keep only the non-cached forward path
  - `output_attentions`, `output_hidden_states`, `return_dict` flags → remove, always return logits only
  - `CausalLMOutputWithPast` and similar output dataclasses → return raw logits tensor
  - `gradient_checkpointing` decorators → remove
  - `_prepare_decoder_attention_mask` / `_update_causal_mask` → use `is_causal=True` in `F.scaled_dot_product_attention` or manual causal mask
  - `_no_split_modules`, `_tied_weights_keys`, `_skip_keys_device_placement` → remove
- **Weight init**: HF splits init across `_init_weights(module)` (on the model class) + `post_init()` (on the base class, applies `_init_weights` recursively and optionally ties weights). Merge these into our single `_init_weights(self, module)` method. Check for residual scaling — HF often applies `1/sqrt(n_layers)` or similar to attention output projections.
- **Weight tying**: If `config.tie_word_embeddings` is True, the input embedding and LM head share weights. Set `tied_embeddings: true` in config.json and implement via `self.lm_head.weight = self.embed_tokens.weight`.

**Trainer / training args → `config.json`:**
- HF `TrainingArguments` field mappings:
  - `learning_rate` → `optimizer.lr`
  - `weight_decay` → `optimizer.weight_decay`
  - `adam_beta1`, `adam_beta2`, `adam_epsilon` → `optimizer.beta1`, `optimizer.beta2`, `optimizer.eps`
  - `max_grad_norm` → `optimizer.grad_clip`
  - `lr_scheduler_type` → `schedule.lr_schedule` (map: `"cosine"` → `"cosine"`, `"linear"` → `"linear"`, `"cosine_with_restarts"` → custom)
  - `warmup_steps` or `warmup_ratio` → `schedule.warmup_steps` (convert ratio to steps if needed: `warmup_ratio × total_steps`)
  - `max_steps` → `schedule.total_steps`
  - `per_device_train_batch_size × gradient_accumulation_steps × seq_len × num_gpus` → `data.batch_tokens`
  - `bf16` / `fp16` → `precision`
- HF's default optimizer is AdamW with `(beta1=0.9, beta2=0.999, eps=1e-8)` — only write optimizer.py if the training code overrides this with a custom optimizer class.

**Common HF gotchas:**
- HF models often have `position_ids` as an explicit input to forward(). Our model.py computes positions internally — generate `position_ids = torch.arange(seq_len)` inside forward.
- Some HF models apply RoPE to Q and K *after* splitting heads, some before. Read the source carefully.
- `LlamaRMSNorm` uses `variance_epsilon` from config, but `T5LayerNorm` uses a different default. Always check the actual epsilon value.
- HF's `apply_rotary_pos_emb` may have a different convention (cos/sin interleaving vs half-half) than what you'd write from scratch. Match the original exactly.

### Nanochat / NanoGPT-style Repos

Single-file repos with minimal abstraction. Key things to watch for:
- Model config is often a `@dataclass` (e.g., `GPTConfig`) rather than a dict — flatten it into `MODEL_CONFIG`.
- Optimizer setup may live in `model.setup_optimizer()` or the training script — check both.
- Look for per-layer scaling (nanochat has `smear`/`backout` and per-layer residual scalars).
- Custom CUDA kernels or triton ops → replace with PyTorch equivalents.
- Hybrid optimizers (Muon+AdamW) need full `optimizer.py` implementation.

## Common Pitfalls

- **Flash attention removal.** When replacing flash attention, make sure to keep the causal mask. Use `F.scaled_dot_product_attention(q, k, v, is_causal=True)` or manual masking.
- **GQA/MQA attention.** Source code may use grouped-query attention. Make sure K and V heads are expanded (repeated) to match Q heads, or handle the reshape correctly.
- **Vocab size from tokenizer.** The vocab size might not be explicitly in the config — it may come from a tokenizer file or be hardcoded. Check carefully.
- **Batch size confusion.** `batch_tokens` in our schema is total tokens per step. Source code often has `batch_size` (in samples) × `seq_len` = batch_tokens. Watch for gradient accumulation multiplying this further.
- **WSD schedule.** Warmup-Stable-Decay uses `cooldown_steps` for the decay phase. `final_lr_ratio` should be 0.0 for most WSD schedules.
- **muP flag.** Set `mup: true` only if the source explicitly uses maximal update parameterization or spectral scaling. Standard init ≠ muP.
- **Weight decay schedule.** If weight decay decays over training (e.g., linear to 0 in nanochat), this is a custom optimizer behavior — capture it in optimizer.py, and put the initial value in the JSON config.

## Verification

After conversion, verify correctness:

### 1. Built-in checks (always run)

The model.py and optimizer.py files include verification blocks that run via `python model.py` / `python optimizer.py`. These check:
- model.py: forward pass produces correct output shape, no NaN/Inf
- optimizer.py: loss decreases over 200 steps, no NaN

### 2. Schema validation

```bash
python scripts/validate_contribution.py --run-dir {OUTPUT_DIR}
```

### 3. Manual equivalence check (when original code is available)

If you have access to the original model, verify numerical equivalence:

```python
import torch

# Load both models with same weights
original_model = ...  # from source codebase
converted_model = Model(MODEL_CONFIG)
# Copy weights (match by shape if names differ)
# ...

# Compare forward pass
x = torch.randint(0, vocab_size, (2, 128))
with torch.no_grad():
    orig_out = original_model(x)
    conv_out = converted_model(x)

print(f"Max diff: {(orig_out - conv_out).abs().max().item():.2e}")
# Should be < 1e-5 for float32, < 1e-2 for bfloat16
```

**Note:** For nanochat runs, code-rewriting is NOT needed — copy raw `gpt.py` and `optim.py`
directly. Use the schema-creator skill instead, which calls `scripts/nanochat_to_bench.py`.

### What mismatches mean

- **Forward mismatch**: the architecture differs. Check: attention implementation, activation function, normalization, position encodings, any features you may have simplified or omitted.
- **Backward mismatch but forward matches**: likely a detach/no_grad issue, or a difference in how a custom operation computes gradients.
- **Optimizer mismatch**: the update rule differs. Check: weight decay application order, momentum formula, learning rate scaling, any periodic operations.

Some mismatches are expected and acceptable:
- Nanochat's "smear", "backout", and "value embeddings" are complex features. If you simplified them out, forward/backward won't match exactly — but the converted model should still be a valid standalone architecture.
- Small floating-point differences (1e-6 range) from reordering operations are normal.
