# Test Case: nanochat d24 (speedrun)

Full nanochat speedrun model (24 layers, 1536 dim) — the flagship config from karpathy/nanochat.

## Source
Same source files as d12 — nanochat uses a single codebase parameterized by `--depth`.

## Simulated Training Config
From `runs/speedrun.sh`:

```
depth=24, aspect_ratio=64 → d_model=1536, n_heads=12, n_kv_heads=12
batch_size=16 per device × 8 GPUs × 2048 seq_len = 262144 tokens/step
total_steps ~= 2B chars / (262144 tokens/step) ≈ 3800 steps
matrix_lr=0.02 (Muon), weight_decay=0.28
warmup=40 steps, warmdown_ratio=0.65
eval_every=250 steps
```

## Key Differences from d12
- Larger model (1536 vs 768 dim)
- Different batch size (16 vs 32 per device, scaled by depth)
- More transformer blocks → more value embedding layers
- LR scaling: AdamW params scale by (d_model/768)^-0.5
