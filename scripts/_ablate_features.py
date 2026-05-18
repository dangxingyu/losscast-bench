#!/usr/bin/env python3
"""Test feature-engineering ideas on Marin. Starts from xgboost_ultimate's
current best (65% CatBoost + 35% Marin-PCA, MAE=0.0157, ρ=0.9246) and adds
various feature groups to the CatBoost head. Reports Marin MAE/ρ per variant
plus CatBoost built-in feature importance on each added group.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import compute_mae, compute_rmse, compute_spearman
from losscast_bench.baselines.xgboost_baseline import (
    CATEGORICAL_FIELDS, _categoricals, _numeric_config_features, _step_features,
)
from losscast_bench.baselines.xgboost_v2 import (
    FEATURE_NAMES_V2, _extra_config_features, _step_phase,
)
from losscast_bench.baselines.pca_xgboost import PCAXGBoostPredictor
from losscast_bench.baselines.chinchilla import chinchilla_loss


def source_of(c):
    return "marin" if c.data.eval_dataset == "c4_en" else "steplaw"


def _num(x, default=0.0):
    if x is None: return default
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v): return default
        return v
    except (TypeError, ValueError):
        return default


def _safe_log(x, floor=1e-12):
    if x is None: return math.log(floor)
    return math.log(max(float(x), floor))


# ── Extra feature groups the user suggested ────────────────────────────────

def _group_compute(c, step):
    """1. Effective compute features."""
    m = c.model; d = c.data
    n_params = max(m.n_params_approx, 1)
    tokens_total = max(_num(d.tokens_total, 1.0), 1.0)
    total_steps = max(int(c.schedule.total_steps or 1), 1)
    frac = step / total_steps if step > 0 else 0.0
    tokens_seen = max(frac * tokens_total, 1.0)
    flops_token = 6.0 * n_params
    total_flops = flops_token * tokens_total
    flops_seen = flops_token * tokens_seen
    dp_size = max(int(c.dp_size or 1), 1)
    tp_size = max(int(c.tp_size or 1), 1)
    return {
        "log_flops_per_token": _safe_log(flops_token),
        "log_total_flops": _safe_log(total_flops),
        "log_flops_per_gpu": _safe_log(total_flops / (dp_size * tp_size)),
        "log_tokens_per_gpu": _safe_log(tokens_total / dp_size),
        "log_flops_seen": _safe_log(flops_seen),
    }


def _group_arch_ratio(c, step):
    """2. Architecture ratio features."""
    m = c.model
    n_params = max(m.n_params_approx, 1)
    n_layers = max(int(m.n_layers), 1)
    n_heads = max(int(m.n_heads or 1), 1)
    d_model = max(int(m.d_model), 1)
    head_dim = m.head_dim or (d_model // n_heads)
    d_ff = m.d_ff or 4 * d_model
    n_kv = m.n_kv_heads or n_heads
    vocab = max(int(m.vocab_size or 1), 1)
    attn_capacity = n_heads * head_dim * head_dim / d_model
    ff_expansion = d_ff / d_model
    kv_compression = n_kv / n_heads
    embedding_fraction = (vocab * d_model) / n_params
    return {
        "log_params_per_layer": _safe_log(n_params / n_layers),
        "log_attention_ratio": _safe_log(attn_capacity),
        "ff_expansion": ff_expansion,
        "kv_compression": kv_compression,
        "embedding_fraction": embedding_fraction,
        "log_head_dim": _safe_log(head_dim),
    }


def _group_opt_arch(c, step):
    """3. Optimizer-architecture interaction features."""
    m = c.model; d = c.data; o = c.optimizer
    n_params = max(m.n_params_approx, 1)
    lr = max(_num(o.lr, 1e-4), 1e-12)
    wd = _num(o.weight_decay, 0.0)
    batch_tokens = max(_num(d.batch_tokens, 1.0), 1.0)
    tokens_total = max(_num(d.tokens_total, 1.0), 1.0)
    log_np = _safe_log(n_params)
    return {
        "lr_over_log_params": lr / log_np if log_np != 0 else 0.0,
        "wd_lr_ratio": wd / lr,
        "log_batch_tokens_per_param": _safe_log(batch_tokens / n_params),
        "log_steps_per_epoch_equiv": _safe_log(tokens_total / batch_tokens),
    }


def _group_schedule_shape(c, step):
    """4. Schedule shape features."""
    s = c.schedule
    total_steps = max(int(s.total_steps or 1), 1)
    warmup = max(int(s.warmup_steps or 0), 0) / total_steps
    cooldown = max(int(s.cooldown_steps or 0), 0) / total_steps
    stable = max(1.0 - warmup - cooldown, 0.0)
    peak_lr = max(_num(c.optimizer.lr, 1e-4), 1e-12)
    final_lr = _num(s.final_lr_ratio, 0.0)
    # Cosine LR integral approximation: for each segment,
    #   warmup (linear ramp): avg_lr = 0.5 * peak_lr
    #   stable or cosine decay: avg_lr = peak_lr * (2/π) * (1+final)/2 ~= 0.318(1+final)
    #   fallback to WSD linear: avg_lr = 0.5 * peak_lr * (1+final)
    is_cosine = (s.lr_schedule or "").lower() in ("cosine",)
    avg_main = peak_lr * (2.0 / math.pi) * (1.0 + final_lr) / 2.0 if is_cosine \
               else peak_lr * 0.5 * (1.0 + final_lr)
    integral = peak_lr * 0.5 * warmup + avg_main * (stable + cooldown)
    total_lr_budget = integral * total_steps
    return {
        "warmup_fraction": warmup,
        "cooldown_fraction": cooldown,
        "stable_fraction": stable,
        "log_lr_integral_total": _safe_log(total_lr_budget),
        "log_lr_budget_per_param": _safe_log(total_lr_budget / max(c.model.n_params_approx, 1)),
    }


def _group_scaling_residual(c, step):
    """5. Chinchilla scaling features (pred, log(pred), pred^2)."""
    m = c.model; d = c.data
    n_params = max(m.n_params_approx, 1)
    tokens_total = max(_num(d.tokens_total, 1.0), 1.0)
    total_steps = max(int(c.schedule.total_steps or 1), 1)
    frac = step / total_steps if step > 0 else 0.0
    tokens_seen = max(frac * tokens_total, 1.0)
    if step == 0:
        pred = math.log(max(int(m.vocab_size or 2), 2))
        pred_final = math.log(max(int(m.vocab_size or 2), 2))
    else:
        pred = chinchilla_loss(n_params, tokens_seen)
        pred_final = chinchilla_loss(n_params, tokens_total)
    return {
        "chinchilla_log_pred": math.log(max(pred, 1e-6)),
        "chinchilla_pred_sq": pred * pred,
        "chinchilla_pred_final": pred_final,
        "chinchilla_pred_gap": pred - pred_final,  # how far from final is this step
    }


def _group_dataset(c, step):
    """6. Dataset-specific binary flags."""
    name = (c.data.dataset or "").lower()
    return {
        "is_fineweb": 1.0 if "fineweb" in name else 0.0,
        "is_c4": 1.0 if "c4" in name else 0.0,
        "is_dolma": 1.0 if "dolma" in name else 0.0,
        "is_openwebtext": 1.0 if "openweb" in name else 0.0,
    }


def _group_diminishing(c, step):
    """7. Diminishing-returns / Chinchilla-ratio features."""
    m = c.model; d = c.data
    n_params = max(m.n_params_approx, 1)
    tokens_total = max(_num(d.tokens_total, 1.0), 1.0)
    ratio = tokens_total / n_params
    return {
        "log_tokens_per_param_alt": _safe_log(ratio),
        "is_overtrained_20x": 1.0 if ratio > 20.0 else 0.0,
        "is_overtrained_50x": 1.0 if ratio > 50.0 else 0.0,
        "log_compute_efficiency": 0.5 * _safe_log(n_params) + 0.5 * _safe_log(tokens_total),
    }


GROUPS = {
    "compute": _group_compute,
    "arch_ratio": _group_arch_ratio,
    "opt_arch": _group_opt_arch,
    "schedule_shape": _group_schedule_shape,
    "scaling_residual": _group_scaling_residual,
    "dataset": _group_dataset,
    "diminishing": _group_diminishing,
}


# ── Row builder with configurable extra groups ─────────────────────────────

def cat_row(c, step, extra_groups=()):
    """Base v2 row + optional extra feature groups appended to numeric section."""
    num = _numeric_config_features(c)
    num.update(_step_features(c, step))
    num.update(_extra_config_features(c))
    num["step_lr_phase"] = _step_phase(c, step)
    vec = [num[n] for n in FEATURE_NAMES_V2 if n in num]
    extra_names = []
    for g in extra_groups:
        gf = GROUPS[g](c, step)
        for k, v in gf.items():
            vec.append(float(v))
            extra_names.append(k)
    cats = _categoricals(c)
    for k in CATEGORICAL_FIELDS:
        vec.append(str(cats[k]))
    return vec, extra_names


def base_numeric_len():
    return len([n for n in FEATURE_NAMES_V2 if not n.endswith("_code")])


def build_matrix(configs, gt, extra_groups):
    Xs, ys, extra_names = [], [], []
    gt_map = {g.run_id: g for g in gt}
    for c in configs:
        g = gt_map.get(c.run_id)
        if g is None: continue
        for s, l in g.losses.items():
            row, extras = cat_row(c, s, extra_groups)
            Xs.append(row); ys.append(l)
            if not extra_names and extras:
                extra_names = extras
    return Xs, ys, extra_names


def marin_metrics_final(preds_by_id, va_gt, va_cfgs):
    gt_by_id = {g.run_id: g for g in va_gt}
    cfg_by_id = {c.run_id: c for c in va_cfgs}
    ps, gs = [], []
    for rid, fp in preds_by_id.items():
        c = cfg_by_id.get(rid); g = gt_by_id.get(rid)
        if c is None or g is None or c.data.eval_dataset != "c4_en": continue
        common = sorted(set(fp) & set(g.losses))
        if not common: continue
        fs = max(common)
        ps.append(fp[fs]); gs.append(g.losses[fs])
    return compute_mae(ps, gs), compute_rmse(ps, gs), compute_spearman(ps, gs), len(ps)


def run_variant(tag, tr_c, tr_g, va_c, va_g, extra_groups, pca_preds_marin, marin_val,
                w_cat=0.65):
    """Fit CatBoost with given groups, blend with Marin PCA, report Marin metrics."""
    import catboost as cb
    Xs, ys, extra_names = build_matrix(tr_c, tr_g, extra_groups)

    numeric_len = base_numeric_len() + len(extra_names)
    cat_idx = list(range(numeric_len, numeric_len + len(CATEGORICAL_FIELDS)))

    t0 = time.time()
    model = cb.CatBoostRegressor(
        iterations=1200, depth=8, learning_rate=0.05,
        cat_features=cat_idx, loss_function="RMSE",
        random_seed=42, verbose=False, allow_writing_files=False,
    )
    model.fit(Xs, ys)
    fit_dt = time.time() - t0

    # Predict Marin val
    cat_preds = {}
    for c in marin_val:
        rows = [cat_row(c, s, extra_groups)[0] for s in c.eval_steps]
        p = model.predict(rows)
        cat_preds[c.run_id] = dict(zip((int(s) for s in c.eval_steps), (float(v) for v in p)))

    # Blend
    final = {}
    for c in marin_val:
        cp = cat_preds.get(c.run_id, {})
        pp = pca_preds_marin.get(c.run_id, {})
        keys = set(cp) & set(pp)
        final[c.run_id] = {int(s): w_cat * cp[s] + (1 - w_cat) * pp[s] for s in keys}
    mae, rmse, rho, n = marin_metrics_final(final, va_g, va_c)

    # Feature importance on new groups
    fi = model.get_feature_importance(type="PredictionValuesChange")
    base_len = base_numeric_len()
    extras_importance = []
    for i, name in enumerate(extra_names):
        extras_importance.append((name, float(fi[base_len + i])))
    extras_importance.sort(key=lambda x: -x[1])

    return {
        "tag": tag, "groups": list(extra_groups),
        "mae": float(mae), "rmse": float(rmse), "rho": float(rho), "n": n,
        "fit_seconds": fit_dt,
        "extras_importance": extras_importance,
    }


def main():
    tr_c, tr_g = load_split("train")
    va_c, va_g = load_split("val")
    marin_val = [c for c in va_c if source_of(c) == "marin"]
    print(f"train={len(tr_c)}, marin val={len(marin_val)}", flush=True)

    # Fit Marin PCA once (unchanged regardless of variant).
    gmap_tr = {g.run_id: g for g in tr_g}
    marin_tr = [c for c in tr_c if source_of(c) == "marin"]
    marin_tr_gt = [gmap_tr[c.run_id] for c in marin_tr if c.run_id in gmap_tr]
    pca = PCAXGBoostPredictor(K=16, n_components=8).fit(marin_tr, marin_tr_gt)
    pca_preds = {p.run_id: p.predictions for p in pca.predict_batch(marin_val)}

    results = []
    hdr = f"{'variant':<46}  {'MAE':<8} {'RMSE':<8} {'ρ':<8}  {'Δ MAE':<8} {'fit(s)':<7}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    # Baseline: current ultimate (no extras)
    base = run_variant("baseline (no extras)", tr_c, tr_g, va_c, va_g, (), pca_preds, marin_val)
    results.append(base)
    print(f"{base['tag']:<46}  {base['mae']:.4f}  {base['rmse']:.4f}  {base['rho']:.4f}  "
          f"{0.0:+.4f}  {base['fit_seconds']:.0f}", flush=True)
    base_mae = base["mae"]

    # Each group alone
    for name in GROUPS:
        r = run_variant(f"+{name}", tr_c, tr_g, va_c, va_g, (name,), pca_preds, marin_val)
        results.append(r)
        delta = r["mae"] - base_mae
        print(f"{r['tag']:<46}  {r['mae']:.4f}  {r['rmse']:.4f}  {r['rho']:.4f}  "
              f"{delta:+.4f}  {r['fit_seconds']:.0f}", flush=True)

    # All groups
    r_all = run_variant("+all 7 groups", tr_c, tr_g, va_c, va_g, tuple(GROUPS), pca_preds, marin_val)
    results.append(r_all)
    delta = r_all["mae"] - base_mae
    print(f"{r_all['tag']:<46}  {r_all['mae']:.4f}  {r_all['rmse']:.4f}  {r_all['rho']:.4f}  "
          f"{delta:+.4f}  {r_all['fit_seconds']:.0f}", flush=True)

    # Print feature importance for the all-groups variant
    print("\nFeature importance (PredictionValuesChange, top 15):")
    for name, score in r_all["extras_importance"][:15]:
        print(f"  {score:6.2f}  {name}")

    # Dump full results
    Path("baselines/feature_ablation.json").write_text(json.dumps(results, indent=2))

    # Summary
    best = min(results, key=lambda r: r["mae"])
    print(f"\nBest MAE: {best['tag']} → MAE={best['mae']:.4f}, ρ={best['rho']:.4f}")
    best_rho = max(results, key=lambda r: r["rho"])
    print(f"Best ρ:   {best_rho['tag']} → MAE={best_rho['mae']:.4f}, ρ={best_rho['rho']:.4f}")


if __name__ == "__main__":
    main()
