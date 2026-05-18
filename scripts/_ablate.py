#!/usr/bin/env python3
"""Rapid-iteration ablation to find tricks that close the Marin OOD gap.

Runs each variant end-to-end (fit on train, predict on val, report per-source
final-loss MAE / RMSE / ρ) and prints a single CSV row per variant.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from losscast_bench.data import load_split
from losscast_bench.metrics.scoring import (
    compute_mae, compute_rmse, compute_spearman,
)
from losscast_bench.baselines.chinchilla import chinchilla_loss
from losscast_bench.baselines.xgboost_v2 import XGBoostPredictorV2
from losscast_bench.baselines.pca_xgboost import (
    PCAXGBoostPredictor, _chinchilla_on_grid, _resample_curve, _run_level_row,
)
from losscast_bench.schema import RunConfig, RunGroundTruth, RunPrediction


def source_of(cfg: RunConfig) -> str:
    return "marin" if cfg.data.eval_dataset == "c4_en" else "steplaw"


def persource_metrics(preds, val_cfgs, val_gt):
    cfg_by_id = {c.run_id: c for c in val_cfgs}
    gt_by_id = {g.run_id: g for g in val_gt}
    out = {}
    for key, src in [("c4_en", "marin"), ("train", "steplaw")]:
        ps, gs = [], []
        for p in preds:
            cfg = cfg_by_id.get(p.run_id); gt = gt_by_id.get(p.run_id)
            if cfg is None or gt is None or cfg.data.eval_dataset != key:
                continue
            common = sorted(set(p.predictions) & set(gt.losses))
            if not common:
                continue
            fs = max(common)
            ps.append(p.predictions[fs]); gs.append(gt.losses[fs])
        out[src] = {
            "MAE": compute_mae(ps, gs),
            "RMSE": compute_rmse(ps, gs),
            "rho": compute_spearman(ps, gs),
            "n": len(ps),
        }
    return out


def _print_row(name: str, m: dict):
    print(f"{name:<38} "
          f"M_MAE={m['marin']['MAE']:.4f}  M_ρ={m['marin']['rho']:.4f}   "
          f"S_MAE={m['steplaw']['MAE']:.4f}  S_ρ={m['steplaw']['rho']:.4f}",
          flush=True)


# ── Variant 1: per-source pca_xgboost ───────────────────────────────────────

def variant_pca_per_source(tr_c, tr_g, va_c):
    """Fit one PCAXGBoostPredictor per source."""
    gt_map = {g.run_id: g for g in tr_g}
    tr_by_src = {"marin": ([], []), "steplaw": ([], [])}
    for c in tr_c:
        g = gt_map.get(c.run_id)
        if g is None: continue
        s = source_of(c)
        tr_by_src[s][0].append(c); tr_by_src[s][1].append(g)

    preds_out = []
    for src in ("marin", "steplaw"):
        src_val = [c for c in va_c if source_of(c) == src]
        if not src_val:
            continue
        model = PCAXGBoostPredictor(K=16, n_components=8)
        model.fit(tr_by_src[src][0], tr_by_src[src][1])
        preds_out.extend(model.predict_batch(src_val))
    # Re-order to match va_c
    by_id = {p.run_id: p for p in preds_out}
    return [by_id.get(c.run_id, RunPrediction(run_id=c.run_id, predictions={}))
            for c in va_c]


# ── Variant 2: LightGBM replacement for XGBoost v2 ──────────────────────────

def variant_lightgbm_v2(tr_c, tr_g, va_c):
    """Same feature set as xgboost_v2 but LightGBM backend."""
    import lightgbm as lgb
    from losscast_bench.baselines.xgboost_baseline import (
        _numeric_config_features, _step_features, CATEGORICAL_FIELDS, _categoricals,
    )
    from losscast_bench.baselines.xgboost_v2 import (
        FEATURE_NAMES_V2, _extra_config_features, _step_phase,
    )

    cat_vocab: dict[str, dict[str, int]] = {}

    def row(c, step, fit):
        num = _numeric_config_features(c)
        num.update(_step_features(c, step))
        num.update(_extra_config_features(c))
        num["step_lr_phase"] = _step_phase(c, step)
        vec = [num[n] for n in FEATURE_NAMES_V2 if n in num]
        cats = _categoricals(c)
        for key in CATEGORICAL_FIELDS:
            v = cats[key]
            vocab = cat_vocab.setdefault(key, {})
            if fit:
                if v not in vocab: vocab[v] = len(vocab)
                vec.append(float(vocab[v]))
            else:
                vec.append(float(vocab.get(v, -1)))
        return vec

    gt_map = {g.run_id: g for g in tr_g}
    Xs, ys = [], []
    for c in tr_c:
        g = gt_map.get(c.run_id)
        if g is None: continue
        for step, loss in g.losses.items():
            Xs.append(row(c, step, fit=True))
            ys.append(loss)
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    model = lgb.LGBMRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        num_leaves=127, min_child_samples=10, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)

    out = []
    for c in va_c:
        rows = [row(c, s, fit=False) for s in c.eval_steps]
        X_va = np.asarray(rows, dtype=np.float32)
        preds = model.predict(X_va)
        out.append(RunPrediction(run_id=c.run_id, predictions={
            int(s): float(p) for s, p in zip(c.eval_steps, preds)
        }))
    return out


# ── Variant 3: CatBoost with native categorical handling ────────────────────

def variant_catboost(tr_c, tr_g, va_c):
    import catboost as cb
    from losscast_bench.baselines.xgboost_baseline import (
        _numeric_config_features, _step_features, _categoricals, CATEGORICAL_FIELDS,
    )
    from losscast_bench.baselines.xgboost_v2 import (
        FEATURE_NAMES_V2, _extra_config_features, _step_phase,
    )

    def row(c, step):
        num = _numeric_config_features(c)
        num.update(_step_features(c, step))
        num.update(_extra_config_features(c))
        num["step_lr_phase"] = _step_phase(c, step)
        vec = [num[n] for n in FEATURE_NAMES_V2 if n in num]
        cats = _categoricals(c)
        for key in CATEGORICAL_FIELDS:
            vec.append(str(cats[key]))  # CatBoost handles strings natively
        return vec

    numeric_len = len([n for n in FEATURE_NAMES_V2 if not n.endswith("_code")])
    cat_idx = list(range(numeric_len, numeric_len + len(CATEGORICAL_FIELDS)))

    gt_map = {g.run_id: g for g in tr_g}
    Xs, ys = [], []
    for c in tr_c:
        g = gt_map.get(c.run_id)
        if g is None: continue
        for step, loss in g.losses.items():
            Xs.append(row(c, step))
            ys.append(loss)

    model = cb.CatBoostRegressor(
        iterations=1000, depth=8, learning_rate=0.05,
        cat_features=cat_idx, random_seed=42,
        loss_function="RMSE", verbose=False,
    )
    model.fit(Xs, ys)

    out = []
    for c in va_c:
        rows = [row(c, s) for s in c.eval_steps]
        preds = model.predict(rows)
        out.append(RunPrediction(run_id=c.run_id, predictions={
            int(s): float(p) for s, p in zip(c.eval_steps, preds)
        }))
    return out


# ── Variant 4: residual-on-Chinchilla-refit (target is residual, not loss) ──

def variant_residual_xgboost(tr_c, tr_g, va_c):
    """Predict loss - chinchilla_refit_pred; add it back at inference."""
    from losscast_bench.baselines.chinchilla_refit import ChinchillaRefitPredictor
    import xgboost as xgb
    from losscast_bench.baselines.xgboost_baseline import (
        _numeric_config_features, _step_features, CATEGORICAL_FIELDS, _categoricals,
    )
    from losscast_bench.baselines.xgboost_v2 import (
        FEATURE_NAMES_V2, _extra_config_features, _step_phase,
    )

    chin = ChinchillaRefitPredictor.load("baselines/chinchilla_refit.json")
    chin_preds = {p.run_id: p for p in chin.predict_batch(tr_c + va_c)}

    cat_vocab = {}

    def row(c, step, fit):
        num = _numeric_config_features(c)
        num.update(_step_features(c, step))
        num.update(_extra_config_features(c))
        num["step_lr_phase"] = _step_phase(c, step)
        vec = [num[n] for n in FEATURE_NAMES_V2 if n in num]
        cats = _categoricals(c)
        for key in CATEGORICAL_FIELDS:
            v = cats[key]; vocab = cat_vocab.setdefault(key, {})
            if fit:
                if v not in vocab: vocab[v] = len(vocab)
                vec.append(float(vocab[v]))
            else:
                vec.append(float(vocab.get(v, -1)))
        return vec

    gt_map = {g.run_id: g for g in tr_g}
    Xs, ys = [], []
    for c in tr_c:
        g = gt_map.get(c.run_id)
        if g is None: continue
        cp = chin_preds.get(c.run_id)
        if cp is None: continue
        for step, loss in g.losses.items():
            if step not in cp.predictions: continue
            Xs.append(row(c, step, fit=True))
            ys.append(loss - cp.predictions[step])  # residual
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=5.0,
        reg_lambda=1.0, random_state=42, n_jobs=-1,
        objective="reg:squarederror", tree_method="hist",
    )
    model.fit(X, y)

    out = []
    for c in va_c:
        cp = chin_preds.get(c.run_id)
        rows = [row(c, s, fit=False) for s in c.eval_steps]
        residuals = model.predict(np.asarray(rows, dtype=np.float32))
        preds = {
            int(s): float(residuals[i] + (cp.predictions.get(s, 0.0) if cp else 0.0))
            for i, s in enumerate(c.eval_steps)
        }
        out.append(RunPrediction(run_id=c.run_id, predictions=preds))
    return out


# ── Variant 5: target-encoded optimizer name ────────────────────────────────

def variant_target_encoded(tr_c, tr_g, va_c):
    """Replace label-encoded optimizer name with mean-target encoding."""
    import xgboost as xgb
    from losscast_bench.baselines.xgboost_baseline import (
        _numeric_config_features, _step_features, CATEGORICAL_FIELDS, _categoricals,
    )
    from losscast_bench.baselines.xgboost_v2 import (
        FEATURE_NAMES_V2, _extra_config_features, _step_phase,
    )

    # Compute mean final-loss per optimizer on the train split (smoothed)
    gt_map = {g.run_id: g for g in tr_g}
    opt_losses: dict[str, list[float]] = {}
    src_opt_losses: dict[tuple[str, str], list[float]] = {}
    global_final = []
    for c in tr_c:
        g = gt_map.get(c.run_id)
        if g is None or not g.losses: continue
        fs = max(g.losses.keys()); fl = g.losses[fs]
        opt_losses.setdefault(c.optimizer.name, []).append(fl)
        src_opt_losses.setdefault((source_of(c), c.optimizer.name), []).append(fl)
        global_final.append(fl)
    gmean = float(np.mean(global_final))
    # Smoothed encoding: (n*mean + k*gmean) / (n + k) with k=20
    K = 20
    opt_enc = {
        k: (len(v) * np.mean(v) + K * gmean) / (len(v) + K)
        for k, v in opt_losses.items()
    }
    src_opt_enc = {
        k: (len(v) * np.mean(v) + K * gmean) / (len(v) + K)
        for k, v in src_opt_losses.items()
    }

    cat_vocab = {}

    def row(c, step, fit):
        num = _numeric_config_features(c)
        num.update(_step_features(c, step))
        num.update(_extra_config_features(c))
        num["step_lr_phase"] = _step_phase(c, step)
        vec = [num[n] for n in FEATURE_NAMES_V2 if n in num]

        # Target-encoded optimizer (global + per-source)
        opt = c.optimizer.name
        vec.append(opt_enc.get(opt, gmean))
        vec.append(src_opt_enc.get((source_of(c), opt), gmean))

        # Keep other categoricals label-encoded (fewer classes)
        cats = _categoricals(c)
        for key in CATEGORICAL_FIELDS:
            if key == "optimizer_name":
                continue
            v = cats[key]; vocab = cat_vocab.setdefault(key, {})
            if fit:
                if v not in vocab: vocab[v] = len(vocab)
                vec.append(float(vocab[v]))
            else:
                vec.append(float(vocab.get(v, -1)))
        return vec

    Xs, ys = [], []
    for c in tr_c:
        g = gt_map.get(c.run_id)
        if g is None: continue
        for step, loss in g.losses.items():
            Xs.append(row(c, step, fit=True))
            ys.append(loss)
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=5.0,
        reg_lambda=1.0, random_state=42, n_jobs=-1,
        objective="reg:squarederror", tree_method="hist",
    )
    model.fit(X, y)

    out = []
    for c in va_c:
        rows = [row(c, s, fit=False) for s in c.eval_steps]
        preds = model.predict(np.asarray(rows, dtype=np.float32))
        out.append(RunPrediction(run_id=c.run_id, predictions={
            int(s): float(p) for s, p in zip(c.eval_steps, preds)
        }))
    return out


# ── Variant 6: stacked blend (xgboost_v2 + pca_xgboost + chinchilla_refit) ──

def variant_stacked_blend(tr_c, tr_g, va_c):
    """Load already-trained models, predict, average their outputs.

    Simple uniform blend. We'd need out-of-fold predictions to properly learn
    blend weights, which is too expensive for a quick ablation — but a uniform
    average often captures most of the stacking gain.
    """
    from losscast_bench.baselines.xgboost_v2 import XGBoostPredictorV2
    from losscast_bench.baselines.pca_xgboost import PCAXGBoostPredictor
    from losscast_bench.baselines.chinchilla_refit import ChinchillaRefitPredictor

    p1 = XGBoostPredictorV2.load("baselines/xgboost_v2.pkl").predict_batch(va_c)
    p2 = PCAXGBoostPredictor.load("baselines/pca_xgboost.pkl").predict_batch(va_c)
    p3 = ChinchillaRefitPredictor.load("baselines/chinchilla_refit.json").predict_batch(va_c)

    by_id = {p.run_id: {"p1": p, "p2": None, "p3": None} for p in p1}
    for p in p2:
        if p.run_id in by_id: by_id[p.run_id]["p2"] = p
    for p in p3:
        if p.run_id in by_id: by_id[p.run_id]["p3"] = p

    out = []
    for c in va_c:
        b = by_id.get(c.run_id)
        if b is None or b["p1"] is None:
            out.append(RunPrediction(run_id=c.run_id, predictions={}))
            continue
        preds = {}
        all_steps = set(b["p1"].predictions)
        if b["p2"]: all_steps &= set(b["p2"].predictions)
        if b["p3"]: all_steps &= set(b["p3"].predictions)
        for s in sorted(all_steps):
            vals = [b["p1"].predictions[s]]
            if b["p2"]: vals.append(b["p2"].predictions[s])
            if b["p3"]: vals.append(b["p3"].predictions[s])
            preds[s] = sum(vals) / len(vals)
        out.append(RunPrediction(run_id=c.run_id, predictions=preds))
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    tr_c, tr_g = load_split("train")
    va_c, va_g = load_split("val")
    print(f"train={len(tr_c)}, val={len(va_c)}", flush=True)
    print(f"{'variant':<38} "
          f"{'Marin MAE':<10} {'Marin ρ':<10}   {'StepLaw MAE':<12} {'StepLaw ρ':<10}",
          flush=True)
    print("-" * 92, flush=True)

    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    runners = {
        "pca_per_source": variant_pca_per_source,
        "lightgbm_v2": variant_lightgbm_v2,
        "catboost": variant_catboost,
        "residual_xgb": variant_residual_xgboost,
        "target_enc": variant_target_encoded,
        "stacked_blend": variant_stacked_blend,
    }
    if which != "all":
        runners = {which: runners[which]}

    for name, fn in runners.items():
        try:
            preds = fn(tr_c, tr_g, va_c)
            m = persource_metrics(preds, va_c, va_g)
            _print_row(name, m)
        except Exception as e:
            print(f"{name}: FAILED — {e}", flush=True)
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
