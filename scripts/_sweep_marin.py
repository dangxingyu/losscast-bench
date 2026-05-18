#!/usr/bin/env python3
"""Thorough Marin tuning: blend ratio sweep, CatBoost Optuna, PCA components,
bagging, loss functions. Reports best config reached on the Marin OOD val.

Honest caveat on Optuna: tuning CatBoost hyperparameters directly against
Marin OOD val is technically overfit-to-val. We tune against a *pseudo-OOD*
holdout inside train (top 20% of Marin train runs by n_params held out) —
that mimics the real ID→OOD scale jump better than a random holdout.
The final sweep shows Marin OOD val numbers as the selection metric but
flags which come from a pseudo-OOD tune vs a direct val tune.
"""

from __future__ import annotations

import json
import math
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
from losscast_bench.schema import RunPrediction


def source_of(c):
    return "marin" if c.data.eval_dataset == "c4_en" else "steplaw"


def cat_row(c, step):
    num = _numeric_config_features(c); num.update(_step_features(c, step))
    num.update(_extra_config_features(c)); num["step_lr_phase"] = _step_phase(c, step)
    vec = [num[n] for n in FEATURE_NAMES_V2 if n in num]
    cats = _categoricals(c)
    for k in CATEGORICAL_FIELDS:
        vec.append(str(cats[k]))
    return vec


NUMERIC_LEN = len([n for n in FEATURE_NAMES_V2 if not n.endswith("_code")])
CAT_IDX = list(range(NUMERIC_LEN, NUMERIC_LEN + len(CATEGORICAL_FIELDS)))


def marin_metrics(final_preds_by_id, va_gt, va_cfgs):
    gt_by_id = {g.run_id: g for g in va_gt}
    cfg_by_id = {c.run_id: c for c in va_cfgs}
    ps, gs = [], []
    for rid, fp in final_preds_by_id.items():
        c = cfg_by_id.get(rid); g = gt_by_id.get(rid)
        if c is None or g is None or c.data.eval_dataset != "c4_en": continue
        common = sorted(set(fp) & set(g.losses))
        if not common: continue
        fs = max(common)
        ps.append(fp[fs]); gs.append(g.losses[fs])
    return compute_mae(ps, gs), compute_rmse(ps, gs), compute_spearman(ps, gs), len(ps)


def build_full_cat_matrix(configs, gt):
    Xs, ys = [], []
    gt_map = {g.run_id: g for g in gt}
    for c in configs:
        g = gt_map.get(c.run_id)
        if g is None: continue
        for s, l in g.losses.items():
            Xs.append(cat_row(c, s)); ys.append(l)
    return Xs, ys


def fit_catboost(Xs, ys, params=None, seed=42):
    import catboost as cb
    p = dict(iterations=1200, depth=8, learning_rate=0.05,
             cat_features=CAT_IDX, loss_function="RMSE",
             random_seed=seed, verbose=False, allow_writing_files=False)
    if params:
        # Avoid duplicate keys
        for k, v in params.items():
            p[k] = v
    return cb.CatBoostRegressor(**p).fit(Xs, ys)


def predict_marin(model, marin_val):
    return {
        c.run_id: dict(zip(
            (int(s) for s in c.eval_steps),
            (float(v) for v in model.predict([cat_row(c, s) for s in c.eval_steps])),
        ))
        for c in marin_val
    }


def main():
    tr_c, tr_g = load_split("train")
    va_c, va_g = load_split("val")
    marin_val = [c for c in va_c if source_of(c) == "marin"]
    print(f"train={len(tr_c)}, marin val={len(marin_val)}", flush=True)

    results = []

    def report(tag, m, dt=None):
        mae, rmse, rho, n = m
        line = f"{tag:<44}  MAE={mae:.4f}  RMSE={rmse:.4f}  ρ={rho:.4f}  n={n}"
        if dt is not None: line += f"  ({dt:.0f}s)"
        print(line, flush=True)
        results.append(dict(tag=tag, mae=float(mae), rmse=float(rmse), rho=float(rho), n=n))

    # ── Pass 1: baseline + 5% blend sweep ───────────────────────────────────
    t0 = time.time()
    print("\n[1/5] Baseline CatBoost + PCA blend sweep (5% granularity)", flush=True)
    Xs_all, ys_all = build_full_cat_matrix(tr_c, tr_g)
    cat = fit_catboost(Xs_all, ys_all)
    cat_preds = predict_marin(cat, marin_val)

    # Marin-only PCA
    gmap_tr = {g.run_id: g for g in tr_g}
    marin_tr = [c for c in tr_c if source_of(c) == "marin"]
    marin_tr_gt = [gmap_tr[c.run_id] for c in marin_tr if c.run_id in gmap_tr]
    pca = PCAXGBoostPredictor(K=16, n_components=8).fit(marin_tr, marin_tr_gt)
    pca_preds_raw = pca.predict_batch(marin_val)
    pca_preds = {p.run_id: p.predictions for p in pca_preds_raw}

    best_blend = (1.0, (0, 0, 0, 0))
    for w_cat_pct in range(0, 105, 5):
        w_cat = w_cat_pct / 100.0
        w_pca = 1.0 - w_cat
        final_by_id = {}
        for c in marin_val:
            cp = cat_preds.get(c.run_id, {}); pp = pca_preds.get(c.run_id, {})
            keys = set(cp) & set(pp)
            final_by_id[c.run_id] = {int(s): w_cat * cp[s] + w_pca * pp[s] for s in keys}
        m = marin_metrics(final_by_id, va_g, va_c)
        report(f"baseline_blend  w_cat={w_cat:.2f}", m)
        if m[0] < best_blend[0]:
            best_blend = (m[0], (w_cat, *m))
    print(f"Blend sweep took {time.time()-t0:.0f}s. Best: w_cat={best_blend[1][0]:.2f} MAE={best_blend[0]:.4f}",
          flush=True)

    # ── Pass 2: CatBoost hyperparameter sweep (Optuna, pseudo-OOD objective) ─
    print("\n[2/5] Optuna CatBoost tune (pseudo-OOD holdout: top-20% Marin train by params)", flush=True)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    marin_tr_sorted = sorted(marin_tr, key=lambda c: -c.model.n_params_approx)
    n_ho = max(1, int(len(marin_tr_sorted) * 0.20))
    ho_ids = set(c.run_id for c in marin_tr_sorted[:n_ho])
    sub_tr = [c for c in tr_c if c.run_id not in ho_ids]
    sub_Xs, sub_ys = build_full_cat_matrix(sub_tr, tr_g)

    ho_rows, ho_truths = [], []
    for c in marin_tr_sorted[:n_ho]:
        g = gmap_tr.get(c.run_id)
        if g is None: continue
        fs = max(g.losses.keys())
        ho_rows.append(cat_row(c, fs))
        ho_truths.append(g.losses[fs])

    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 500, 3000, step=100),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 3.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        )
        m = fit_catboost(sub_Xs, sub_ys, params=params)
        preds = m.predict(ho_rows)
        return compute_mae(list(preds), ho_truths)

    t0 = time.time()
    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=80, show_progress_bar=False)
    print(f"Optuna: 80 trials in {time.time()-t0:.0f}s  best holdout MAE={study.best_value:.4f}",
          flush=True)
    print(f"Best params: {study.best_params}", flush=True)

    # Refit on full train, eval on val, sweep blend ratio
    cat_tuned = fit_catboost(Xs_all, ys_all, params=study.best_params)
    cat_tuned_preds = predict_marin(cat_tuned, marin_val)

    for w_cat_pct in range(0, 105, 5):
        w_cat = w_cat_pct / 100.0; w_pca = 1.0 - w_cat
        final_by_id = {}
        for c in marin_val:
            cp = cat_tuned_preds.get(c.run_id, {}); pp = pca_preds.get(c.run_id, {})
            keys = set(cp) & set(pp)
            final_by_id[c.run_id] = {int(s): w_cat * cp[s] + w_pca * pp[s] for s in keys}
        m = marin_metrics(final_by_id, va_g, va_c)
        report(f"optuna_tuned   w_cat={w_cat:.2f}", m)

    # ── Pass 3: PCA component sweep (blended with baseline CatBoost) ────────
    print("\n[3/5] PCA n_components sweep (paired with baseline CatBoost at blend=0.6)", flush=True)
    for n_comp in (4, 8, 12, 16, 20):
        t0 = time.time()
        p = PCAXGBoostPredictor(K=16, n_components=n_comp).fit(marin_tr, marin_tr_gt)
        pp_preds = {pred.run_id: pred.predictions for pred in p.predict_batch(marin_val)}
        for w_cat_pct in (55, 60, 65, 70):
            w_cat = w_cat_pct / 100.0; w_pca = 1.0 - w_cat
            final_by_id = {}
            for c in marin_val:
                cp = cat_preds.get(c.run_id, {}); pp = pp_preds.get(c.run_id, {})
                keys = set(cp) & set(pp)
                final_by_id[c.run_id] = {int(s): w_cat * cp[s] + w_pca * pp[s] for s in keys}
            m = marin_metrics(final_by_id, va_g, va_c)
            report(f"pca_n={n_comp:2d} w_cat={w_cat:.2f}", m)
        print(f"  n_comp={n_comp} took {time.time()-t0:.0f}s", flush=True)

    # ── Pass 4: bagging on top of best blend ────────────────────────────────
    print("\n[4/5] Bagging N={3,5,7} on top of best-blend (w_cat=0.6)", flush=True)
    # Reuse pca (K=16, n_components=8). Different seeds of CatBoost.
    for n_bags in (3, 5, 7):
        t0 = time.time()
        cat_preds_bag = None
        for seed in range(n_bags):
            m = fit_catboost(Xs_all, ys_all, seed=42 + seed)
            p = predict_marin(m, marin_val)
            if cat_preds_bag is None:
                cat_preds_bag = {rid: {s: v / n_bags for s, v in d.items()} for rid, d in p.items()}
            else:
                for rid, d in p.items():
                    for s, v in d.items():
                        cat_preds_bag[rid][s] += v / n_bags
        for w_cat_pct in (55, 60, 65):
            w_cat = w_cat_pct / 100.0; w_pca = 1.0 - w_cat
            final_by_id = {}
            for c in marin_val:
                cp = cat_preds_bag.get(c.run_id, {}); pp = pca_preds.get(c.run_id, {})
                keys = set(cp) & set(pp)
                final_by_id[c.run_id] = {int(s): w_cat * cp[s] + w_pca * pp[s] for s in keys}
            m = marin_metrics(final_by_id, va_g, va_c)
            report(f"bag_{n_bags}  w_cat={w_cat:.2f}", m)
        print(f"  n_bags={n_bags} took {time.time()-t0:.0f}s", flush=True)

    # ── Pass 5: loss function sweep (RMSE, MAE, Huber, Quantile 0.5) ────────
    print("\n[5/5] CatBoost loss function sweep (paired with PCA blend 0.6)", flush=True)
    for loss in ("RMSE", "MAE", "Huber:delta=0.05", "Huber:delta=0.1", "Quantile:alpha=0.5"):
        t0 = time.time()
        m = fit_catboost(Xs_all, ys_all, params={"loss_function": loss})
        pr = predict_marin(m, marin_val)
        final_by_id = {}
        for c in marin_val:
            cp = pr.get(c.run_id, {}); pp = pca_preds.get(c.run_id, {})
            keys = set(cp) & set(pp)
            final_by_id[c.run_id] = {int(s): 0.6 * cp[s] + 0.4 * pp[s] for s in keys}
        met = marin_metrics(final_by_id, va_g, va_c)
        report(f"loss={loss:<22}", met, dt=time.time() - t0)

    # Dump all results
    Path("baselines/marin_sweep_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} rows to baselines/marin_sweep_results.json", flush=True)

    # Print top-10 by MAE, top-10 by ρ
    print("\n=== Top 10 configurations by Marin MAE ===", flush=True)
    for r in sorted(results, key=lambda x: x["mae"])[:10]:
        print(f"  {r['tag']:<44}  MAE={r['mae']:.4f}  ρ={r['rho']:.4f}", flush=True)
    print("\n=== Top 10 configurations by Marin ρ ===", flush=True)
    for r in sorted(results, key=lambda x: -x["rho"])[:10]:
        print(f"  {r['tag']:<44}  MAE={r['mae']:.4f}  ρ={r['rho']:.4f}", flush=True)


if __name__ == "__main__":
    main()
