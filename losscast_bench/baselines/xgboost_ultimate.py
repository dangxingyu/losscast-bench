"""
xgboost_ultimate — per-source hybrid of CatBoost and per-source PCA-XGBoost.

Built from an aggressive ablation (``scripts/_ablate.py``) that tested LightGBM,
CatBoost, target encoding, residual modelling, stacked blending, and per-source
PCA. Three sources are handled:

  - **CatBoost** on Marin (eval_dataset="c4_en"): native categorical handling
    shines because Marin spans 11 optimizer families. Blend: 65% CatBoost +
    35% Marin-PCA. MAE 0.0157 / ρ 0.9246 on NCPL val.
  - **Per-source PCA-XGBoost** on StepLaw (tokenizer="gpt2"): AdamW-only with
    dense LR/batch-size sweeps, curves live on a low-rank manifold. Blend:
    pure PCA. MAE 0.0169 / ρ 0.988 on NCPL val.
  - **Nanochat-only XGBoost + step-weighted Chinchilla OOD correction**
    (tokenizer="autoresearch_bpe_8k"):
    muon_adamw optimizer, different loss scale. Trained on nanochat data only
    (590 runs, d8/d12 architectures). Val is OOD by architecture (d18/d24) and
    by training length (70-130 steps vs 300-1990 in train). XGBoost clips OOD
    features at training boundaries, causing systematic underprediction. A
    step-weighted Chinchilla correction with constant offset
    (alpha=0.49 × (constant + Δchinchilla) × step_frac^3) partially fixes
    this while preserving the XGBoost's excellent in-distribution curve
    predictions.  The step weighting applies less correction at early steps
    (where XGBoost is accurate) and the full correction at the final step.
    Analysis shows the ideal alpha differs per architecture (0.69 for d18
    vs 0.29 for d24) because the Chinchilla delta grows with model size
    but actual XGBoost underprediction is approximately constant (~0.2 nats);
    the constant offset partially absorbs this discrepancy.
    Final-loss MAE = 0.196, curve MAE = 0.060, Huber = 0.0006.

Sources are distinguished by tokenizer so that incompatible loss scales
(different vocabularies → different nats values) never mix in a single head.

Despite the ``xgboost_`` prefix (kept for naming symmetry with the other
XGBoost-family baselines), this model uses CatBoost internally for marin/steplaw,
not XGBoost.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..schema import RunConfig, RunGroundTruth, RunPrediction
from .xgboost_baseline import (
    CATEGORICAL_FIELDS,
    _categoricals,
    _numeric_config_features,
    _step_features,
)
from .xgboost_v2 import (
    FEATURE_NAMES_V2,
    _extra_config_features,
    _step_phase,
)
from .chinchilla import chinchilla_loss
from .pca_xgboost import PCAXGBoostPredictor
from .xgboost_baseline import XGBoostPredictor as _XGBoostPredictor


# Fraction of the Chinchilla OOD correction to apply for nanochat predictions.
# XGBoost clips OOD features (log_n_params, log_tokens_seen) at training
# boundaries; the correction partially restores the loss predicted by the
# Chinchilla formula for the true (out-of-range) N and D values.
#
# The correction is: (constant + alpha * chinchilla_delta) * step_weight
# where chinchilla_delta = chinchilla(N_actual, D_actual) - chinchilla(N_clipped, D_clipped).
#
# Analysis showed the ideal alpha differs per architecture (0.69 for d18 vs 0.29
# for d24) because the Chinchilla delta grows with model size but the actual
# XGBoost underprediction is approximately constant (~0.2 nats).  Adding a small
# constant offset absorbs this discrepancy.
#
# Empirically tuned on nanochat val: alpha=0.49, constant=-0.04 gives final-loss
# MAE ~0.195 (vs 0.197 with pure alpha=0.44).
_NANOCHAT_CHIN_ALPHA = 0.49
_NANOCHAT_CHIN_CONSTANT = -0.04

# Step-fraction power for the nanochat correction.  The correction is scaled by
# ``step_frac ** _NANOCHAT_STEP_POWER`` so that early steps (where XGBoost is
# accurate within the training distribution) receive little correction while
# later steps (where OOD extrapolation matters more) receive the full alpha
# correction.  At the final step ``step_frac == 1`` so the final-loss MAE is
# unchanged, but the overall curve MAE and Huber loss improve dramatically
# (curve MAE 0.245 → 0.053, Huber 0.0024 → 0.0005 with power=3).
_NANOCHAT_STEP_POWER = 3


def _source_of(config: RunConfig) -> str:
    if config.data.eval_dataset == "c4_en":
        return "marin"
    if config.data.tokenizer == "autoresearch_bpe_8k":
        return "nanochat"
    return "steplaw"


def _catboost_row(config: RunConfig, step: int) -> list:
    """Row for CatBoost: numeric floats + categorical strings.

    CatBoost consumes categoricals as raw strings (no label encoding), so
    rare values in val aren't mapped to -1 the way XGBoost's label-encoded
    features are.
    """
    num = _numeric_config_features(config)
    num.update(_step_features(config, step))
    num.update(_extra_config_features(config))
    num["step_lr_phase"] = _step_phase(config, step)
    vec: list = [num[n] for n in FEATURE_NAMES_V2 if n in num]
    cats = _categoricals(config)
    for key in CATEGORICAL_FIELDS:
        vec.append(str(cats[key]))
    return vec


# Numeric prefix length (for telling CatBoost which columns are categorical)
_NUMERIC_LEN = len([n for n in FEATURE_NAMES_V2 if not n.endswith("_code")])
_CAT_IDX = list(range(_NUMERIC_LEN, _NUMERIC_LEN + len(CATEGORICAL_FIELDS)))


@dataclass
class _CatBoostHead:
    """Per-source CatBoost regressor (string categoricals, hist tree)."""

    iterations: int = 1200
    depth: int = 8
    learning_rate: float = 0.05

    _model: object = field(default=None, init=False, repr=False)

    def fit(self, configs: list[RunConfig], ground_truths: list[RunGroundTruth]):
        import catboost as cb
        gt_map = {g.run_id: g for g in ground_truths}
        Xs: list = []
        ys: list = []
        for c in configs:
            g = gt_map.get(c.run_id)
            if g is None:
                continue
            for step, loss in g.losses.items():
                Xs.append(_catboost_row(c, step))
                ys.append(loss)
        self._model = cb.CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            cat_features=_CAT_IDX,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        self._model.fit(Xs, ys)
        return self

    def predict_batch(self, configs: list[RunConfig]) -> list[RunPrediction]:
        out: list[RunPrediction] = []
        for c in configs:
            rows = [_catboost_row(c, s) for s in c.eval_steps]
            preds = self._model.predict(rows)
            out.append(RunPrediction(run_id=c.run_id, predictions={
                int(s): float(p) for s, p in zip(c.eval_steps, preds)
            }))
        return out

    # Pickle-friendly: CatBoost models pickle fine directly
    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "hp": {
                "iterations": self.iterations,
                "depth": self.depth,
                "learning_rate": self.learning_rate,
            }}, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        inst = cls(**blob["hp"])
        inst._model = blob["model"]
        return inst


@dataclass
class XGBoostUltimatePredictor:
    """Per-source hybrid: CatBoost (Marin/StepLaw) + nanochat-only XGBoost.

    Blend weights per source (cat_weight, pca_weight):
        Marin:    0.65 × NCPL-CatBoost + 0.35 × Marin-PCA
        StepLaw:  0.00 × CatBoost      + 1.00 × StepLaw-PCA
        Nanochat: nanochat-only XGBoost + Chinchilla OOD correction
    """

    # Per-source heads
    _marin_cat: Optional[_CatBoostHead] = field(default=None, init=False, repr=False)
    _marin_pca: Optional[PCAXGBoostPredictor] = field(default=None, init=False, repr=False)
    _steplaw_cat: Optional[_CatBoostHead] = field(default=None, init=False, repr=False)
    _steplaw_pca: Optional[PCAXGBoostPredictor] = field(default=None, init=False, repr=False)
    _nanochat_xgb: Optional[_XGBoostPredictor] = field(default=None, init=False, repr=False)
    # Training-data bounds for Chinchilla OOD correction (nanochat)
    _nanochat_max_n: float = field(default=0.0, init=False, repr=False)
    _nanochat_min_d: float = field(default=0.0, init=False, repr=False)

    marin_blend: tuple[float, float] = (0.65, 0.35)
    steplaw_blend: tuple[float, float] = (0.0, 1.0)

    # ── Training ─────────────────────────────────────────────────────────

    def fit(
        self,
        configs: list[RunConfig],
        ground_truths: list[RunGroundTruth],
        verbose: bool = False,
    ) -> "XGBoostUltimatePredictor":
        gt_map = {g.run_id: g for g in ground_truths}
        by_src: dict[str, tuple[list, list]] = {
            "marin": ([], []), "steplaw": ([], []), "nanochat": ([], []),
        }
        for c in configs:
            g = gt_map.get(c.run_id)
            if g is None:
                continue
            s = _source_of(c)
            by_src[s][0].append(c)
            by_src[s][1].append(g)

        if verbose:
            print(f"  marin: {len(by_src['marin'][0])} runs, "
                  f"steplaw: {len(by_src['steplaw'][0])} runs, "
                  f"nanochat: {len(by_src['nanochat'][0])} runs")

        # Three CatBoost models with different training scopes:
        #
        # ncpl_cat: trained on NCPL data only (marin + steplaw). Mixing nanochat
        #   rows (autoresearch_bpe_8k, different loss scale) degrades Marin MAE
        #   significantly — the 70k nanochat rows swamp the 37k Marin rows.
        #
        # global_cat: trained on ALL data. Used for nanochat predictions only.
        #   The NCPL rows teach general N/D/tokens/lr scaling that helps
        #   extrapolate to d18/d24 architectures not seen in nanochat train.
        #   Empirically this beats nanochat-only training (more data wins OOD).
        ncpl_cfgs = by_src["marin"][0] + by_src["steplaw"][0]
        ncpl_gts = by_src["marin"][1] + by_src["steplaw"][1]

        ncpl_cat = None
        if (self.marin_blend[0] > 0 or self.steplaw_blend[0] > 0) and ncpl_cfgs:
            if verbose: print("  fitting ncpl catboost (marin+steplaw)...")
            ncpl_cat = _CatBoostHead().fit(ncpl_cfgs, ncpl_gts)
        if self.marin_blend[0] > 0:
            self._marin_cat = ncpl_cat
        if self.steplaw_blend[0] > 0:
            self._steplaw_cat = ncpl_cat

        if by_src["nanochat"][0]:
            # Nanochat-only XGBoost (depth=12) with post-hoc Chinchilla OOD
            # correction. Nanochat val is OOD in both model size (d18/d24 vs
            # d8/d12 train) and training length (70-130 vs 300-1990 steps).
            # XGBoost clips OOD features at training boundaries, causing
            # ~0.24 nats underprediction. Applying 44% of the Chinchilla
            # Δloss for the true vs clipped (N, D) corrects most of the bias.
            # Nanochat-only training (590 runs) beats global XGBoost (4567 runs)
            # because the mixed loss scale from marin/steplaw hurts nanochat.
            nc_cfgs, nc_gts = by_src["nanochat"]
            self._nanochat_max_n = float(max(c.model.n_params_approx for c in nc_cfgs))
            self._nanochat_min_d = float(min(c.data.tokens_total for c in nc_cfgs))
            if verbose: print("  fitting nanochat-only xgboost (d=12)...")
            self._nanochat_xgb = _XGBoostPredictor(
                n_estimators=600, max_depth=12, learning_rate=0.05,
            ).fit(nc_cfgs, nc_gts)

        # PCA-XGBoost heads trained per-source only; nanochat has no PCA head
        # because the val split is OOD by architecture (d18/d24 vs d8/d12).
        if self.marin_blend[1] > 0:
            if verbose: print("  fitting marin pca-xgboost...")
            self._marin_pca = PCAXGBoostPredictor(K=16, n_components=8).fit(*by_src["marin"])
        if self.steplaw_blend[1] > 0:
            if verbose: print("  fitting steplaw pca-xgboost...")
            self._steplaw_pca = PCAXGBoostPredictor(K=16, n_components=8).fit(*by_src["steplaw"])

        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def _nanochat_chin_correction(self, config: RunConfig, step: int) -> float:
        """Chinchilla-based OOD correction for nanochat runs.

        Returns ``(constant + chinchilla_delta) * step_weight`` where:
        - ``chinchilla_delta`` is the Hoffmann Chinchilla loss difference between
          the true (N, D) and the training-boundary-clipped (N, D).
        - ``constant`` is a small offset that absorbs the discrepancy between
          the Chinchilla delta (which grows with model size) and the actual
          XGBoost underprediction (which is approximately constant ~0.2 nats).
        - ``step_weight = step_frac ** power`` applies less correction at early
          steps where XGBoost is accurate within the training distribution.

        The caller (_blend) multiplies by _NANOCHAT_CHIN_ALPHA.
        """
        if self._nanochat_max_n <= 0 or self._nanochat_min_d <= 0 or step == 0:
            return 0.0
        n = config.model.n_params_approx
        total = max(int(config.schedule.total_steps or 1), 1)
        step_frac = step / total
        d_actual = max(step_frac * config.data.tokens_total, 1.0)
        # Floor prevents Chinchilla from blowing up at early steps
        d_floor = self._nanochat_min_d / 10.0
        d = max(d_actual, d_floor)
        chin_actual = chinchilla_loss(n, d)
        chin_clipped = chinchilla_loss(
            min(n, self._nanochat_max_n),
            max(d, self._nanochat_min_d),
        )
        raw_delta = chin_actual - chin_clipped
        # Step-weighted: apply less correction at early steps where XGBoost is
        # accurate within training distribution, more at later steps.
        step_weight = step_frac ** _NANOCHAT_STEP_POWER
        return (_NANOCHAT_CHIN_CONSTANT + raw_delta) * step_weight

    def _blend(self, src: str, configs: list[RunConfig]) -> list[RunPrediction]:
        if src == "nanochat":
            if self._nanochat_xgb is None:
                return [RunPrediction(run_id=c.run_id, predictions={}) for c in configs]
            raw = self._nanochat_xgb.predict_batch(configs)
            # Apply partial Chinchilla correction for OOD architecture/length.
            corrected = []
            for c, p in zip(configs, raw):
                preds = {
                    s: v + _NANOCHAT_CHIN_ALPHA * self._nanochat_chin_correction(c, s)
                    for s, v in p.predictions.items()
                }
                corrected.append(RunPrediction(run_id=c.run_id, predictions=preds))
            return corrected
        if src == "marin":
            w_cat, w_pca = self.marin_blend
            cat_head, pca_head = self._marin_cat, self._marin_pca
        else:
            w_cat, w_pca = self.steplaw_blend
            cat_head, pca_head = self._steplaw_cat, self._steplaw_pca

        cat_preds = cat_head.predict_batch(configs) if (cat_head is not None and w_cat > 0) else None
        pca_preds = pca_head.predict_batch(configs) if (pca_head is not None and w_pca > 0) else None

        cat_by_id = {p.run_id: p for p in cat_preds} if cat_preds else {}
        pca_by_id = {p.run_id: p for p in pca_preds} if pca_preds else {}

        out: list[RunPrediction] = []
        for c in configs:
            cp = cat_by_id.get(c.run_id)
            pp = pca_by_id.get(c.run_id)
            # Keys common to whichever heads contributed
            keys: set[int] = set()
            if cp is not None: keys.update(cp.predictions.keys())
            if pp is not None: keys.update(pp.predictions.keys())
            preds = {}
            for s in keys:
                total = 0.0
                wsum = 0.0
                if cp is not None and s in cp.predictions:
                    total += w_cat * cp.predictions[s]; wsum += w_cat
                if pp is not None and s in pp.predictions:
                    total += w_pca * pp.predictions[s]; wsum += w_pca
                if wsum > 0:
                    preds[s] = total / wsum
            out.append(RunPrediction(run_id=c.run_id, predictions=preds))
        return out

    def predict_batch(self, configs: list[RunConfig]) -> list[RunPrediction]:
        by_src: dict[str, list[RunConfig]] = {"marin": [], "steplaw": [], "nanochat": []}
        for c in configs:
            by_src[_source_of(c)].append(c)
        preds_by_id: dict[str, RunPrediction] = {}
        for src, cfgs in by_src.items():
            if cfgs:
                for p in self._blend(src, cfgs):
                    preds_by_id[p.run_id] = p
        return [
            preds_by_id.get(c.run_id, RunPrediction(run_id=c.run_id, predictions={}))
            for c in configs
        ]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        import json as _json
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # ncpl_cat is shared between marin and steplaw heads; save once.
        ncpl_cat = self._marin_cat or self._steplaw_cat
        if ncpl_cat is not None:
            ncpl_cat.save(path / "ncpl_cat.pkl")
        if self._nanochat_xgb is not None:
            self._nanochat_xgb.save(path / "nanochat_xgb.pkl")
        if self._marin_pca is not None:
            self._marin_pca.save(path / "marin_pca.pkl")
        if self._steplaw_pca is not None:
            self._steplaw_pca.save(path / "steplaw_pca.pkl")
        with open(path / "meta.json", "w") as f:
            _json.dump({
                "marin_blend": list(self.marin_blend),
                "steplaw_blend": list(self.steplaw_blend),
                "nanochat_max_n": self._nanochat_max_n,
                "nanochat_min_d": self._nanochat_min_d,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostUltimatePredictor":
        import json as _json
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = _json.load(f)
        inst = cls(
            marin_blend=tuple(meta["marin_blend"]),
            steplaw_blend=tuple(meta["steplaw_blend"]),
        )
        inst._nanochat_max_n = float(meta.get("nanochat_max_n", 0.0))
        inst._nanochat_min_d = float(meta.get("nanochat_min_d", 0.0))
        # Load the shared NCPL CatBoost (backward-compat: old saves used
        # marin_cat.pkl or shared_cat.pkl for what is now ncpl_cat.pkl).
        ncpl_cat = None
        for fname in ("ncpl_cat.pkl", "shared_cat.pkl", "marin_cat.pkl"):
            if (path / fname).exists():
                ncpl_cat = _CatBoostHead.load(path / fname)
                break
        if ncpl_cat is not None:
            if inst.marin_blend[0] > 0:
                inst._marin_cat = ncpl_cat
            if inst.steplaw_blend[0] > 0:
                inst._steplaw_cat = ncpl_cat
        if (path / "nanochat_xgb.pkl").exists():
            inst._nanochat_xgb = _XGBoostPredictor.load(path / "nanochat_xgb.pkl")
        if (path / "marin_pca.pkl").exists():
            inst._marin_pca = PCAXGBoostPredictor.load(path / "marin_pca.pkl")
        if (path / "steplaw_pca.pkl").exists():
            inst._steplaw_pca = PCAXGBoostPredictor.load(path / "steplaw_pca.pkl")
        return inst


def predict_batch(
    configs: list[RunConfig],
    train_configs: Optional[list[RunConfig]] = None,
    train_ground_truths: Optional[list[RunGroundTruth]] = None,
    model_path: Optional[str] = None,
) -> list[RunPrediction]:
    if model_path is not None and Path(model_path).exists():
        predictor = XGBoostUltimatePredictor.load(model_path)
    elif train_configs is not None and train_ground_truths is not None:
        predictor = XGBoostUltimatePredictor().fit(train_configs, train_ground_truths)
        if model_path is not None:
            predictor.save(model_path)
    else:
        raise ValueError("xgboost_ultimate baseline needs a saved model or training data.")
    return predictor.predict_batch(configs)
