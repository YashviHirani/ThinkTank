"""
Phase 1.4 — LightGBM Model Trainer
=====================================
Trains a binary LightGBM classifier on pair features.
Evaluates using the exact grouped macro-F0.5 scorer.
Tunes the decision threshold using out-of-fold predictions.

Key design decisions (matching the plan):
- Binary log-loss training (not custom metric)
- Threshold sweep using exact grouped macro-F0.5
- Hard negatives from blocking buckets (already in candidate set)
- No pair-level fold leakage

Usage:
    from src.trainer import train_and_evaluate, predict, tune_threshold
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import lightgbm as lgb
from typing import Tuple, List, Optional, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.scorer import macro_f05
from src.feature_extractor import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Threshold tuning — grouped macro F0.5
# ---------------------------------------------------------------------------

def _reconstruct_predictions(
    pair_ids: list,
    probas: np.ndarray,
    threshold: float,
) -> dict:
    """Convert pair predictions to grouped {s1_id: set(tgt_id)} format."""
    preds = {}
    for (s1_id, tgt_id), prob in zip(pair_ids, probas):
        if prob >= threshold:
            preds.setdefault(s1_id, set()).add(tgt_id)
    return preds


def tune_threshold(
    pair_ids: list,
    probas: np.ndarray,
    gt_matches: dict,
    val_s1_ids: set,
    n_steps: int = 50,
) -> Tuple[float, float]:
    """
    Sweep threshold in n_steps from 0.01 to 0.99.
    Returns (best_threshold, best_score).
    """
    # Filter gt to val split only
    gt_val = {k: v for k, v in gt_matches.items() if k in val_s1_ids}

    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_thresh = 0.5
    best_score = -1.0

    for thresh in thresholds:
        preds = _reconstruct_predictions(pair_ids, probas, thresh)
        # Fill in missing S1 IDs with empty predictions (singleton prediction)
        for s1_id in val_s1_ids:
            if s1_id not in preds:
                preds[s1_id] = set()
        score = macro_f05(gt_val, preds)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh, best_score


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

LGB_PARAMS = {
    'objective':        'binary',
    'metric':           'binary_logloss',
    'verbosity':        -1,
    'n_estimators':     500,
    'learning_rate':    0.05,
    'num_leaves':       63,
    'max_depth':        -1,
    'min_child_samples': 50,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'random_state':     42,
    'n_jobs':           -1,
}


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_pair_ids: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_pair_ids: list,
    gt_matches: dict,
    val_s1_ids: set,
    scale_pos_weight: float = 3.0,
    model_out_path: str = None,
    threshold_out_path: str = None,
) -> dict:
    """
    Train a LightGBM binary classifier, evaluate on validation set,
    and tune the decision threshold.

    Returns a dict with all metrics and the model.
    """
    t0 = time.time()

    print(f"\n{'='*55}")
    print(f"  Training LightGBM")
    print(f"{'='*55}")
    print(f"  Train pairs:  {len(X_train):,}  (pos={y_train.sum():,}, neg={(y_train==0).sum():,})")
    print(f"  Val pairs:    {len(X_val):,}    (pos={y_val.sum():,}, neg={(y_val==0).sum():,})")
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    params = {**LGB_PARAMS, 'scale_pos_weight': scale_pos_weight}

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s | best_iteration: {model.best_iteration_}")

    # Val predictions
    val_probas = model.predict_proba(X_val)[:, 1]

    # Threshold tuning
    print("\n  Tuning decision threshold...")
    best_thresh, best_f05 = tune_threshold(
        val_pair_ids, val_probas, gt_matches, val_s1_ids, n_steps=100
    )
    print(f"  Best threshold: {best_thresh:.3f} → Grouped macro F0.5: {best_f05:.4f}")

    # Also compute at 0.5 for comparison
    val_preds_50 = _reconstruct_predictions(val_pair_ids, val_probas, 0.5)
    for s1_id in val_s1_ids:
        if s1_id not in val_preds_50:
            val_preds_50[s1_id] = set()
    gt_val = {k: v for k, v in gt_matches.items() if k in val_s1_ids}
    f05_at_50 = macro_f05(gt_val, val_preds_50)
    print(f"  F0.5 at threshold=0.5: {f05_at_50:.4f}")

    # Feature importances
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: -x[1]
    )
    print("\n  Top-10 feature importances (gain):")
    for fname, imp in importances[:10]:
        bar = '█' * int(imp / max(i for _, i in importances) * 30)
        print(f"    {fname:35s} {imp:8.1f}  {bar}")

    # Save model
    if model_out_path:
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        with open(model_out_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n  Model saved to: {model_out_path}")

    if threshold_out_path:
        os.makedirs(os.path.dirname(threshold_out_path), exist_ok=True)
        threshold_data = {
            'best_threshold': best_thresh,
            'best_val_macro_f05': best_f05,
            'f05_at_threshold_50': f05_at_50,
            'best_iteration': model.best_iteration_,
            'scale_pos_weight': scale_pos_weight,
            'feature_importances': {k: int(v) for k, v in importances},
        }
        with open(threshold_out_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        print(f"  Threshold saved to: {threshold_out_path}")

    return {
        'model': model,
        'best_threshold': best_thresh,
        'best_val_macro_f05': best_f05,
        'f05_at_threshold_50': f05_at_50,
        'val_probas': val_probas,
        'val_pair_ids': val_pair_ids,
        'feature_importances': importances,
    }


# ---------------------------------------------------------------------------
# Prediction generation — apply model to a candidate dict
# ---------------------------------------------------------------------------

def predict(
    model,
    X_test: np.ndarray,
    test_pair_ids: list,
    threshold: float,
    all_test_s1_ids: list,
) -> dict:
    """
    Apply model to test pairs and return {s1_id: set(tgt_id)}.
    Ensures every test S1 ID appears (empty set for singletons).

    Parameters
    ----------
    model           : trained LGBMClassifier
    X_test          : feature matrix for test candidate pairs
    test_pair_ids   : list of (s1_id, tgt_id)
    threshold       : decision threshold
    all_test_s1_ids : complete list of test S1 IDs

    Returns
    -------
    predictions: {s1_id: set(tgt_id)}
    """
    if len(X_test) > 0:
        probas = model.predict_proba(X_test)[:, 1]
    else:
        probas = np.array([])

    preds = {}
    for (s1_id, tgt_id), prob in zip(test_pair_ids, probas):
        if prob >= threshold:
            preds.setdefault(s1_id, set()).add(tgt_id)

    # Ensure all test S1 IDs are present
    for s1_id in all_test_s1_ids:
        if s1_id not in preds:
            preds[s1_id] = set()

    return preds
