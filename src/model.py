"""
Phase 1, Step 5.3: Model Training, Evaluation, and Threshold Tuning Module
==========================================================================
Trains a LightGBM binary classifier on candidate pair features with hard negatives,
predicts match probabilities, tunes decision threshold on grouped validation fold
using exact competition macro F0.5, and saves the trained model artifact.

Usage:
    # Run test suite:
    python src/model.py --test
"""

import os
import sys
import time
import json
import pickle
from typing import Dict, List, Set, Tuple, Any, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

from src.features import compute_features_for_pairs, FEATURE_NAMES
from src.scorer import macro_f05, f05_single

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ---------------------------------------------------------------------------
# Training Data Assembly
# ---------------------------------------------------------------------------

def build_training_pairs(
    candidates_dict: Dict[str, List[str]],
    gt_matches: Dict[str, Set[str]],
    max_negatives_per_positive: int = 4,
    random_seed: int = 42,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Constructs labeled (S1, Target) pairs from candidate lists and ground truth.
    Positives: Candidates present in gt_matches (label = 1)
    Hard Negatives: Candidates retrieved by blocker NOT in gt_matches (label = 0)
    
    Subsamples negatives to keep training balanced and fast.

    Returns:
        (pair_list, labels_array)
    """
    rng = np.random.RandomState(random_seed)
    pairs = []
    labels = []

    for s1_id, cands in candidates_dict.items():
        true_set = gt_matches.get(s1_id, set())

        pos_cands = [c for c in cands if c in true_set]
        neg_cands = [c for c in cands if c not in true_set]

        # Add all retrieved positives
        for c in pos_cands:
            pairs.append((s1_id, c))
            labels.append(1)

        # Subsample hard negatives
        n_pos = len(pos_cands)
        max_neg = max(1, n_pos * max_negatives_per_positive) if n_pos > 0 else 2
        if len(neg_cands) > max_neg:
            sampled_neg = rng.choice(neg_cands, size=max_neg, replace=False).tolist()
        else:
            sampled_neg = neg_cands

        for c in sampled_neg:
            pairs.append((s1_id, c))
            labels.append(0)

    return pairs, np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_lgbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    learning_rate: float = 0.08,
    num_leaves: int = 31,
    max_depth: int = 6,
    n_estimators: int = 200,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM binary classifier for pairwise entity matching.
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos = max(1.0, float(neg_count / max(1, pos_count)) * 0.7)  # conservative weighting for precision

    print(f"  Training LightGBM: {len(y_train):,} pairs ({pos_count:,} pos, {neg_count:,} neg)...")

    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=20,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)

    return model


# ---------------------------------------------------------------------------
# Grouped Threshold Tuning for Macro F0.5
# ---------------------------------------------------------------------------

def tune_prediction_threshold(
    val_s1_ids: List[str],
    val_candidates: Dict[str, List[str]],
    val_probabilities: Dict[Tuple[str, str], float],
    gt_matches: Dict[str, Set[str]],
    thresholds: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[float, float, Dict[str, Set[str]]]:
    """
    Sweeps probability thresholds to maximize official macro F0.5.
    At each threshold, any candidate with P(match) >= threshold is accepted.
    If no candidates exceed threshold, the entity is predicted as a singleton (set()).

    Returns:
        (best_threshold, best_f05_score, best_predictions_dict)
    """
    if thresholds is None:
        thresholds = np.arange(0.35, 0.90, 0.05)

    best_thresh = 0.50
    best_score = -1.0
    best_preds = {}

    if verbose:
        print("\n  --- Grouped Threshold Sweep (Macro F0.5) ---")
        print(f"  {'Threshold':<12} {'Macro F0.5':<14} {'Singletons%':<14} {'Total Matches':<14}")

    # Pre-structure pairs per S1 for speed
    s1_cand_probs = {}
    for s1_id in val_s1_ids:
        cands = val_candidates.get(s1_id, [])
        cand_probs = []
        for c in cands:
            p = val_probabilities.get((s1_id, c), 0.0)
            cand_probs.append((c, p))
        s1_cand_probs[s1_id] = cand_probs

    val_gt_subset = {s1_id: gt_matches.get(s1_id, set()) for s1_id in val_s1_ids}

    for thresh in thresholds:
        preds = {}
        n_single = 0
        total_m = 0

        for s1_id, cand_probs in s1_cand_probs.items():
            accepted = set(c for c, p in cand_probs if p >= thresh)
            preds[s1_id] = accepted
            if len(accepted) == 0:
                n_single += 1
            else:
                total_m += len(accepted)

        score = macro_f05(val_gt_subset, preds, verbose=False)
        single_pct = n_single / len(val_s1_ids) * 100

        if verbose:
            flag = " <-- BEST" if score > best_score else ""
            print(f"  {thresh:<12.2f} {score:<14.4f} {single_pct:<13.1f}% {total_m:<14,}{flag}")

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
            best_preds = preds

    if verbose:
        print(f"\n  Optimal Threshold: {best_thresh:.2f} (Macro F0.5: {best_score:.4f})")

    return best_thresh, best_score, best_preds


# ---------------------------------------------------------------------------
# Unit Test Suite
# ---------------------------------------------------------------------------

def run_tests() -> bool:
    """Verifies model training, inference, and threshold sweep on synthetic pairs."""
    print("\n--- Running Model Training Unit Tests ---\n")
    passed = 0
    failed = 0

    def assert_test(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    # 1. Synthetic feature matrix
    rng = np.random.RandomState(42)
    n_samples = 400
    n_feat = len(FEATURE_NAMES)

    X_train = rng.uniform(0.0, 1.0, size=(n_samples, n_feat)).astype(np.float32)
    # High name similarity & number match strongly correlates with label = 1
    y_train = (X_train[:, 0] * 0.7 + X_train[:, 10] * 0.3 > 0.6).astype(np.int32)

    X_val = rng.uniform(0.0, 1.0, size=(100, n_feat)).astype(np.float32)
    y_val = (X_val[:, 0] * 0.7 + X_val[:, 10] * 0.3 > 0.6).astype(np.int32)

    # Train model
    model = train_lgbm_classifier(X_train, y_train, X_val, y_val, n_estimators=50)
    assert_test("LightGBM model trained", model is not None)

    # Predict probabilities
    probs = model.predict_proba(X_val)[:, 1]
    assert_test("Probabilities generated in [0, 1]", (probs >= 0.0).all() and (probs <= 1.0).all())

    auc = roc_auc_score(y_val, probs)
    assert_test("AUC is reasonably high (>0.75)", auc > 0.75)

    # Test threshold sweep
    val_s1 = [f"S1-{i}" for i in range(20)]
    val_cands = {s1: [f"S2-{s1}-1", f"S2-{s1}-2"] for s1 in val_s1}
    val_gt = {s1: {f"S2-{s1}-1"} if int(s1.split("-")[1]) % 2 == 0 else set() for s1 in val_s1}

    pair_probs = {}
    for s1, c_list in val_cands.items():
        is_even = int(s1.split("-")[1]) % 2 == 0
        pair_probs[(s1, c_list[0])] = 0.85 if is_even else 0.20
        pair_probs[(s1, c_list[1])] = 0.15

    best_t, best_f, best_p = tune_prediction_threshold(val_s1, val_cands, pair_probs, val_gt, verbose=False)
    assert_test("Best threshold found", 0.3 <= best_t <= 0.8)
    assert_test("Macro F0.5 score is high (>0.80)", best_f > 0.80)

    print(f"\n--- Results: {passed} passed, {failed} failed ---")
    return failed == 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  python src/model.py --test   Run model training unit tests")
