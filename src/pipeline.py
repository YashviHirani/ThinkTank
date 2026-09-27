"""
Entity Resolution Pipeline — Amazon ML Challenge 2026
======================================================
End-to-end orchestration: load → normalise → block → features → train → predict → output.

Run modes
---------
  python src/pipeline.py --mode train_val
      Full pipeline on train/val split. Measures blocking recall, trains model,
      tunes threshold, reports local grouped macro F0.5.

  python src/pipeline.py --mode submission
      Full pipeline: train on ALL training data, generate test submission files.

  python src/pipeline.py --mode phase2_blocking
      Phase 2 blocking experiment sweep (K sweep + key combination analysis).

  python src/pipeline.py --mode phase1_dryrun
      Phase 1 fast baseline dry-run verification.

Optional flags
--------------
  --K           20      Top-K cap for candidate generation
  --spw         3.0     LightGBM scale_pos_weight
  --no-train          Skip training, load cached model
  --model-path        Path to cached model (for --no-train)

Usage:
    python src/pipeline.py --mode train_val
    python src/pipeline.py --mode submission --K 20
"""

import os
import sys
import json
import time
import pickle
import argparse
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_all_data
from src.normaliser import normalise_df
from src.blocker import BlockingIndex, blocking_stats
from src.feature_extractor import build_feature_matrix, FEATURE_NAMES
from src.trainer import train_and_evaluate, predict, tune_threshold
from src.output_writer import write_outputs
from src.scorer import macro_f05

EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR      = os.path.join(BASE_DIR, 'output')

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBMISSION_LOG = os.path.join(EXPERIMENTS_DIR, 'submission_log.csv')


# ---------------------------------------------------------------------------
# Load split IDs
# ---------------------------------------------------------------------------

def load_split_ids():
    train_path = os.path.join(EXPERIMENTS_DIR, 'train_s1_ids.txt')
    val_path   = os.path.join(EXPERIMENTS_DIR, 'val_s1_ids.txt')
    with open(train_path) as f:
        train_ids = set(f.read().splitlines())
    with open(val_path) as f:
        val_ids = set(f.read().splitlines())
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Normalise all DataFrames
# ---------------------------------------------------------------------------

def normalise_all(data: dict) -> dict:
    print("\n--- Normalising DataFrames ---")
    t0 = time.time()
    norm = {}
    for key in ['train_s1', 'train_s2', 'train_s3', 'test_s1', 'test_s2', 'test_s3']:
        print(f"  Normalising {key}...", flush=True)
        norm[key] = normalise_df(data[key])
    print(f"  Normalisation complete in {time.time()-t0:.1f}s")
    return norm


# ---------------------------------------------------------------------------
# Build lookup dicts from normalised DFs
# ---------------------------------------------------------------------------

def build_lookup(df: pd.DataFrame) -> dict:
    """entity_id → row dict. Vectorized — avoids iterrows on large DFs."""
    records = df.to_dict('records')
    return {r['entity_id']: r for r in records}


# ---------------------------------------------------------------------------
# TRAIN_VAL mode: validate full pipeline on train/val split
# ---------------------------------------------------------------------------

def run_train_val(K: int = 20, spw: float = 3.0, no_train: bool = False, model_path: str = None):
    t_total = time.time()

    print("\n" + "="*60)
    print("  TRAIN_VAL MODE")
    print("="*60)

    # 1. Load data
    print("\n--- Loading data ---")
    data = load_all_data()
    gt_matches = data['gt_matches']

    # 2. Load split
    train_ids, val_ids = load_split_ids()
    print(f"  Train S1 IDs: {len(train_ids):,} | Val S1 IDs: {len(val_ids):,}")

    # 3. Normalise
    norm = normalise_all(data)

    # Build lookups
    train_s2_dict = build_lookup(norm['train_s2'])
    train_s3_dict = build_lookup(norm['train_s3'])
    target_dict   = {**train_s2_dict, **train_s3_dict}

    train_s1_df = norm['train_s1'][norm['train_s1']['entity_id'].isin(train_ids)].reset_index(drop=True)
    val_s1_df   = norm['train_s1'][norm['train_s1']['entity_id'].isin(val_ids)].reset_index(drop=True)

    train_s1_dict = build_lookup(train_s1_df)
    val_s1_dict   = build_lookup(val_s1_df)

    # 4. Block
    print("\n--- Blocking (train+val S2+S3) ---")
    idx = BlockingIndex(norm['train_s2'], norm['train_s3'])

    print("\n  Retrieving candidates for TRAIN split...")
    train_candidates = idx.retrieve(train_s1_df, K=K)

    print("\n  Retrieving candidates for VAL split...")
    val_candidates = idx.retrieve(val_s1_df, K=K)

    # Blocking recall on val
    val_blocking = blocking_stats(val_candidates, gt_matches, val_ids)
    print(f"\n  Val blocking stats: {val_blocking}")

    # 5. Feature extraction
    print("\n--- Building feature matrices ---")
    t_feat = time.time()

    print("  Train pairs...")
    X_train, y_train, train_pair_ids = build_feature_matrix(
        train_candidates, train_s1_dict, target_dict, gt_matches
    )
    print(f"  Train: {len(X_train):,} pairs, {y_train.sum():,} positives")

    print("  Val pairs...")
    X_val, y_val, val_pair_ids = build_feature_matrix(
        val_candidates, val_s1_dict, target_dict, gt_matches
    )
    print(f"  Val: {len(X_val):,} pairs, {y_val.sum():,} positives")
    print(f"  Feature extraction: {time.time()-t_feat:.1f}s")

    # 6. Train
    if not no_train:
        model_out  = os.path.join(MODELS_DIR, 'lgbm_model.pkl')
        thresh_out = os.path.join(MODELS_DIR, 'threshold.json')

        result = train_and_evaluate(
            X_train, y_train, train_pair_ids,
            X_val, y_val, val_pair_ids,
            gt_matches, val_ids,
            scale_pos_weight=spw,
            model_out_path=model_out,
            threshold_out_path=thresh_out,
        )
        model          = result['model']
        best_threshold = result['best_threshold']
        best_f05       = result['best_val_macro_f05']
    else:
        # Load cached model
        with open(model_path or os.path.join(MODELS_DIR, 'lgbm_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'threshold.json')) as f:
            tdata = json.load(f)
        best_threshold = tdata['best_threshold']
        best_f05       = tdata['best_val_macro_f05']

    # 7. Val predictions
    val_probas = model.predict_proba(X_val)[:, 1]
    val_preds  = {}
    for (s1_id, tgt_id), prob in zip(val_pair_ids, val_probas):
        if prob >= best_threshold:
            val_preds.setdefault(s1_id, set()).add(tgt_id)
    for s1_id in val_ids:
        if s1_id not in val_preds:
            val_preds[s1_id] = set()

    gt_val = {k: v for k, v in gt_matches.items() if k in val_ids}
    final_score = macro_f05(gt_val, val_preds, verbose=True)

    elapsed = time.time() - t_total
    print(f"\n  Total pipeline time: {elapsed:.0f}s")
    print(f"  FINAL VAL MACRO F0.5: {final_score:.4f}")

    # 8. Log experiment
    _log_experiment({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'mode': 'train_val',
        'K': K,
        'scale_pos_weight': spw,
        'val_blocking_recall': val_blocking['blocking_recall'],
        'val_avg_candidates': val_blocking['avg_candidates'],
        'val_macro_f05': final_score,
        'threshold': best_threshold,
        'notes': 'train_val run',
    })

    return model, best_threshold, final_score


# ---------------------------------------------------------------------------
# SUBMISSION mode: train on full training set, predict on test
# ---------------------------------------------------------------------------

def run_submission(K: int = 20, spw: float = 3.0, no_train: bool = False, model_path: str = None):
    t_total = time.time()

    print("\n" + "="*60)
    print("  SUBMISSION MODE")
    print("="*60)

    # 1. Load data
    data = load_all_data()
    gt_matches = data['gt_matches']

    # 2. Normalise
    norm = normalise_all(data)

    # Build lookups
    train_s2_dict = build_lookup(norm['train_s2'])
    train_s3_dict = build_lookup(norm['train_s3'])
    test_s2_dict  = build_lookup(norm['test_s2'])
    test_s3_dict  = build_lookup(norm['test_s3'])

    train_target_dict = {**train_s2_dict, **train_s3_dict}
    test_target_dict  = {**test_s2_dict, **test_s3_dict}

    train_s1_dict = build_lookup(norm['train_s1'])
    test_s1_dict  = build_lookup(norm['test_s1'])

    # 3. Block for TRAIN (to get train features) and TEST
    print("\n--- Blocking (train data, for model training) ---")
    train_idx = BlockingIndex(norm['train_s2'], norm['train_s3'])
    print("\n  Retrieving candidates for ALL TRAIN S1...")
    train_candidates = train_idx.retrieve(norm['train_s1'], K=K)

    print("\n--- Blocking (test data) ---")
    test_idx = BlockingIndex(norm['test_s2'], norm['test_s3'])
    print("\n  Retrieving candidates for ALL TEST S1...")
    test_candidates = test_idx.retrieve(norm['test_s1'], K=K)

    # Test blocking stats (no ground truth, but count candidates)
    total_test_cands = sum(len(v) for v in test_candidates.values())
    avg_test_cands   = total_test_cands / max(1, len(test_candidates))
    print(f"\n  Test candidates — total: {total_test_cands:,}, avg per S1: {avg_test_cands:.1f}")

    # 4. Feature matrices
    print("\n--- Feature matrices ---")

    print("  Train pairs (full training data)...")
    X_train, y_train, train_pair_ids = build_feature_matrix(
        train_candidates, train_s1_dict, train_target_dict, gt_matches
    )
    print(f"  Train: {len(X_train):,} pairs, {y_train.sum():,} positives")

    print("  Test pairs...")
    X_test, _, test_pair_ids = build_feature_matrix(
        test_candidates, test_s1_dict, test_target_dict, gt_matches=None
    )
    print(f"  Test: {len(X_test):,} pairs")

    # 5. Train on full training data (no val split — use tuned threshold from train_val)
    if not no_train:
        # Load threshold from train_val run
        thresh_path = os.path.join(MODELS_DIR, 'threshold.json')
        if os.path.exists(thresh_path):
            with open(thresh_path) as f:
                tdata = json.load(f)
            threshold = tdata['best_threshold']
            print(f"\n  Using saved threshold: {threshold:.3f}")
        else:
            threshold = 0.5
            print(f"\n  No saved threshold found, using default: {threshold}")

        model_out = os.path.join(MODELS_DIR, 'lgbm_model_submission.pkl')

        # Train on full training data
        import lightgbm as lgb
        from src.trainer import LGB_PARAMS

        params = {**LGB_PARAMS, 'scale_pos_weight': spw}
        model = lgb.LGBMClassifier(n_estimators=LGB_PARAMS.get('n_estimators', 500), **{
            k: v for k, v in LGB_PARAMS.items() if k != 'n_estimators'
        }, scale_pos_weight=spw)

        print(f"\n  Training on full training set ({len(X_train):,} pairs)...")
        t_tr = time.time()
        model.fit(X_train, y_train)
        print(f"  Training done in {time.time()-t_tr:.1f}s")

        with open(model_out, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Submission model saved to: {model_out}")
    else:
        mp = model_path or os.path.join(MODELS_DIR, 'lgbm_model_submission.pkl')
        with open(mp, 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'threshold.json')) as f:
            tdata = json.load(f)
        threshold = tdata['best_threshold']

    # 6. Predict
    all_test_s1_ids = norm['test_s1']['entity_id'].tolist()
    predictions = predict(model, X_test, test_pair_ids, threshold, all_test_s1_ids)

    # 7. Write outputs
    print("\n--- Writing output files ---")
    write_outputs(predictions, test_candidates, all_test_s1_ids, OUTPUT_DIR)

    elapsed = time.time() - t_total
    print(f"\n  Total submission pipeline time: {elapsed:.0f}s")

    # Log
    _log_experiment({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'mode': 'submission',
        'K': K,
        'scale_pos_weight': spw,
        'threshold': threshold,
        'notes': 'submission run',
    })


# ---------------------------------------------------------------------------
# PHASE 2 — Blocking experiment sweep
# ---------------------------------------------------------------------------

def run_phase2_blocking(K_values: list = None):
    """
    Phase 2.1: Systematic blocking experiment.
    Sweeps K values and reports:
      - blocking recall
      - avg / max candidates
    """
    if K_values is None:
        K_values = [10, 15, 20, 25, 30]

    print("\n" + "="*60)
    print("  PHASE 2 — Blocking Experiment Matrix")
    print("="*60)

    data = load_all_data()
    gt_matches = data['gt_matches']
    train_ids, val_ids = load_split_ids()
    norm = normalise_all(data)

    val_s1_df = norm['train_s1'][norm['train_s1']['entity_id'].isin(val_ids)].reset_index(drop=True)

    print("\n--- Building blocking index ---")
    idx = BlockingIndex(norm['train_s2'], norm['train_s3'])

    results = []
    report_path = os.path.join(BASE_DIR, 'reports', 'blocking_experiment.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    for K in K_values:
        print(f"\n  --- K = {K} ---")
        t0 = time.time()
        val_cands = idx.retrieve(val_s1_df, K=K)
        elapsed   = time.time() - t0

        stats = blocking_stats(val_cands, gt_matches, val_ids)
        stats['K'] = K
        stats['retrieval_time_s'] = round(elapsed, 1)
        results.append(stats)

        print(f"  K={K}: recall={stats['blocking_recall']:.4f} "
              f"avg_cands={stats['avg_candidates']:.1f} "
              f"max_cands={stats['max_candidates']} "
              f"time={elapsed:.1f}s")

    # Write blocking report
    lines = ["# Phase 2 — Blocking Experiment Report\n"]
    lines.append("| K | Blocking Recall | Avg Candidates | Max Candidates | Time (s) |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['K']} | {r['blocking_recall']:.4f} | "
            f"{r['avg_candidates']:.1f} | {r['max_candidates']} | {r['retrieval_time_s']} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    good = [r for r in results if r['blocking_recall'] >= 0.90]
    if good:
        best = min(good, key=lambda x: x['avg_candidates'])
        lines.append(f"\nRecommended K = **{best['K']}** "
                     f"(recall={best['blocking_recall']:.4f}, avg_cands={best['avg_candidates']:.1f})")
    else:
        lines.append("\nBlocking recall below 0.90 for all K values tested. Consider relaxing blocking keys.")

    report_text = '\n'.join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n  Blocking report saved to: {report_path}")

    return results


# ---------------------------------------------------------------------------
# Experiment log helper
# ---------------------------------------------------------------------------

def _log_experiment(entry: dict):
    fieldnames = [
        'timestamp', 'mode', 'K', 'scale_pos_weight', 'val_blocking_recall',
        'val_avg_candidates', 'val_macro_f05', 'threshold',
        'leaderboard_score', 'notes'
    ]
    exists = os.path.exists(SUBMISSION_LOG)
    with open(SUBMISSION_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(entry)
    print(f"  Experiment logged to: {SUBMISSION_LOG}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    parser.add_argument('--mode', choices=['train_val', 'submission', 'phase2_blocking', 'phase1_dryrun'],
                        default='train_val')
    parser.add_argument('--K', type=int, default=20,
                        help='Top-K candidate cap (default: 20)')
    parser.add_argument('--spw', type=float, default=3.0,
                        help='LightGBM scale_pos_weight (default: 3.0)')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training, load cached model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to cached model (used with --no-train)')
    args = parser.parse_args()

    if args.mode == 'train_val':
        run_train_val(K=args.K, spw=args.spw, no_train=args.no_train,
                      model_path=args.model_path)

    elif args.mode == 'submission':
        run_submission(K=args.K, spw=args.spw, no_train=args.no_train,
                       model_path=args.model_path)

    elif args.mode == 'phase2_blocking':
        run_phase2_blocking(K_values=[10, 15, 20, 25, 30])

    elif args.mode == 'phase1_dryrun':
        from src.phase1_pipeline import run_pipeline as run_phase1
        run_phase1(dry_run=True, top_k=args.K)


if __name__ == '__main__':
    main()
