"""
Phase 2 — Full Experiment Suite
================================
Runs the complete Phase 2 experiment matrix from the plan:
  2.1 Blocking experiment matrix (K sweep)
  2.2 Stronger model training (feature group ablation, hard negatives)
  2.3 Decision threshold tuning (global sweep + singleton analysis)

Usage:
    python src/phase2_experiments.py --exp blocking
    python src/phase2_experiments.py --exp model
    python src/phase2_experiments.py --exp threshold
    python src/phase2_experiments.py --exp all
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_loader        import load_all_data
from src.normaliser         import normalise_df
from src.blocker            import BlockingIndex, blocking_stats
from src.feature_extractor  import build_feature_matrix, FEATURE_NAMES, extract_features
from src.trainer            import train_and_evaluate, predict, tune_threshold, LGB_PARAMS, _reconstruct_predictions
from src.scorer             import macro_f05

EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
REPORTS_DIR     = os.path.join(BASE_DIR, 'reports')

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared setup: load + normalise + split
# ---------------------------------------------------------------------------

def setup(verbose=True):
    """Load, normalise, split. Returns shared objects used by all experiments."""
    if verbose:
        print("Loading data...")
    data = load_all_data()
    gt_matches = data['gt_matches']

    # Load split
    with open(os.path.join(EXPERIMENTS_DIR, 'train_s1_ids.txt')) as f:
        train_ids = set(f.read().splitlines())
    with open(os.path.join(EXPERIMENTS_DIR, 'val_s1_ids.txt')) as f:
        val_ids = set(f.read().splitlines())

    if verbose:
        print(f"  Train: {len(train_ids):,} | Val: {len(val_ids):,}")

    # Normalise
    if verbose:
        print("Normalising...")
    norm = {}
    for key in ['train_s1', 'train_s2', 'train_s3']:
        norm[key] = normalise_df(data[key])

    # Split S1 into train/val
    norm['train_s1_train'] = norm['train_s1'][norm['train_s1']['entity_id'].isin(train_ids)].reset_index(drop=True)
    norm['train_s1_val']   = norm['train_s1'][norm['train_s1']['entity_id'].isin(val_ids)].reset_index(drop=True)

    # Lookup dicts (vectorized — no iterrows)
    target_combined = pd.concat([norm['train_s2'], norm['train_s3']], ignore_index=True)
    target_records = target_combined.to_dict('records')
    target_dict   = {r['entity_id']: r for r in target_records}

    train_records = norm['train_s1_train'].to_dict('records')
    train_s1_dict = {r['entity_id']: r for r in train_records}

    val_records = norm['train_s1_val'].to_dict('records')
    val_s1_dict = {r['entity_id']: r for r in val_records}

    return {
        'data': data,
        'gt_matches': gt_matches,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'norm': norm,
        'target_dict': target_dict,
        'train_s1_dict': train_s1_dict,
        'val_s1_dict': val_s1_dict,
    }


# ---------------------------------------------------------------------------
# Experiment 2.1: Blocking K sweep
# ---------------------------------------------------------------------------

def exp_blocking(ctx):
    """Sweep K = [10, 15, 20, 25, 30] and report blocking recall."""
    print("\n" + "="*60)
    print("  EXPERIMENT 2.1 — Blocking K Sweep")
    print("="*60)

    norm        = ctx['norm']
    gt_matches  = ctx['gt_matches']
    val_ids     = ctx['val_ids']

    print("\nBuilding blocking index (train S2+S3)...")
    idx = BlockingIndex(norm['train_s2'], norm['train_s3'])

    results = []
    K_values = [10, 15, 20, 25, 30]

    for K in K_values:
        print(f"\n  --- K = {K} ---")
        t0 = time.time()
        val_cands = idx.retrieve(norm['train_s1_val'], K=K)
        elapsed = time.time() - t0

        stats = blocking_stats(val_cands, gt_matches, val_ids)
        stats['K'] = K
        stats['time_s'] = round(elapsed, 1)
        results.append(stats)
        print(f"  Blocking recall={stats['blocking_recall']:.4f} | avg_cands={stats['avg_candidates']:.1f} | max_cands={stats['max_candidates']} | time={elapsed:.1f}s")

    # Report
    lines = ["# Phase 2.1 — Blocking Experiment Results\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("| K | Blocking Recall | Avg Candidates | Max Candidates | Time (s) |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['K']} | {r['blocking_recall']:.4f} | {r['avg_candidates']:.1f} | {r['max_candidates']} | {r['time_s']} |")

    good = [r for r in results if r['blocking_recall'] >= 0.90]
    best = min(good, key=lambda x: x['avg_candidates']) if good else results[-1]
    lines.append(f"\n## Recommendation\nBest K = **{best['K']}** (recall={best['blocking_recall']:.4f}, avg_cands={best['avg_candidates']:.1f})")

    report_text = '\n'.join(lines)
    path = os.path.join(REPORTS_DIR, 'blocking_experiment.md')
    with open(path, 'w') as f:
        f.write(report_text)
    print(f"\n  Report saved: {path}")
    print(f"  Recommended K = {best['K']}")
    return results, best['K']


# ---------------------------------------------------------------------------
# Experiment 2.2: Model training with feature ablation
# ---------------------------------------------------------------------------

# Feature groups for ablation
FEATURE_GROUPS = {
    'name_basic':    ['name_ratio', 'name_token_sort_ratio', 'name_token_set_ratio', 'name_jaro_winkler'],
    'name_token':    ['name_token_jaccard', 'name_suffix_match'],
    'name_core':     ['name_core_ratio', 'name_core_token_sort', 'name_alpha_ratio'],
    'addr_basic':    ['addr_ratio', 'addr_token_sort_ratio', 'addr_partial_ratio'],
    'addr_token':    ['addr_token_jaccard', 'addr_common_token_count', 'addr_alpha_ratio'],
    'addr_flags':    ['addr_s1_blank', 'addr_tgt_blank', 'addr_both_present'],
    'structure':     ['country_equal', 'name_len_ratio', 'addr_len_ratio', 'shared_numeric_count'],
    'numeric':       ['house_agree', 'house_conflict', 'postcode_agree', 'postcode_conflict'],
    'provenance':    ['pre_score'],
}

ALL_FEAT_IDX = list(range(len(FEATURE_NAMES)))


def _get_feature_indices(groups: list) -> list:
    """Return indices of features in specified groups."""
    idx = []
    for g in groups:
        for feat in FEATURE_GROUPS.get(g, []):
            if feat in FEATURE_NAMES:
                idx.append(FEATURE_NAMES.index(feat))
    return sorted(set(idx))


def exp_model(ctx, K: int = 20):
    """Train model with all features; run scale_pos_weight sweep."""
    print("\n" + "="*60)
    print("  EXPERIMENT 2.2 — Model Training + SPW Sweep")
    print("="*60)

    gt_matches    = ctx['gt_matches']
    val_ids       = ctx['val_ids']
    train_ids     = ctx['train_ids']
    norm          = ctx['norm']
    target_dict   = ctx['target_dict']
    train_s1_dict = ctx['train_s1_dict']
    val_s1_dict   = ctx['val_s1_dict']

    # Block once
    print(f"\nBlocking (K={K})...")
    idx = BlockingIndex(norm['train_s2'], norm['train_s3'])
    train_cands = idx.retrieve(norm['train_s1_train'], K=K)
    val_cands   = idx.retrieve(norm['train_s1_val'], K=K)

    # Build features
    print("\nBuilding feature matrices...")
    X_train, y_train, train_pair_ids = build_feature_matrix(train_cands, train_s1_dict, target_dict, gt_matches)
    X_val,   y_val,   val_pair_ids   = build_feature_matrix(val_cands,   val_s1_dict,   target_dict, gt_matches)
    print(f"  Train: {len(X_train):,} pairs | {y_train.sum():,} pos | {(y_train==0).sum():,} neg")
    print(f"  Val:   {len(X_val):,} pairs | {y_val.sum():,} pos | {(y_val==0).sum():,} neg")

    # SPW sweep
    spw_values = [1.0, 2.0, 3.0, 5.0, 8.0]
    results = []

    for spw in spw_values:
        print(f"\n  --- scale_pos_weight = {spw} ---")
        res = train_and_evaluate(
            X_train, y_train, train_pair_ids,
            X_val, y_val, val_pair_ids,
            gt_matches, val_ids,
            scale_pos_weight=spw,
        )
        results.append({
            'spw': spw,
            'val_macro_f05': res['best_val_macro_f05'],
            'threshold': res['best_threshold'],
            'f05_at_50': res['f05_at_threshold_50'],
        })
        print(f"  SPW={spw}: F0.5={res['best_val_macro_f05']:.4f} @ threshold={res['best_threshold']:.3f}")

    # Best SPW
    best = max(results, key=lambda x: x['val_macro_f05'])
    print(f"\n  Best SPW = {best['spw']} (F0.5={best['val_macro_f05']:.4f})")

    # Train final model with best SPW and save
    print(f"\n  Training final model with SPW={best['spw']}...")
    final_res = train_and_evaluate(
        X_train, y_train, train_pair_ids,
        X_val, y_val, val_pair_ids,
        gt_matches, val_ids,
        scale_pos_weight=best['spw'],
        model_out_path=os.path.join(MODELS_DIR, 'lgbm_model.pkl'),
        threshold_out_path=os.path.join(MODELS_DIR, 'threshold.json'),
    )

    # Save experiment results
    exp_path = os.path.join(EXPERIMENTS_DIR, 'model_experiment.json')
    with open(exp_path, 'w') as f:
        json.dump({'spw_sweep': results, 'best': best, 'K': K}, f, indent=2)
    print(f"  Experiment saved: {exp_path}")

    return final_res, best


# ---------------------------------------------------------------------------
# Experiment 2.3: Detailed threshold analysis
# ---------------------------------------------------------------------------

def exp_threshold(ctx, K: int = 20):
    """
    Detailed threshold analysis:
    - Report total F0.5, singleton accuracy, non-singleton F0.5
    - Predicted match count, FP count at each threshold step
    """
    print("\n" + "="*60)
    print("  EXPERIMENT 2.3 — Threshold Analysis")
    print("="*60)

    gt_matches  = ctx['gt_matches']
    val_ids     = ctx['val_ids']
    norm        = ctx['norm']
    target_dict = ctx['target_dict']
    val_s1_dict = ctx['val_s1_dict']

    # Load model
    model_path = os.path.join(MODELS_DIR, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        print("  No model found. Run --exp model first.")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Block and features
    print(f"\nBlocking val (K={K})...")
    idx = BlockingIndex(norm['train_s2'], norm['train_s3'])
    val_cands = idx.retrieve(norm['train_s1_val'], K=K)

    X_val, y_val, val_pair_ids = build_feature_matrix(val_cands, val_s1_dict, target_dict, gt_matches)
    val_probas = model.predict_proba(X_val)[:, 1]

    gt_val = {k: v for k, v in gt_matches.items() if k in val_ids}
    val_singletons = {k for k, v in gt_val.items() if not v}
    val_non_single = {k for k, v in gt_val.items() if v}

    # Threshold sweep with detailed reporting
    thresholds = np.linspace(0.05, 0.95, 90)
    records = []

    for thresh in thresholds:
        preds = _reconstruct_predictions(val_pair_ids, val_probas, thresh)
        for s1_id in val_ids:
            if s1_id not in preds:
                preds[s1_id] = set()

        total_f05 = macro_f05(gt_val, preds)

        sing_preds = {k: preds[k] for k in val_singletons}
        sing_correct = sum(1 for k in val_singletons if not preds[k])
        sing_acc = sing_correct / len(val_singletons) if val_singletons else 0.0

        non_single_preds = {k: preds[k] for k in val_non_single}
        from src.scorer import f05_single
        non_single_f05 = np.mean([f05_single(gt_val[k], preds[k]) for k in val_non_single]) if val_non_single else 0.0

        total_predicted = sum(len(v) for v in preds.values())
        fp_count = sum(
            len(preds[k] - gt_val.get(k, set()))
            for k in val_ids if k in preds
        )

        records.append({
            'threshold': round(float(thresh), 3),
            'total_f05': round(total_f05, 4),
            'singleton_acc': round(sing_acc, 4),
            'non_singleton_f05': round(non_single_f05, 4),
            'total_predicted': total_predicted,
            'fp_count': fp_count,
        })

    # Find best
    best = max(records, key=lambda x: x['total_f05'])
    print(f"\n  Best threshold: {best['threshold']} → F0.5={best['total_f05']}")
    print(f"  Singleton acc: {best['singleton_acc']:.4f}")
    print(f"  Non-singleton F0.5: {best['non_singleton_f05']:.4f}")
    print(f"  Total predicted: {best['total_predicted']:,} | FP: {best['fp_count']:,}")

    # Save detailed report
    df_results = pd.DataFrame(records)
    report_path = os.path.join(EXPERIMENTS_DIR, 'threshold_analysis.csv')
    df_results.to_csv(report_path, index=False)
    print(f"  Detailed results saved: {report_path}")

    # Update threshold.json
    thresh_path = os.path.join(MODELS_DIR, 'threshold.json')
    with open(thresh_path, 'r') as f:
        tdata = json.load(f)
    tdata['best_threshold_phase2'] = best['threshold']
    tdata['phase2_val_macro_f05'] = best['total_f05']
    tdata['phase2_singleton_acc'] = best['singleton_acc']
    with open(thresh_path, 'w') as f:
        json.dump(tdata, f, indent=2)

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['blocking', 'model', 'threshold', 'all'], default='all')
    parser.add_argument('--K', type=int, default=20)
    args = parser.parse_args()

    ctx = setup()

    if args.exp in ('blocking', 'all'):
        _, recommended_K = exp_blocking(ctx)
        K = recommended_K
    else:
        K = args.K

    if args.exp in ('model', 'all'):
        exp_model(ctx, K=K)

    if args.exp in ('threshold', 'all'):
        exp_threshold(ctx, K=K)

    print("\n" + "="*60)
    print("  Phase 2 experiments complete.")
    print("="*60)


if __name__ == '__main__':
    main()
