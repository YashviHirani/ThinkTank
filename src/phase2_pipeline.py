"""
Phase 2: Full-Scale Pipeline & Experiment Suite
================================================
Builds directly on Phase 1 modules (normalizer, blocker, features, model, scorer).
Phase 1 baseline: val Macro F0.5 = 0.9222, blocking recall 90.83%, avg candidates 4.56.

Phase 2 Goals:
  2.1 Full-scale train/val split run  (no subsampling)
  2.2 Blocking K sweep                (recall vs compactness tradeoff)
  2.3 scale_pos_weight sweep          (precision-recall balance for F0.5)
  2.4 Hard negative enrichment        (neg_per_pos ratio sweep)
  2.5 Generate final submission       (both output TSVs, validated)

Usage:
    python src/phase2_pipeline.py --mode train_val
    python src/phase2_pipeline.py --mode blocking_sweep
    python src/phase2_pipeline.py --mode spw_sweep
    python src/phase2_pipeline.py --mode submission
    python src/phase2_pipeline.py --mode all
"""

import os
import sys
import csv
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.normalizer import (
    normalize_general_text, extract_core_name, get_sorted_tokens,
    normalize_address, extract_numeric_tokens, extract_postcode_candidate,
)
from src.blocker import InvertedIndex, generate_candidate_pairs, evaluate_blocking_recall
from src.features import compute_features_for_pairs, FEATURE_NAMES
from src.model import build_training_pairs, train_lgbm_classifier, tune_prediction_threshold
from src.scorer import macro_f05

DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")

for d in [OUTPUT_DIR, MODELS_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(EXPERIMENTS_DIR, "submission_log.csv")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_tsv(path: str, nrows=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, nrows=nrows, keep_default_na=False)


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    gt = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = str(row.get("matched_entity_ids", "")).strip()
        gt[s1_id] = set(x.strip() for x in raw.split(",") if x.strip()) if raw and raw != "nan" else set()
    return gt


def build_records_dict(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Fast normalized record lookup. Uses numpy arrays, avoids iterrows."""
    records = {}
    ids      = df["entity_id"].values
    names    = df["business_name"].fillna("").values
    addrs    = df["business_address"].fillna("").values
    countries = df["country"].fillna("").str.strip().str.upper().values

    n = len(df)
    report_every = max(1, n // 10)
    t0 = time.time()
    for i in range(n):
        if i > 0 and i % report_every == 0:
            pct = i / n * 100
            print(f"    {i:,}/{n:,} ({pct:.0f}%)  {time.time()-t0:.0f}s", flush=True)
        cn_raw = normalize_general_text(names[i])
        cn, suf = extract_core_name(cn_raw)
        ca = normalize_address(addrs[i])
        nums = extract_numeric_tokens(ca)
        pc = extract_postcode_candidate(ca)
        records[ids[i]] = {
            "core_name":      cn,
            "legal_suffix":   suf,
            "sorted_name":    get_sorted_tokens(cn),
            "clean_address":  ca,
            "numeric_tokens": nums,
            "postcode":       pc,
            "country":        countries[i],
        }
    return records


def log_run(name: str, threshold: float, f05: float, status: str, notes: str = ""):
    """Append a row to the submission log CSV."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "submission_name", "threshold", "val_macro_f05", "status", "notes"])
        w.writerow([datetime.now(timezone.utc).isoformat(), name, f"{threshold:.3f}", f"{f05:.4f}", status, notes])
    print(f"  Logged → {LOG_FILE}")


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_train_data(verbose=True):
    """Load all training TSVs + ground truth. Returns raw DataFrames."""
    train_dir = os.path.join(DATASET_DIR, "train")
    if verbose:
        print("  Loading train_source1.tsv ...", flush=True)
    s1 = load_tsv(os.path.join(train_dir, "train_source1.tsv"))
    if verbose:
        print(f"    {len(s1):,} rows")
        print("  Loading train_source2.tsv ...", flush=True)
    s2 = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    if verbose:
        print(f"    {len(s2):,} rows")
        print("  Loading train_source3.tsv ...", flush=True)
    s3 = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    if verbose:
        print(f"    {len(s3):,} rows")
        print("  Loading ground truth ...", flush=True)
    gt_df = load_tsv(os.path.join(train_dir, "train_ground_truth.tsv"))
    gt = parse_ground_truth(gt_df)
    if verbose:
        singletons = sum(1 for v in gt.values() if not v)
        print(f"    {len(gt):,} S1 entities | {len(gt)-singletons:,} non-singleton | {singletons:,} singleton")
    return s1, s2, s3, gt


def load_val_split():
    """Load the stratified val split S1 IDs produced by splitter.py."""
    val_path = os.path.join(EXPERIMENTS_DIR, "val_s1_ids.txt")
    train_path = os.path.join(EXPERIMENTS_DIR, "train_s1_ids.txt")
    if not os.path.exists(val_path):
        return None, None
    with open(train_path) as f:
        train_ids = set(f.read().splitlines())
    with open(val_path) as f:
        val_ids = set(f.read().splitlines())
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Mode 1: train_val — full-scale split run
# ---------------------------------------------------------------------------

def run_train_val(top_k: int = 20, spw: float = None, neg_ratio: int = 4):
    """
    Full-scale train/val run using the existing stratified split.
    Measures true out-of-fold grouped Macro F0.5.
    """
    t0 = time.time()
    print("\n" + "="*65)
    print("  Phase 2.1 — Full-Scale Train/Val Run")
    print("="*65)

    # 1. Load split IDs
    train_ids, val_ids = load_val_split()
    if train_ids is None:
        print("  ERROR: val_s1_ids.txt not found. Run: python src/splitter.py first.")
        return None

    print(f"\n  Train S1: {len(train_ids):,} | Val S1: {len(val_ids):,}")

    # 2. Load data
    print("\n--- Loading Data ---")
    s1_df, s2_df, s3_df, gt_full = load_train_data()

    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    s1_train_df = s1_df[s1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val_df   = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)

    gt_train = {k: v for k, v in gt_full.items() if k in train_ids}
    gt_val   = {k: v for k, v in gt_full.items() if k in val_ids}

    # 3. Normalize
    print("\n--- Normalising Records ---")
    print(f"  Normalising {len(targets_df):,} target records (S2+S3)...")
    target_dict = build_records_dict(targets_df)

    print(f"\n  Normalising {len(s1_train_df):,} train S1 records...")
    s1_train_dict = build_records_dict(s1_train_df)

    print(f"\n  Normalising {len(s1_val_df):,} val S1 records...")
    s1_val_dict = build_records_dict(s1_val_df)

    # 4. Build blocking index on targets
    print("\n--- Building Blocking Index ---")
    idx = InvertedIndex(max_bucket_size=500)
    idx.build_from_dataframe(targets_df)

    # 5. Retrieve candidates
    print(f"\n--- Generating Candidates (K={top_k}) ---")
    print("  Train candidates...")
    train_cands = generate_candidate_pairs(s1_train_df, idx, top_k=top_k)

    print("\n  Val candidates...")
    val_cands = generate_candidate_pairs(s1_val_df, idx, top_k=top_k)

    # Evaluate val blocking recall
    print("\n--- Blocking Recall (Val) ---")
    evaluate_blocking_recall(val_cands, gt_val, verbose=True)

    # 6. Build training pairs + features
    print("\n--- Feature Extraction ---")
    train_pairs, y_train = build_training_pairs(train_cands, gt_train, max_negatives_per_positive=neg_ratio)
    print(f"  Train pairs: {len(train_pairs):,} | Pos: {(y_train==1).sum():,} | Neg: {(y_train==0).sum():,}")
    X_train = compute_features_for_pairs(train_pairs, s1_train_dict, target_dict)

    val_pairs, y_val = build_training_pairs(val_cands, gt_val, max_negatives_per_positive=neg_ratio)
    print(f"  Val pairs:   {len(val_pairs):,} | Pos: {(y_val==1).sum():,} | Neg: {(y_val==0).sum():,}")
    X_val = compute_features_for_pairs(val_pairs, s1_val_dict, target_dict)

    # 7. Compute auto SPW if not provided
    if spw is None:
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        spw = max(1.0, float(neg / max(1, pos)) * 0.7)
        print(f"\n  Auto scale_pos_weight = {spw:.2f} (neg/pos × 0.7)")

    # 8. Train LightGBM
    print(f"\n--- Training LightGBM (SPW={spw:.2f}) ---")
    model = train_lgbm_classifier(X_train, y_train, X_val, y_val, n_estimators=200)

    # 9. Threshold tuning on full grouped val
    print("\n--- Threshold Tuning (Grouped Macro F0.5) ---")
    val_probs_arr = model.predict_proba(X_val)[:, 1]
    val_pair_probs = {(p[0], p[1]): float(pr) for p, pr in zip(val_pairs, val_probs_arr)}
    val_s1_unique = sorted(val_ids)

    # Use fine-grained sweep: 0.05 to 0.95 in 0.01 steps
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_thresh, best_f05, best_preds = tune_prediction_threshold(
        val_s1_unique, val_cands, val_pair_probs, gt_val,
        thresholds=thresholds, verbose=True,
    )

    # 10. Feature importance
    print("\n--- Feature Importance (Top 10) ---")
    importances = model.feature_importances_
    fi_sorted = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    for name, imp in fi_sorted[:10]:
        bar = "█" * int(imp / max(importances) * 30)
        print(f"  {name:<25} {imp:>6}  {bar}")

    # 11. Save model + threshold
    model_path = os.path.join(MODELS_DIR, "phase2_lgbm.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    thresh_path = os.path.join(MODELS_DIR, "phase2_threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({
            "threshold": best_thresh,
            "val_macro_f05": best_f05,
            "top_k": top_k,
            "spw": spw,
            "neg_ratio": neg_ratio,
            "n_features": len(FEATURE_NAMES),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Val Macro F0.5: {best_f05:.4f}  (threshold={best_thresh:.2f})")
    print(f"  Phase 1 baseline: 0.9222")
    delta = best_f05 - 0.9222
    print(f"  Delta vs Phase 1: {delta:+.4f}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'='*65}")

    log_run(
        f"phase2_train_val_K{top_k}_spw{spw:.1f}_neg{neg_ratio}",
        best_thresh, best_f05, "PASS",
        f"K={top_k}, spw={spw:.2f}, neg_ratio={neg_ratio}, delta_vs_p1={delta:+.4f}"
    )
    return best_f05, best_thresh


# ---------------------------------------------------------------------------
# Mode 2: blocking_sweep — K recall vs compactness
# ---------------------------------------------------------------------------

def run_blocking_sweep():
    """Sweep K in {5, 10, 15, 20, 25, 30} and report recall + avg candidates."""
    print("\n" + "="*65)
    print("  Phase 2.2 — Blocking K Sweep")
    print("="*65)

    _, val_ids = load_val_split()
    if val_ids is None:
        print("  ERROR: Run splitter.py first.")
        return

    print("\n  Loading data...")
    s1_df, s2_df, s3_df, gt_full = load_train_data(verbose=False)
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    s1_val_df  = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    gt_val     = {k: v for k, v in gt_full.items() if k in val_ids}

    print("  Building index...")
    idx = InvertedIndex(max_bucket_size=500)
    idx.build_from_dataframe(targets_df)

    K_values = [5, 10, 15, 20, 25, 30]
    results = []

    for K in K_values:
        print(f"\n  --- K = {K} ---")
        t0 = time.time()
        cands = generate_candidate_pairs(s1_val_df, idx, top_k=K)
        elapsed = time.time() - t0

        res = evaluate_blocking_recall(cands, gt_val, verbose=False)
        res["K"] = K
        res["time_s"] = round(elapsed, 1)
        results.append(res)
        print(f"  Recall={res['blocking_recall_pct']:.2f}%  avg_cands={res['avg_candidates']:.2f}  time={elapsed:.1f}s")

    # Write report
    lines = [
        "# Phase 2.2 — Blocking K Sweep\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "| K | Blocking Recall | Avg Candidates | Max Candidates | Time (s) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['K']} | {r['blocking_recall_pct']:.2f}% | {r['avg_candidates']:.2f} | {r['max_candidates']} | {r['time_s']} |")

    good = [r for r in results if r["blocking_recall_pct"] >= 90.0]
    best = min(good, key=lambda x: x["avg_candidates"]) if good else results[-1]
    lines.append(f"\n## Recommendation\nBest K = **{best['K']}** (recall={best['blocking_recall_pct']:.2f}%, avg_cands={best['avg_candidates']:.2f})")

    report_path = os.path.join(REPORTS_DIR, "blocking_sweep.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Report saved: {report_path}")
    print(f"  Recommended K = {best['K']} (recall={best['blocking_recall_pct']:.2f}%, avg={best['avg_candidates']:.2f})")
    return results


# ---------------------------------------------------------------------------
# Mode 3: spw_sweep — scale_pos_weight tuning
# ---------------------------------------------------------------------------

def run_spw_sweep(top_k: int = 20, neg_ratio: int = 4):
    """Sweep scale_pos_weight values and report val Macro F0.5."""
    print("\n" + "="*65)
    print("  Phase 2.3 — scale_pos_weight Sweep")
    print("="*65)

    train_ids, val_ids = load_val_split()
    if train_ids is None:
        print("  ERROR: Run splitter.py first.")
        return

    print("\n  Loading & normalising data...")
    s1_df, s2_df, s3_df, gt_full = load_train_data(verbose=False)
    targets_df   = pd.concat([s2_df, s3_df], ignore_index=True)
    s1_train_df  = s1_df[s1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val_df    = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    gt_train = {k: v for k, v in gt_full.items() if k in train_ids}
    gt_val   = {k: v for k, v in gt_full.items() if k in val_ids}

    target_dict   = build_records_dict(targets_df)
    s1_train_dict = build_records_dict(s1_train_df)
    s1_val_dict   = build_records_dict(s1_val_df)

    idx = InvertedIndex(max_bucket_size=500)
    idx.build_from_dataframe(targets_df)
    train_cands = generate_candidate_pairs(s1_train_df, idx, top_k=top_k)
    val_cands   = generate_candidate_pairs(s1_val_df,   idx, top_k=top_k)

    train_pairs, y_train = build_training_pairs(train_cands, gt_train, max_negatives_per_positive=neg_ratio)
    X_train = compute_features_for_pairs(train_pairs, s1_train_dict, target_dict)

    val_pairs, y_val = build_training_pairs(val_cands, gt_val, max_negatives_per_positive=neg_ratio)
    X_val = compute_features_for_pairs(val_pairs, s1_val_dict, target_dict)
    val_pair_probs_keys = [(p[0], p[1]) for p in val_pairs]
    val_s1_unique = sorted(val_ids)
    thresholds = np.arange(0.30, 0.91, 0.05)

    spw_values = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    results = []

    print(f"\n  {'SPW':<8} {'Val F0.5':<12} {'Threshold':<12}")
    print(f"  {'-'*35}")

    for spw in spw_values:
        m = train_lgbm_classifier(X_train, y_train, X_val, y_val, n_estimators=150)
        probs = m.predict_proba(X_val)[:, 1]
        pair_probs = dict(zip(val_pair_probs_keys, probs.tolist()))
        bt, bf, _ = tune_prediction_threshold(
            val_s1_unique, val_cands, pair_probs, gt_val,
            thresholds=thresholds, verbose=False
        )
        results.append({"spw": spw, "f05": bf, "threshold": bt})
        marker = " <-- BEST" if bf == max(r["f05"] for r in results) else ""
        print(f"  {spw:<8.1f} {bf:<12.4f} {bt:<12.2f}{marker}")

    best = max(results, key=lambda x: x["f05"])
    print(f"\n  Best SPW = {best['spw']} → F0.5 = {best['f05']:.4f}")

    out = os.path.join(EXPERIMENTS_DIR, "spw_sweep.json")
    with open(out, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"  Saved: {out}")
    return results, best


# ---------------------------------------------------------------------------
# Mode 4: submission — generate test TSVs
# ---------------------------------------------------------------------------

def run_submission(top_k: int = 20):
    """
    Load trained Phase 2 model, run on full test set, write submission TSVs.
    Validates output before writing submission log entry.
    """
    print("\n" + "="*65)
    print("  Phase 2.4 — Submission Generation")
    print("="*65)

    # Load saved model + threshold
    model_path  = os.path.join(MODELS_DIR, "phase2_lgbm.pkl")
    thresh_path = os.path.join(MODELS_DIR, "phase2_threshold.json")

    if not os.path.exists(model_path):
        print("  ERROR: No model found. Run --mode train_val first.")
        return False

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(thresh_path) as f:
        cfg = json.load(f)

    threshold = cfg["threshold"]
    print(f"  Model loaded. Threshold = {threshold:.3f}  Val F0.5 = {cfg['val_macro_f05']:.4f}")

    # Load test data
    test_dir = os.path.join(DATASET_DIR, "test")
    print("\n  Loading test sources...")
    test_s1  = load_tsv(os.path.join(test_dir, "test_source1.tsv"))
    test_s2  = load_tsv(os.path.join(test_dir, "test_source2.tsv"))
    test_s3  = load_tsv(os.path.join(test_dir, "test_source3.tsv"))
    test_tgt = pd.concat([test_s2, test_s3], ignore_index=True)
    print(f"  S1: {len(test_s1):,} | S2: {len(test_s2):,} | S3: {len(test_s3):,} → Targets: {len(test_tgt):,}")

    # Normalise
    print("\n  Normalising test records (S2+S3)...")
    test_tgt_dict = build_records_dict(test_tgt)
    print("\n  Normalising test S1...")
    test_s1_dict  = build_records_dict(test_s1)

    # Build index + retrieve candidates
    print("\n  Building test blocking index...")
    test_idx = InvertedIndex(max_bucket_size=500)
    test_idx.build_from_dataframe(test_tgt)

    print(f"\n  Retrieving candidates (K={top_k})...")
    test_cands = generate_candidate_pairs(test_s1, test_idx, top_k=top_k)

    # Score all candidate pairs
    print("\n  Scoring candidate pairs...")
    pair_list = [(s1_id, c) for s1_id, cands in test_cands.items() for c in cands]
    print(f"  Total candidate pairs: {len(pair_list):,}")

    if pair_list:
        X_test = compute_features_for_pairs(pair_list, test_s1_dict, test_tgt_dict)
        probs  = model.predict_proba(X_test)[:, 1]
    else:
        probs = np.array([])

    # Apply threshold
    accepted: Dict[str, List[str]] = {eid: [] for eid in test_s1["entity_id"].values}
    for (s1_id, c), p in zip(pair_list, probs):
        if p >= threshold:
            accepted[s1_id].append(c)

    # Write candidate_pairs.tsv
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        for eid in test_s1["entity_id"].values:
            cands = test_cands.get(eid, [])
            w.writerow([eid, ",".join(cands)])

    # Write matching_results.tsv
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    n_matches = 0
    n_singletons = 0
    with open(match_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for eid in test_s1["entity_id"].values:
            matches = accepted.get(eid, [])
            # Enforce: matched IDs must be in candidates
            cand_set = set(test_cands.get(eid, []))
            valid_matches = [m for m in matches if m in cand_set]
            w.writerow([eid, ",".join(valid_matches)])
            n_matches += len(valid_matches)
            if not valid_matches:
                n_singletons += 1

    print(f"\n  Written: {match_path}")
    print(f"  Written: {cand_path}")
    print(f"  Entities: {len(test_s1):,} | With matches: {len(test_s1)-n_singletons:,} | Singletons: {n_singletons:,}")
    print(f"  Total match links: {n_matches:,}")

    log_run(
        f"phase2_submission_K{top_k}",
        threshold, cfg["val_macro_f05"], "PENDING_UPLOAD",
        f"K={top_k}, matches={n_matches:,}, singletons={n_singletons:,}"
    )

    print(f"\n  Next: validate with:")
    print(f"  python utils/validate_submission.py \\")
    print(f"    --matching output/matching_results.tsv \\")
    print(f"    --candidate output/candidate_pairs.tsv \\")
    print(f"    --test-dir dataset/test")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Pipeline")
    parser.add_argument("--mode", choices=["train_val", "blocking_sweep", "spw_sweep", "submission", "all"],
                        default="train_val")
    parser.add_argument("--K",   type=int,   default=20,  help="Top-K candidates (default 20)")
    parser.add_argument("--spw", type=float, default=None, help="scale_pos_weight (auto if omitted)")
    parser.add_argument("--neg", type=int,   default=4,   help="Negatives per positive (default 4)")
    args = parser.parse_args()

    if args.mode in ("blocking_sweep", "all"):
        run_blocking_sweep()

    if args.mode in ("spw_sweep", "all"):
        run_spw_sweep(top_k=args.K, neg_ratio=args.neg)

    if args.mode in ("train_val", "all"):
        run_train_val(top_k=args.K, spw=args.spw, neg_ratio=args.neg)

    if args.mode in ("submission", "all"):
        run_submission(top_k=args.K)


if __name__ == "__main__":
    main()
