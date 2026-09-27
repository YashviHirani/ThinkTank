"""
Phase 1, Step 5.4: End-to-End Baseline Pipeline Orchestrator
============================================================
Connects data loading, normalization, inverted-index blocking, feature extraction,
LightGBM training, grouped macro F0.5 threshold tuning, test prediction, TSV generation,
and submission validation.

Usage:
    # Quick dry-run sanity check on sample data:
    python src/pipeline.py --dry-run

    # Full training and submission generation:
    python src/pipeline.py --full
"""

import os
import sys
import time
import argparse
import pickle
import csv
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Any

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.normalizer import (
    normalize_general_text,
    extract_core_name,
    get_sorted_tokens,
    normalize_address,
    extract_numeric_tokens,
    extract_postcode_candidate,
)
from src.blocker import (
    InvertedIndex,
    generate_candidate_pairs,
    evaluate_blocking_recall,
)
from src.features import compute_features_for_pairs, FEATURE_NAMES
from src.model import (
    build_training_pairs,
    train_lgbm_classifier,
    tune_prediction_threshold,
)
from src.scorer import macro_f05
from utils.validate_submission import validate_submission


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

LOG_FILE = os.path.join(EXPERIMENTS_DIR, "submission_log.csv")


# ---------------------------------------------------------------------------
# Helper: Fast Record Normalisation Cache
# ---------------------------------------------------------------------------

def build_normalized_records_dict(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Builds a fast lookup dictionary of normalized entity attributes.
    Key: entity_id -> dict of normalized views.
    """
    records = {}
    ids = df["entity_id"].values
    names = df["business_name"].fillna("").values
    addrs = df["business_address"].fillna("").values
    countries = df["country"].fillna("").str.strip().str.upper().values

    for i in range(len(df)):
        clean_n = normalize_general_text(names[i])
        cn, suf = extract_core_name(clean_n)
        clean_a = normalize_address(addrs[i])
        nums = extract_numeric_tokens(clean_a)
        pc = extract_postcode_candidate(clean_a)

        records[ids[i]] = {
            "core_name": cn,
            "legal_suffix": suf,
            "sorted_name": get_sorted_tokens(cn),
            "clean_address": clean_a,
            "numeric_tokens": nums,
            "postcode": pc,
            "country": countries[i],
        }
    return records


# ---------------------------------------------------------------------------
# Logging Function
# ---------------------------------------------------------------------------

def log_submission_entry(
    submission_name: str,
    threshold: float,
    val_f05: float,
    status: str,
    notes: str = "",
):
    """Appends submission metadata to experiments/submission_log.csv."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "submission_name", "threshold", "val_macro_f05", "status", "notes"
            ])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            submission_name,
            f"{threshold:.2f}",
            f"{val_f05:.4f}",
            status,
            notes,
        ])
    print(f"  Recorded in {LOG_FILE}")


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    dry_run: bool = False,
    sample_size: Optional[int] = None,
    top_k: int = 20,
):
    t_start = time.time()
    print("=" * 65)
    print("  Amazon ML Challenge 2026 — Baseline Pipeline Run")
    mode = "DRY-RUN (Sample)" if dry_run else "FULL RUN"
    print(f"  Mode: {mode}")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    print("\n[Step 1/6] Loading source data...")

    train_dir = os.path.join(DATASET_DIR, "train")
    test_dir = os.path.join(DATASET_DIR, "test")

    nrows_target = 25000 if dry_run else (sample_size or None)
    nrows_s1 = 2000 if dry_run else (sample_size or None)

    if dry_run:
        print(f"  Sampling {nrows_target:,} target rows from S2 and S3...")
        train_s2 = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", dtype=str, nrows=nrows_target)
        train_s3 = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", dtype=str, nrows=nrows_target)
        train_targets_df = pd.concat([train_s2, train_s3], ignore_index=True)
        target_ids_set = set(train_targets_df["entity_id"])

        print("  Finding S1 entities that match sample targets...")
        gt_df = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str, nrows=60000)
        gt_matches = {}
        matched_s1_ids = []
        for _, r in gt_df.iterrows():
            val = str(r["matched_entity_ids"]).strip()
            m_set = set(x.strip() for x in val.split(",") if x.strip()) if val and val != "nan" else set()
            valid_m = m_set & target_ids_set
            s1_id = r["source1_entity_id"]
            if valid_m:
                gt_matches[s1_id] = valid_m
                matched_s1_ids.append(s1_id)
            elif len(matched_s1_ids) < nrows_s1 // 10 and len(m_set) == 0:  # add singletons
                gt_matches[s1_id] = set()
                matched_s1_ids.append(s1_id)
            if len(matched_s1_ids) >= nrows_s1:
                break

        print(f"    Selected {len(matched_s1_ids):,} S1 entities for dry-run.")
        s1_needed = set(matched_s1_ids)
        s1_chunks = []
        for chunk in pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", dtype=str, chunksize=50000):
            hit = chunk[chunk["entity_id"].isin(s1_needed)]
            if len(hit) > 0:
                s1_chunks.append(hit)
                s1_needed -= set(hit["entity_id"])
            if not s1_needed:
                break
        train_s1 = pd.concat(s1_chunks, ignore_index=True) if s1_chunks else pd.DataFrame()
    else:
        print(f"  Loading train sources (nrows_target={nrows_target}, nrows_s1={nrows_s1})...")
        train_s2 = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", dtype=str, nrows=nrows_target)
        train_s3 = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", dtype=str, nrows=nrows_target)
        train_targets_df = pd.concat([train_s2, train_s3], ignore_index=True)
        train_s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", dtype=str, nrows=nrows_s1)

        print("  Loading ground truth...")
        gt_df = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str, nrows=nrows_s1 * 3 if nrows_s1 else None)
        gt_matches = {}
        for _, r in gt_df.iterrows():
            val = str(r["matched_entity_ids"]).strip()
            gt_matches[r["source1_entity_id"]] = set(x.strip() for x in val.split(",") if x.strip()) if val and val != "nan" else set()

    # Normalize records dictionaries
    print("  Building normalized attribute caches...")
    s1_dict = build_normalized_records_dict(train_s1)
    target_dict = build_normalized_records_dict(train_targets_df)

    # ------------------------------------------------------------------
    # 2. Inverted Index Blocking on Training Targets
    # ------------------------------------------------------------------
    print("\n[Step 2/6] Building inverted index on training targets...")
    train_index = InvertedIndex(max_bucket_size=500)
    train_index.build_from_dataframe(train_targets_df)

    print(f"  Generating candidate pairs for training S1 entities (Top-K={top_k})...")
    train_candidates = generate_candidate_pairs(train_s1, train_index, top_k=top_k)

    # Evaluate blocking recall
    eval_res = evaluate_blocking_recall(train_candidates, gt_matches, verbose=True)

    # ------------------------------------------------------------------
    # 3. Feature Extraction & LightGBM Model Training
    # ------------------------------------------------------------------
    print("\n[Step 3/6] Assembling labeled pairs & computing features...")
    train_pairs, y_train = build_training_pairs(train_candidates, gt_matches, max_negatives_per_positive=4)
    print(f"  Extracted {len(train_pairs):,} labeled pairs (Positives: {(y_train==1).sum():,}, Negatives: {(y_train==0).sum():,})")

    X_train = compute_features_for_pairs(train_pairs, s1_dict, target_dict)

    # Split for threshold tuning
    n_total = len(X_train)
    val_split_idx = int(n_total * 0.8)
    X_tr, y_tr = X_train[:val_split_idx], y_train[:val_split_idx]
    X_vl, y_vl = X_train[val_split_idx:], y_train[val_split_idx:]
    pairs_val = train_pairs[val_split_idx:]

    print("\n  Training LightGBM classifier...")
    model = train_lgbm_classifier(X_tr, y_tr, X_vl, y_vl, n_estimators=100)

    # Save model
    model_path = os.path.join(MODELS_DIR, "baseline_lgbm.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to {model_path}")

    # ------------------------------------------------------------------
    # 4. Threshold Tuning on Grouped Validation Pairs
    # ------------------------------------------------------------------
    print("\n[Step 4/6] Tuning probability threshold for Macro F0.5...")
    val_probs = model.predict_proba(X_vl)[:, 1]

    val_pair_probs = {(p[0], p[1]): float(prob) for p, prob in zip(pairs_val, val_probs)}
    val_s1_unique = list(set(p[0] for p in pairs_val))

    best_thresh, best_val_f05, _ = tune_prediction_threshold(
        val_s1_unique, train_candidates, val_pair_probs, gt_matches, verbose=True
    )

    # ------------------------------------------------------------------
    # 5. Generate Test Candidates & Predictions
    # ------------------------------------------------------------------
    print("\n[Step 5/6] Generating Test Set Submissions...")

    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"  Loading test entities...")
    test_s1_df = pd.read_csv(test_s1_path, sep="\t", dtype=str, nrows=nrows_s1)
    test_s2_df = pd.read_csv(test_s2_path, sep="\t", dtype=str, nrows=nrows_target)
    test_s3_df = pd.read_csv(test_s3_path, sep="\t", dtype=str, nrows=nrows_target)
    test_targets_df = pd.concat([test_s2_df, test_s3_df], ignore_index=True)

    print("  Building test inverted index...")
    test_index = InvertedIndex(max_bucket_size=500)
    test_index.build_from_dataframe(test_targets_df)

    print(f"  Generating candidates for test S1 (Top-K={top_k})...")
    test_candidates = generate_candidate_pairs(test_s1_df, test_index, top_k=top_k)

    # Write output/candidate_pairs.tsv
    cand_out_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    print(f"  Writing {cand_out_path}...")
    with open(cand_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in test_s1_df["entity_id"].values:
            cands = test_candidates.get(s1_id, [])
            writer.writerow([s1_id, ",".join(cands)])

    # Predict test matches
    print("  Extracting test features and scoring candidates...")
    test_s1_dict = build_normalized_records_dict(test_s1_df)
    test_target_dict = build_normalized_records_dict(test_targets_df)

    test_pair_list = []
    for s1_id, cands in test_candidates.items():
        for c in cands:
            test_pair_list.append((s1_id, c))

    if test_pair_list:
        X_test = compute_features_for_pairs(test_pair_list, test_s1_dict, test_target_dict)
        test_probs = model.predict_proba(X_test)[:, 1]
    else:
        test_probs = np.array([])

    test_accepted: Dict[str, List[str]] = {s1_id: [] for s1_id in test_s1_df["entity_id"].values}
    for (s1_id, c), prob in zip(test_pair_list, test_probs):
        if prob >= best_thresh:
            test_accepted[s1_id].append(c)

    # Write output/matching_results.tsv
    match_out_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    print(f"  Writing {match_out_path}...")
    with open(match_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in test_s1_df["entity_id"].values:
            matches = test_accepted.get(s1_id, [])
            writer.writerow([s1_id, ",".join(matches)])

    # ------------------------------------------------------------------
    # 6. Submission Validation Gate
    # ------------------------------------------------------------------
    print("\n[Step 6/6] Running Submission Validator...")
    # For dry-run, we validate schema & format
    if not dry_run:
        is_valid = validate_submission(match_out_path, cand_out_path, test_dir)
    else:
        # Check basic schema on sample outputs
        m_df = pd.read_csv(match_out_path, sep="\t", dtype=str, keep_default_na=False)
        c_df = pd.read_csv(cand_out_path, sep="\t", dtype=str, keep_default_na=False)
        is_valid = (
            list(m_df.columns) == ["source1_entity_id", "matched_entity_ids"]
            and list(c_df.columns) == ["source1_entity_id", "candidate_entity_ids"]
            and len(m_df) == len(test_s1_df)
            and len(c_df) == len(test_s1_df)
        )
        if is_valid:
            print("\n  RESULT: PASS (Dry-run sample outputs valid!)")

    log_submission_entry(
        submission_name="phase1_baseline" + ("_dryrun" if dry_run else ""),
        threshold=best_thresh,
        val_f05=best_val_f05,
        status="PASS" if is_valid else "FAIL",
        notes=f"K={top_k}, dry_run={dry_run}",
    )

    elapsed = time.time() - t_start
    print(f"\nPipeline finished in {elapsed:.1f}s.")
    return is_valid


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 End-to-End Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run fast dry-run on sample records")
    parser.add_argument("--full", action="store_true", help="Run full pipeline on complete dataset")
    parser.add_argument("--sample", type=int, default=None, help="Run on specific sample size")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K candidates per S1 entity")
    args = parser.parse_args()

    dry_run = args.dry_run or (not args.full and args.sample is None)
    run_pipeline(dry_run=dry_run, sample_size=args.sample, top_k=args.top_k)


if __name__ == "__main__":
    main()
