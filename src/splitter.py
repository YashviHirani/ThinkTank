"""
Phase 0, Step 6: Train/Validation Split
=========================================
Multi-stratified 80/20 grouped split on source1_entity_id.
Stratifies on country + match-count bucket. No hard-coded countries.

Usage:
    python src/splitter.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

GT_PATH = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")
S1_PATH = os.path.join(DATASET_DIR, "train", "train_source1.tsv")

TRAIN_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "train_s1_ids.txt")
VAL_IDS_PATH = os.path.join(EXPERIMENTS_DIR, "val_s1_ids.txt")
METADATA_PATH = os.path.join(EXPERIMENTS_DIR, "split_metadata.json")

SEED = 42
TEST_SIZE = 0.20
DIFF_THRESHOLD = 0.5  # flag if train% and val% differ by more than this


# ---------------------------------------------------------------------------
# Safe match counting — handles trailing commas, double commas, whitespace
# ---------------------------------------------------------------------------

def safe_match_count(value: str) -> int:
    """Count match IDs safely by splitting, filtering empties, and counting."""
    if not value or value.strip() == "":
        return 0
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    return len(tokens)


# ---------------------------------------------------------------------------
# Match-count bucketing
# ---------------------------------------------------------------------------

def match_bucket(count: int) -> str:
    """Assign a match count to a bucket."""
    if count == 0:
        return "singleton"
    elif count <= 2:
        return "low"
    elif count <= 4:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Balance report
# ---------------------------------------------------------------------------

def balance_report(train_df: pd.DataFrame, val_df: pd.DataFrame,
                   all_df: pd.DataFrame) -> list:
    """Generate balance report lines comparing train vs val distributions."""
    lines = []
    L = lines.append

    L("\n" + "=" * 70)
    L("  Split Balance Report")
    L("=" * 70)

    n_train = len(train_df)
    n_val = len(val_df)
    n_total = n_train + n_val

    L(f"\n  Total S1 entities:  {n_total:>10,}")
    L(f"  Train:              {n_train:>10,} ({n_train/n_total*100:.1f}%)")
    L(f"  Validation:         {n_val:>10,} ({n_val/n_total*100:.1f}%)")

    # --- Country distribution ---
    L(f"\n  {'--- Country Distribution ---':^60}")
    L(f"  {'Category':<20} {'Train':>10} {'Train%':>8} {'Val':>10} {'Val%':>8} {'Diff':>6} {'':>6}")

    for country in sorted(all_df["country"].unique()):
        t_count = (train_df["country"] == country).sum()
        v_count = (val_df["country"] == country).sum()
        t_pct = t_count / n_train * 100
        v_pct = v_count / n_val * 100
        diff = abs(t_pct - v_pct)
        flag = "[WARN]" if diff > DIFF_THRESHOLD else "[OK]"
        L(f"  {country:<20} {t_count:>10,} {t_pct:>7.1f}% {v_count:>10,} {v_pct:>7.1f}% {diff:>5.2f}pp {flag}")

    # --- Match-count bucket distribution (country-independent) ---
    L(f"\n  {'--- Match-Count Bucket Distribution ---':^60}")
    L(f"  {'Bucket':<20} {'Train':>10} {'Train%':>8} {'Val':>10} {'Val%':>8} {'Diff':>6} {'':>6}")

    for bucket in ["singleton", "low", "medium", "high"]:
        t_count = (train_df["match_bucket"] == bucket).sum()
        v_count = (val_df["match_bucket"] == bucket).sum()
        t_pct = t_count / n_train * 100
        v_pct = v_count / n_val * 100
        diff = abs(t_pct - v_pct)
        flag = "[WARN]" if diff > DIFF_THRESHOLD else "[OK]"
        L(f"  {bucket:<20} {t_count:>10,} {t_pct:>7.1f}% {v_count:>10,} {v_pct:>7.1f}% {diff:>5.2f}pp {flag}")

    # --- Combined country + bucket distribution ---
    L(f"\n  {'--- Combined Country + Bucket Distribution ---':^60}")
    L(f"  {'Strat Key':<25} {'Train':>10} {'Train%':>8} {'Val':>10} {'Val%':>8} {'Diff':>6} {'':>6}")

    for key in sorted(all_df["strat_key"].unique()):
        t_count = (train_df["strat_key"] == key).sum()
        v_count = (val_df["strat_key"] == key).sum()
        t_pct = t_count / n_train * 100
        v_pct = v_count / n_val * 100
        diff = abs(t_pct - v_pct)
        flag = "[WARN]" if diff > DIFF_THRESHOLD else "[OK]"
        L(f"  {key:<25} {t_count:>10,} {t_pct:>7.1f}% {v_count:>10,} {v_pct:>7.1f}% {diff:>5.2f}pp {flag}")

    # --- Average and median match counts ---
    L(f"\n  {'--- Match Count Statistics ---':^60}")
    L(f"  {'Metric':<30} {'Train':>12} {'Val':>12}")
    L(f"  {'Average match count':<30} {train_df['match_count'].mean():>12.4f} {val_df['match_count'].mean():>12.4f}")
    L(f"  {'Median match count':<30} {train_df['match_count'].median():>12.1f} {val_df['match_count'].median():>12.1f}")
    L(f"  {'Max match count':<30} {train_df['match_count'].max():>12} {val_df['match_count'].max():>12}")

    L("=" * 70)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # ==================================================================
    # 1. Load data — only what's needed
    # ==================================================================
    print("Loading ground truth...")
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False)

    print("Loading train_source1 (entity_id + country only)...")
    s1 = pd.read_csv(S1_PATH, sep="\t", dtype=str, keep_default_na=False,
                      usecols=["entity_id", "country"])

    # ==================================================================
    # 2. Validate ground truth
    # ==================================================================
    print("Validating data...")

    # Assert expected columns
    assert set(gt.columns) == {"source1_entity_id", "matched_entity_ids"}, \
        f"Unexpected GT columns: {gt.columns.tolist()}"

    # Assert unique S1 IDs in ground truth
    n_dupes = gt["source1_entity_id"].duplicated().sum()
    assert n_dupes == 0, f"Ground truth has {n_dupes} duplicate source1_entity_ids"

    # Assert no empty S1 IDs
    assert (gt["source1_entity_id"].str.strip() == "").sum() == 0, \
        "Ground truth has empty source1_entity_ids"

    # Assert all S1 IDs have correct prefix
    assert gt["source1_entity_id"].str.startswith("S1-").all(), \
        "Ground truth has source1_entity_ids without S1- prefix"

    # ==================================================================
    # 3. Merge country — validate one-to-one
    # ==================================================================
    n_before = len(gt)
    gt = gt.merge(s1, left_on="source1_entity_id", right_on="entity_id", how="left")

    # Validate merge
    assert len(gt) == n_before, \
        f"Merge changed row count: {n_before} -> {len(gt)} (possible many-to-one)"

    n_missing_country = gt["country"].isna().sum() + (gt["country"].str.strip() == "").sum()
    assert n_missing_country == 0, \
        f"{n_missing_country} S1 entities have no country after merge"

    # Drop the duplicate entity_id column from merge
    gt = gt.drop(columns=["entity_id"])

    print(f"  Merged {n_before:,} S1 entities with country. All matched.")

    # ==================================================================
    # 4. Compute match counts safely
    # ==================================================================
    print("Computing match counts...")
    gt["match_count"] = gt["matched_entity_ids"].apply(safe_match_count)

    # Sanity check: match counts should be non-negative
    assert (gt["match_count"] >= 0).all(), "Negative match counts detected"

    # ==================================================================
    # 5. Build stratification key
    # ==================================================================
    gt["match_bucket"] = gt["match_count"].apply(match_bucket)
    gt["strat_key"] = gt["country"] + "_" + gt["match_bucket"]

    # Check bucket sizes
    bucket_counts = gt["strat_key"].value_counts()
    print(f"\n  Stratification buckets ({len(bucket_counts)}):")
    for key, count in bucket_counts.items():
        pct = count / len(gt) * 100
        warn = " [WARN: very small]" if count < 10 else ""
        print(f"    {key:<25} {count:>10,} ({pct:.2f}%){warn}")

    # Check if any bucket is too small for stratified split
    min_bucket = bucket_counts.min()
    min_bucket_name = bucket_counts.idxmin()
    if min_bucket < 2:
        print(f"\n  [ERROR] Bucket '{min_bucket_name}' has only {min_bucket} entities.")
        print(f"  Cannot perform stratified split. Merging small buckets...")
        # Merge tiny buckets: replace their strat_key with country + "_rare"
        rare_keys = bucket_counts[bucket_counts < 5].index
        gt.loc[gt["strat_key"].isin(rare_keys), "strat_key"] = \
            gt.loc[gt["strat_key"].isin(rare_keys), "country"] + "_rare"
        bucket_counts = gt["strat_key"].value_counts()
        print(f"  After merging: {len(bucket_counts)} buckets, min size = {bucket_counts.min()}")

    # ==================================================================
    # 6. Split
    # ==================================================================
    print(f"\nSplitting (test_size={TEST_SIZE}, random_state={SEED})...")

    train_idx, val_idx = train_test_split(
        gt.index,
        test_size=TEST_SIZE,
        stratify=gt["strat_key"],
        random_state=SEED,
    )

    train_df = gt.loc[train_idx]
    val_df = gt.loc[val_idx]

    train_ids = set(train_df["source1_entity_id"])
    val_ids = set(val_df["source1_entity_id"])

    # ==================================================================
    # 7. Post-split assertions
    # ==================================================================
    print("Verifying split integrity...")

    # Zero overlap
    overlap = train_ids & val_ids
    assert len(overlap) == 0, \
        f"LEAKAGE: {len(overlap)} S1 IDs appear in both train and val"

    # Complete coverage
    all_ids = set(gt["source1_entity_id"])
    assert train_ids | val_ids == all_ids, \
        f"Coverage gap: {len(all_ids - (train_ids | val_ids))} S1 IDs missing from split"

    # No duplicates within each set
    assert len(train_ids) == len(train_df), "Duplicate S1 IDs in train split"
    assert len(val_ids) == len(val_df), "Duplicate S1 IDs in val split"

    # All IDs have S1- prefix
    assert all(i.startswith("S1-") for i in train_ids), "Non-S1 ID in train split"
    assert all(i.startswith("S1-") for i in val_ids), "Non-S1 ID in val split"

    print("  [OK] Zero overlap")
    print("  [OK] Complete coverage")
    print("  [OK] No duplicates")
    print("  [OK] All IDs have S1- prefix")

    # ==================================================================
    # 8. Balance report
    # ==================================================================
    report_lines = balance_report(train_df, val_df, gt)
    for line in report_lines:
        print(line)

    # ==================================================================
    # 9. Save outputs
    # ==================================================================
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # Save ID files — sorted for deterministic output
    train_sorted = sorted(train_ids)
    val_sorted = sorted(val_ids)

    with open(TRAIN_IDS_PATH, "w") as f:
        f.write("\n".join(train_sorted))

    with open(VAL_IDS_PATH, "w") as f:
        f.write("\n".join(val_sorted))

    print(f"\n  Saved {len(train_sorted):,} train IDs to {TRAIN_IDS_PATH}")
    print(f"  Saved {len(val_sorted):,} val IDs to {VAL_IDS_PATH}")

    # Save metadata
    metadata = {
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "total_s1_entities": len(gt),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_pct": round(len(train_ids) / len(gt) * 100, 2),
        "val_pct": round(len(val_ids) / len(gt) * 100, 2),
        "stratification_buckets": {
            k: int(v) for k, v in gt["strat_key"].value_counts().items()
        },
        "train_avg_match_count": round(float(train_df["match_count"].mean()), 4),
        "val_avg_match_count": round(float(val_df["match_count"].mean()), 4),
        "train_median_match_count": round(float(train_df["match_count"].median()), 1),
        "val_median_match_count": round(float(val_df["match_count"].median()), 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata to {METADATA_PATH}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
