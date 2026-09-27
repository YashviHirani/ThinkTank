"""
Phase 0, Steps 2-3: Data Loading & Ground Truth Validation
===========================================================
Loads all 7 TSV files, validates schemas, parses ground truth
into Python sets, and runs cross-reference integrity assertions.

Usage:
    # As a standalone sanity check:
    python src/data_loader.py

    # As a module in the pipeline:
    from src.data_loader import load_all_data
    data = load_all_data()
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SOURCE_FILES = {
    "train_s1": ("train", "train_source1.tsv", "S1-"),
    "train_s2": ("train", "train_source2.tsv", "S2-"),
    "train_s3": ("train", "train_source3.tsv", "S3-"),
    "test_s1":  ("test",  "test_source1.tsv",  "S1-"),
    "test_s2":  ("test",  "test_source2.tsv",  "S2-"),
    "test_s3":  ("test",  "test_source3.tsv",  "S3-"),
}

GROUND_TRUTH_FILE = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")

EXPECTED_SOURCE_COLS = {"entity_id", "business_name", "business_address", "country"}
EXPECTED_GT_COLS = {"source1_entity_id", "matched_entity_ids"}


# ---------------------------------------------------------------------------
# Step 1: Load source entity files
# ---------------------------------------------------------------------------

def load_source_file(subdir: str, filename: str, expected_prefix: str) -> pd.DataFrame:
    """Load a single source TSV, validate schema and ID prefixes."""
    path = os.path.join(DATASET_DIR, subdir, filename)
    print(f"  Loading {filename} ...", end=" ", flush=True)

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    print(f"{len(df):,} rows")

    # Assert exact columns
    actual_cols = set(df.columns)
    assert actual_cols == EXPECTED_SOURCE_COLS, (
        f"{filename}: expected columns {EXPECTED_SOURCE_COLS}, got {actual_cols}"
    )

    # Assert entity_id is never null/empty
    empty_ids = df["entity_id"].eq("").sum()
    assert empty_ids == 0, (
        f"{filename}: found {empty_ids} empty entity_id values"
    )

    # Assert all IDs have the expected prefix
    bad_prefix = ~df["entity_id"].str.startswith(expected_prefix)
    n_bad = bad_prefix.sum()
    assert n_bad == 0, (
        f"{filename}: {n_bad} IDs don't start with '{expected_prefix}'. "
        f"Examples: {df.loc[bad_prefix, 'entity_id'].head(5).tolist()}"
    )

    # Assert no duplicate entity_ids
    n_dupes = df["entity_id"].duplicated().sum()
    assert n_dupes == 0, (
        f"{filename}: found {n_dupes} duplicate entity_ids"
    )

    return df


# ---------------------------------------------------------------------------
# Step 2: Load and parse ground truth
# ---------------------------------------------------------------------------

def parse_matched_ids(value: str) -> set:
    """Parse a comma-separated matched_entity_ids string into a set.

    '' (empty string) → set()   ... this is a singleton
    'S2-123,S3-456'  → {'S2-123', 'S3-456'}
    """
    if not value or value.strip() == "":
        return set()
    return set(id_str.strip() for id_str in value.split(",") if id_str.strip())


def load_ground_truth() -> tuple[pd.DataFrame, dict]:
    """Load ground truth TSV and parse matched_entity_ids into a dict of sets.

    Returns:
        (gt_df, gt_matches) where gt_matches = { 'S1-XXX': set('S2-YYY', ...) }
    """
    print(f"  Loading train_ground_truth.tsv ...", end=" ", flush=True)

    gt_df = pd.read_csv(GROUND_TRUTH_FILE, sep="\t", dtype=str, keep_default_na=False)
    print(f"{len(gt_df):,} rows")

    # Assert exact columns
    actual_cols = set(gt_df.columns)
    assert actual_cols == EXPECTED_GT_COLS, (
        f"Ground truth: expected columns {EXPECTED_GT_COLS}, got {actual_cols}"
    )

    # Assert source1_entity_id is never null/empty
    empty_s1 = gt_df["source1_entity_id"].eq("").sum()
    assert empty_s1 == 0, (
        f"Ground truth: found {empty_s1} empty source1_entity_id values"
    )

    # Assert no duplicate source1_entity_ids
    n_dupes = gt_df["source1_entity_id"].duplicated().sum()
    assert n_dupes == 0, (
        f"Ground truth: found {n_dupes} duplicate source1_entity_ids"
    )

    # Parse matched_entity_ids into sets — vectorized for speed
    print("  Parsing matched_entity_ids ...", end=" ", flush=True)
    parsed = gt_df["matched_entity_ids"].apply(parse_matched_ids)
    gt_matches = dict(zip(gt_df["source1_entity_id"], parsed))
    print("done")

    return gt_df, gt_matches


# ---------------------------------------------------------------------------
# Step 3: Cross-reference integrity assertions
# ---------------------------------------------------------------------------

def validate_integrity(id_sets: dict, gt_matches: dict):
    """Run cross-reference assertions between ground truth and source files.

    Args:
        id_sets: { 'train_s1': set, 'train_s2': set, 'train_s3': set, ... }
        gt_matches: { 'S1-XXX': set('S2-YYY', 'S3-ZZZ', ...) }
    """
    train_s1_ids = id_sets["train_s1"]
    train_s2_ids = id_sets["train_s2"]
    train_s3_ids = id_sets["train_s3"]

    print("\n--- Integrity Checks ---")

    # 1. Every S1 in ground truth must exist in train_source1
    gt_s1_ids = set(gt_matches.keys())
    missing_s1 = gt_s1_ids - train_s1_ids
    assert len(missing_s1) == 0, (
        f"Ground truth references {len(missing_s1)} S1 IDs not in train_source1.tsv. "
        f"Examples: {list(missing_s1)[:5]}"
    )
    print(f"  [OK] All {len(gt_s1_ids):,} ground truth S1 IDs exist in train_source1")

    # 2 & 3. Every match ID must be S2-/S3- and exist in the corresponding source
    all_match_ids = set()
    for match_set in gt_matches.values():
        all_match_ids.update(match_set)

    # Check prefix
    bad_prefix_ids = {mid for mid in all_match_ids
                      if not (mid.startswith("S2-") or mid.startswith("S3-"))}
    assert len(bad_prefix_ids) == 0, (
        f"Ground truth contains {len(bad_prefix_ids)} match IDs that are not S2-/S3-. "
        f"Examples: {list(bad_prefix_ids)[:5]}"
    )
    print(f"  [OK] All {len(all_match_ids):,} match IDs have S2- or S3- prefix")

    # Check existence in source files
    s2_match_ids = {mid for mid in all_match_ids if mid.startswith("S2-")}
    s3_match_ids = {mid for mid in all_match_ids if mid.startswith("S3-")}

    missing_s2 = s2_match_ids - train_s2_ids
    assert len(missing_s2) == 0, (
        f"{len(missing_s2)} S2 match IDs not found in train_source2.tsv. "
        f"Examples: {list(missing_s2)[:5]}"
    )
    print(f"  [OK] All {len(s2_match_ids):,} S2 match IDs exist in train_source2")

    missing_s3 = s3_match_ids - train_s3_ids
    assert len(missing_s3) == 0, (
        f"{len(missing_s3)} S3 match IDs not found in train_source3.tsv. "
        f"Examples: {list(missing_s3)[:5]}"
    )
    print(f"  [OK] All {len(s3_match_ids):,} S3 match IDs exist in train_source3")

    # 4. Check if every S1 in train_source1 appears in ground truth (warning only)
    orphan_s1 = train_s1_ids - gt_s1_ids
    if orphan_s1:
        print(f"  [WARN] {len(orphan_s1):,} S1 IDs in train_source1 have no ground truth entry")
    else:
        print(f"  [OK] Every S1 in train_source1 has a ground truth entry")


# ---------------------------------------------------------------------------
# Main: load everything and validate
# ---------------------------------------------------------------------------

def load_all_data() -> dict:
    """Load all data, validate, and return a dict of DataFrames + parsed matches.

    Returns:
        {
            'train_s1': DataFrame, 'train_s2': DataFrame, 'train_s3': DataFrame,
            'test_s1': DataFrame,  'test_s2': DataFrame,  'test_s3': DataFrame,
            'ground_truth': DataFrame,
            'gt_matches': { 'S1-XXX': set('S2-YYY', ...) }
        }
    """
    print("=" * 50)
    print("  Data Loading & Validation")
    print("=" * 50)

    # Load source files
    print("\n--- Source Files ---")
    data = {}
    id_sets = {}

    for key, (subdir, filename, prefix) in SOURCE_FILES.items():
        df = load_source_file(subdir, filename, prefix)
        data[key] = df
        id_sets[key] = set(df["entity_id"])

    # Load and parse ground truth
    print("\n--- Ground Truth ---")
    gt_df, gt_matches = load_ground_truth()
    data["ground_truth"] = gt_df
    data["gt_matches"] = gt_matches

    # Cross-reference integrity checks
    validate_integrity(id_sets, gt_matches)

    # Summary
    n_singletons = sum(1 for v in gt_matches.values() if len(v) == 0)
    n_with_matches = sum(1 for v in gt_matches.values() if len(v) > 0)

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    print(f"  Train Source 1:  {len(data['train_s1']):>10,} rows")
    print(f"  Train Source 2:  {len(data['train_s2']):>10,} rows")
    print(f"  Train Source 3:  {len(data['train_s3']):>10,} rows")
    print(f"  Test Source 1:   {len(data['test_s1']):>10,} rows")
    print(f"  Test Source 2:   {len(data['test_s2']):>10,} rows")
    print(f"  Test Source 3:   {len(data['test_s3']):>10,} rows")
    print(f"  Ground Truth:    {len(gt_df):>10,} rows")
    print(f"    - With matches:  {n_with_matches:>8,}")
    print(f"    - Singletons:    {n_singletons:>8,}")
    print(f"\n  All assertions PASSED [OK]")
    print("=" * 50)

    return data


if __name__ == "__main__":
    load_all_data()
