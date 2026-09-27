"""
Submission Validator for Amazon ML Challenge 2026
=================================================
Validates the structural integrity and competition compliance of:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv

Usage:
    python utils/validate_submission.py \\
      --matching output/matching_results.tsv \\
      --candidate output/candidate_pairs.tsv \\
      --test-dir dataset/test
"""

import os
import sys
import argparse
import pandas as pd


FORBIDDEN_EMPTY_STRINGS = {"[]", "none", "null", "nan", "undefined", "{}"}


def parse_id_list(cell_value: str) -> list:
    """Parses a comma-separated ID string into a clean list of IDs."""
    if pd.isna(cell_value):
        return []
    s = str(cell_value).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def validate_submission(
    matching_path: str,
    candidate_path: str,
    test_dir: str,
    verbose: bool = True,
) -> bool:
    """
    Validates submission files against all competition requirements.
    Prints PASS and returns True if valid; prints errors and returns False otherwise.
    """
    def log(msg):
        if verbose:
            print(msg)

    log("\n" + "=" * 65)
    log("  Amazon ML Challenge 2026 — Submission Validation Gate")
    log("=" * 65)

    # 1. Check file existence
    if not os.path.exists(matching_path):
        log(f"  [FAIL] Missing matching file: {matching_path}")
        return False
    if not os.path.exists(candidate_path):
        log(f"  [FAIL] Missing candidate file: {candidate_path}")
        return False

    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    s2_test_path = os.path.join(test_dir, "test_source2.tsv")
    s3_test_path = os.path.join(test_dir, "test_source3.tsv")

    for p in [s1_test_path, s2_test_path, s3_test_path]:
        if not os.path.exists(p):
            log(f"  [FAIL] Missing reference test file: {p}")
            return False

    # 2. Load ground-truth test S1 IDs
    log("  Loading test source1 entities...")
    test_s1_df = pd.read_csv(s1_test_path, sep="\t", dtype=str, usecols=["entity_id"])
    expected_s1_ids = set(test_s1_df["entity_id"])
    n_expected = len(expected_s1_ids)
    log(f"    Expected test S1 entities: {n_expected:,}")

    # 3. Validate matching_results.tsv schema
    log("\n  Validating matching_results.tsv...")
    match_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    expected_match_cols = ["source1_entity_id", "matched_entity_ids"]
    if list(match_df.columns) != expected_match_cols:
        log(f"  [FAIL] Invalid headers in {matching_path}. Got: {list(match_df.columns)}, expected: {expected_match_cols}")
        return False

    if len(match_df) != n_expected:
        log(f"  [FAIL] Row count mismatch in matching_results.tsv! Got: {len(match_df):,}, expected: {n_expected:,}")
        return False

    match_s1_ids = set(match_df["source1_entity_id"])
    if match_s1_ids != expected_s1_ids:
        missing = expected_s1_ids - match_s1_ids
        extra = match_s1_ids - expected_s1_ids
        log(f"  [FAIL] S1 IDs mismatch in matching_results.tsv! Missing: {len(missing):,}, Extra: {len(extra):,}")
        return False

    # 4. Validate candidate_pairs.tsv schema
    log("  Validating candidate_pairs.tsv...")
    cand_df = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    expected_cand_cols = ["source1_entity_id", "candidate_entity_ids"]
    if list(cand_df.columns) != expected_cand_cols:
        log(f"  [FAIL] Invalid headers in {candidate_path}. Got: {list(cand_df.columns)}, expected: {expected_cand_cols}")
        return False

    if len(cand_df) != n_expected:
        log(f"  [FAIL] Row count mismatch in candidate_pairs.tsv! Got: {len(cand_df):,}, expected: {n_expected:,}")
        return False

    cand_s1_ids = set(cand_df["source1_entity_id"])
    if cand_s1_ids != expected_s1_ids:
        missing = expected_s1_ids - cand_s1_ids
        extra = cand_s1_ids - expected_s1_ids
        log(f"  [FAIL] S1 IDs mismatch in candidate_pairs.tsv! Missing: {len(missing):,}, Extra: {len(extra):,}")
        return False

    # 5. Check cell formatting (no literal "[]", "NaN", "None")
    log("  Checking singleton cell formatting...")
    for name, df, col in [
        ("matching_results.tsv", match_df, "matched_entity_ids"),
        ("candidate_pairs.tsv", cand_df, "candidate_entity_ids"),
    ]:
        bad_entries = df[df[col].str.lower().isin(FORBIDDEN_EMPTY_STRINGS)]
        if len(bad_entries) > 0:
            log(f"  [FAIL] Found forbidden string representation for singletons in {name} (e.g. {bad_entries[col].iloc[0]!r})")
            return False

    # 6. Check duplicate IDs within single cells & prefix validation
    log("  Checking ID prefixes and within-cell uniqueness...")
    for name, df, col in [
        ("matching_results.tsv", match_df, "matched_entity_ids"),
        ("candidate_pairs.tsv", cand_df, "candidate_entity_ids"),
    ]:
        for idx, row in df.iterrows():
            val = row[col].strip()
            if not val:
                continue
            ids = [x.strip() for x in val.split(",") if x.strip()]
            if len(ids) != len(set(ids)):
                log(f"  [FAIL] Duplicate IDs inside {name} for {row['source1_entity_id']}: {val}")
                return False
            for target_id in ids:
                if not (target_id.startswith("S2-") or target_id.startswith("S3-")):
                    log(f"  [FAIL] Invalid ID prefix in {name}: {target_id} (must be S2- or S3-)")
                    return False

    # 7. Check candidate containment: every match must be in candidates
    log("  Checking candidate containment (all matches in candidate list)...")
    cand_dict = dict(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]))

    containment_errors = 0
    for _, row in match_df.iterrows():
        s1_id = row["source1_entity_id"]
        matches = parse_id_list(row["matched_entity_ids"])
        if not matches:
            continue
        cands = set(parse_id_list(cand_dict.get(s1_id, "")))
        uncontained = set(matches) - cands
        if uncontained:
            if containment_errors < 3:
                log(f"  [FAIL] Matches not in candidates for {s1_id}: {uncontained}")
            containment_errors += 1

    if containment_errors > 0:
        log(f"  [FAIL] Found {containment_errors:,} S1 entities where predicted matches are missing from candidate_pairs.tsv")
        return False

    # 8. Check target ID existence against test_source2 and test_source3
    log("  Checking target ID existence in test source2/3...")
    test_s2_df = pd.read_csv(s2_test_path, sep="\t", dtype=str, usecols=["entity_id"])
    test_s3_df = pd.read_csv(s3_test_path, sep="\t", dtype=str, usecols=["entity_id"])
    valid_target_ids = set(test_s2_df["entity_id"]) | set(test_s3_df["entity_id"])

    # Sample check of predicted matches
    all_predicted_matches = set()
    for val in match_df["matched_entity_ids"].values:
        if val:
            all_predicted_matches.update(parse_id_list(val))

    log(f"    Total distinct predicted matches: {len(all_predicted_matches):,}")
    invalid_targets = all_predicted_matches - valid_target_ids
    if invalid_targets:
        log(f"  [FAIL] {len(invalid_targets):,} predicted IDs do not exist in test_source2 or test_source3! (e.g. {list(invalid_targets)[:3]})")
        return False

    log("\n" + "=" * 65)
    log("  RESULT: PASS")
    log("  All format, schema, prefix, and containment checks PASSED [OK]")
    log("=" * 65 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate Amazon ML Challenge submission files")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")
    args = parser.parse_args()

    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
