"""
Phase 1.5 — Output Generation
================================
Writes the two required TSV submission files:
  1. output/matching_results.tsv
  2. output/candidate_pairs.tsv

Guarantees per the spec:
  - Every test S1 ID appears exactly once in both files.
  - Empty match/candidate lists are truly empty cells (not [], NaN, None).
  - All listed IDs exist in test S2+S3, have no duplicates, are not S1 IDs.
  - Every predicted match is also in the candidate set.
  - Headers are exactly: source1_entity_id / matched_entity_ids
                         source1_entity_id / candidate_entity_ids

Usage:
    from src.output_writer import write_outputs
    write_outputs(predictions, candidates, test_s1_ids, output_dir)
"""

import os
from typing import Dict, List, Tuple, Set


def write_outputs(
    predictions: Dict[str, Set[str]],
    candidates:  Dict[str, List[Tuple[str, float]]],
    test_s1_ids: list,
    output_dir:  str = 'output',
    validate:    bool = True,
) -> None:
    """
    Write matching_results.tsv and candidate_pairs.tsv.

    Parameters
    ----------
    predictions  : {s1_id: set(matched_tgt_ids)}      — model output
    candidates   : {s1_id: [(tgt_id, prescore), ...]} — blocked candidates
    test_s1_ids  : ordered list of all test S1 entity IDs
    output_dir   : directory to write files into
    validate     : run internal consistency checks before writing
    """
    os.makedirs(output_dir, exist_ok=True)

    matching_path   = os.path.join(output_dir, 'matching_results.tsv')
    candidates_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    if validate:
        _validate(predictions, candidates, test_s1_ids)

    # --- Write matching_results.tsv ---
    with open(matching_path, 'w', encoding='utf-8', newline='') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in test_s1_ids:
            match_set = predictions.get(s1_id, set())
            # Sort for determinism; comma-separated, no trailing comma
            match_str = ','.join(sorted(match_set)) if match_set else ''
            f.write(f'{s1_id}\t{match_str}\n')

    # --- Write candidate_pairs.tsv ---
    with open(candidates_path, 'w', encoding='utf-8', newline='') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in test_s1_ids:
            cand_list = candidates.get(s1_id, [])
            cand_ids  = [cid for cid, _ in cand_list]
            cand_str  = ','.join(cand_ids) if cand_ids else ''
            f.write(f'{s1_id}\t{cand_str}\n')

    print(f"\n  [OK] Written: {matching_path}")
    print(f"  [OK] Written: {candidates_path}")

    # Summary
    n_matched   = sum(1 for s1_id in test_s1_ids if predictions.get(s1_id))
    n_singleton = sum(1 for s1_id in test_s1_ids if not predictions.get(s1_id))
    total_preds = sum(len(v) for v in predictions.values())
    total_cands = sum(len(v) for v in candidates.values())
    print(f"  S1 with ≥1 predicted match: {n_matched:,}")
    print(f"  S1 predicted as singleton:  {n_singleton:,}")
    print(f"  Total predicted matches:    {total_preds:,}")
    print(f"  Total candidate pairs:      {total_cands:,}")


def _validate(
    predictions: Dict[str, Set[str]],
    candidates:  Dict[str, List[Tuple[str, float]]],
    test_s1_ids: list,
) -> None:
    """Run pre-write sanity checks."""
    errors = []

    # 1. All test S1 IDs covered
    covered = set(predictions.keys())
    for s1_id in test_s1_ids:
        if s1_id not in covered:
            errors.append(f"MISSING from predictions: {s1_id}")

    # 2. No predicted ID is an S1 ID
    for s1_id, match_set in predictions.items():
        s1_bad = {m for m in match_set if m.startswith('S1-')}
        if s1_bad:
            errors.append(f"{s1_id}: predicted S1 IDs {list(s1_bad)[:3]}")

    # 3. No duplicate predictions per S1
    for s1_id, match_set in predictions.items():
        if len(match_set) != len(set(match_set)):
            errors.append(f"{s1_id}: duplicate predicted IDs")

    # 4. Every predicted match is in candidates
    for s1_id, match_set in predictions.items():
        cand_set = set(c[0] for c in candidates.get(s1_id, []))
        missing = match_set - cand_set
        if missing:
            errors.append(
                f"{s1_id}: predicted IDs not in candidate set: {list(missing)[:3]}"
            )

    if errors:
        print("\n  [VALIDATION ERRORS]")
        for e in errors[:20]:
            print(f"    {e}")
        raise ValueError(f"Output validation failed with {len(errors)} errors")

    print("  [OK] Output validation passed")
