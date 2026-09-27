"""
Phase 0, Step 5: Grouped Macro F0.5 Scorer
============================================
Precision-heavy metric (beta=0.5). Pure Python set ops — no dependencies.

Competition formula:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Usage:
    # Run unit tests:
    python src/scorer.py --test

    # As a module:
    from src.scorer import macro_f05, f05_single
"""

import sys
import time


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

BETA = 0.5
BETA_SQ = BETA * BETA  # 0.25


def f05_single(truth: set, preds: set) -> float:
    """F0.5 score for one S1 entity.

    Args:
        truth: set of correct match IDs (empty = singleton)
        preds: set of predicted match IDs (empty = predict singleton)

    Returns:
        F0.5 score between 0.0 and 1.0
    """
    # Singleton handling
    if len(truth) == 0:
        return 1.0 if len(preds) == 0 else 0.0

    # Non-singleton
    if len(preds) == 0:
        return 0.0

    tp = len(truth & preds)
    fp = len(preds - truth)
    fn = len(truth - preds)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0

    # F0.5 = (1 + 0.25) * P * R / (0.25 * P + R)
    return (1.0 + BETA_SQ) * precision * recall / (BETA_SQ * precision + recall)


def macro_f05(gt_matches: dict, predictions: dict, verbose: bool = False) -> float:
    """Macro-averaged F0.5 across all S1 entities.

    Args:
        gt_matches:  { "S1-XXX": set("S2-YYY", ...) }  — ground truth
        predictions: { "S1-XXX": set("S2-YYY", ...) }  — model predictions
        verbose:     if True, print detailed breakdown

    Returns:
        Macro F0.5 score
    """
    t0 = time.time()

    total = 0
    score_sum = 0.0

    # Breakdown counters
    n_singleton = 0
    n_nonsingle = 0
    singleton_score_sum = 0.0
    nonsingle_score_sum = 0.0
    n_perfect = 0
    n_zero = 0

    for s1_id, truth in gt_matches.items():
        preds = predictions.get(s1_id, set())
        score = f05_single(truth, preds)

        score_sum += score
        total += 1

        if len(truth) == 0:
            n_singleton += 1
            singleton_score_sum += score
        else:
            n_nonsingle += 1
            nonsingle_score_sum += score

        if score == 1.0:
            n_perfect += 1
        elif score == 0.0:
            n_zero += 1

    macro = score_sum / total if total > 0 else 0.0
    elapsed = time.time() - t0

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Macro F0.5 Scoring Results")
        print(f"{'='*50}")
        print(f"  Total S1 entities:      {total:,}")
        print(f"  Singletons:             {n_singleton:,}")
        print(f"  Non-singletons:         {n_nonsingle:,}")
        print(f"  ---")
        avg_sing = singleton_score_sum / n_singleton if n_singleton > 0 else 0.0
        avg_nsing = nonsingle_score_sum / n_nonsingle if n_nonsingle > 0 else 0.0
        print(f"  Avg singleton score:    {avg_sing:.4f}")
        print(f"  Avg non-singleton score:{avg_nsing:.4f}")
        print(f"  Perfect scores (1.0):   {n_perfect:,}")
        print(f"  Zero scores (0.0):      {n_zero:,}")
        print(f"  ---")
        print(f"  MACRO F0.5:             {macro:.6f}")
        print(f"  Scoring time:           {elapsed:.2f}s")
        print(f"{'='*50}")

    return macro


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_tests():
    """Run 5 unit tests matching the worked examples."""
    passed = 0
    failed = 0

    def check(name, truth, preds, expected, tol=1e-6):
        nonlocal passed, failed
        got = f05_single(truth, preds)
        ok = abs(got - expected) < tol
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}: expected={expected:.4f}, got={got:.4f}")

    print("\n--- Unit Tests ---\n")

    # Test 1: Correct singleton
    check(
        "Correct singleton",
        truth=set(),
        preds=set(),
        expected=1.0,
    )

    # Test 2: False match on singleton
    check(
        "False match on singleton",
        truth=set(),
        preds={"S2-100"},
        expected=0.0,
    )

    # Test 3: Perfect multi-match
    check(
        "Perfect multi-match",
        truth={"S2-100", "S3-200"},
        preds={"S2-100", "S3-200"},
        expected=1.0,
    )

    # Test 4: One correct + one extra (precision penalty)
    # tp=2, fp=1, fn=0 → P=2/3, R=1.0 → F0.5 = 1.25*0.6667*1.0/(0.25*0.6667+1.0)
    expected_4 = 1.25 * (2/3) * 1.0 / (0.25 * (2/3) + 1.0)
    check(
        "One correct + one extra",
        truth={"S2-100", "S3-200"},
        preds={"S2-100", "S3-200", "S3-999"},
        expected=expected_4,
    )

    # Test 5: Missed all matches
    check(
        "Missed all matches",
        truth={"S2-100", "S3-200"},
        preds=set(),
        expected=0.0,
    )

    # Test 6: Partial match (1 of 2 correct, no extras)
    # tp=1, fp=0, fn=1 → P=1.0, R=0.5 → F0.5 = 1.25*1.0*0.5/(0.25*1.0+0.5)
    expected_6 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5)
    check(
        "Partial match (1 of 2, no extras)",
        truth={"S2-100", "S3-200"},
        preds={"S2-100"},
        expected=expected_6,
    )

    # Test 7: Completely wrong predictions
    check(
        "All wrong predictions",
        truth={"S2-100"},
        preds={"S2-999", "S3-888"},
        expected=0.0,
    )

    # Test 8: macro_f05 end-to-end
    print("\n  --- Macro F0.5 integration test ---\n")
    gt = {
        "S1-001": {"S2-100", "S3-200"},   # non-singleton
        "S1-002": set(),                    # singleton
        "S1-003": {"S2-300"},              # non-singleton
    }
    preds_dict = {
        "S1-001": {"S2-100", "S3-200"},   # perfect → 1.0
        "S1-002": set(),                    # correct singleton → 1.0
        "S1-003": set(),                    # missed → 0.0
    }
    expected_macro = (1.0 + 1.0 + 0.0) / 3.0
    got_macro = macro_f05(gt, preds_dict)
    ok = abs(got_macro - expected_macro) < 1e-6
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] Macro F0.5 integration: expected={expected_macro:.4f}, got={got_macro:.4f}")

    print(f"\n--- Results: {passed} passed, {failed} failed ---")
    return failed == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  python src/scorer.py --test    Run unit tests")
        print("")
        print("As a module:")
        print("  from src.scorer import macro_f05, f05_single")
