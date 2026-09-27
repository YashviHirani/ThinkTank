"""
Phase 1, Step 5.3: Pairwise Feature Extraction Module
======================================================
Computes 14 tabular features for candidate pairs (S1, Target) using rapidfuzz,
token Jaccard, address similarity, numeric agreement, and country equality.

Feature List:
  0. name_ratio: Levenshtein ratio between core names
  1. name_token_sort: Word-order invariant name similarity
  2. name_token_set: Extra-token invariant name similarity
  3. name_jaro_winkler: Prefix-biased name similarity
  4. name_jaccard: Token Jaccard overlap between name words
  5. core_name_exact: 1.0 if core names match exactly, else 0.0
  6. suffix_agreement: 1.0 (same suffix), -1.0 (conflict), 0.0 (missing)
  7. addr_jaccard: Token Jaccard overlap between addresses
  8. addr_ratio: Levenshtein similarity on address text
  9. common_words_count: Count of shared address words (len >= 3)
 10. house_num_match: 1.0 if building/plot numbers share at least 1 number
 11. house_num_conflict: 1.0 if both have numbers but none match, else 0.0
 12. postcode_match: 1.0 if postcodes match, else 0.0
 13. country_equal: 1.0 if countries match, else 0.0

Usage:
    python src/features.py --test
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rapidfuzz import fuzz, distance

FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_jaccard",
    "core_name_exact",
    "suffix_agreement",
    "addr_jaccard",
    "addr_ratio",
    "common_words_count",
    "house_num_match",
    "house_num_conflict",
    "postcode_match",
    "country_equal",
]


def compute_pair_features(
    s1_rec: Dict[str, Any],
    tgt_rec: Dict[str, Any],
) -> List[float]:
    """
    Computes a vector of 14 features for one candidate pair.
    Expected dict keys for both records:
      - 'core_name': str
      - 'legal_suffix': str
      - 'clean_address': str
      - 'numeric_tokens': list of str
      - 'postcode': str
      - 'country': str
    """
    cn1 = s1_rec.get("core_name", "") or ""
    cn2 = tgt_rec.get("core_name", "") or ""

    addr1 = s1_rec.get("clean_address", "") or ""
    addr2 = tgt_rec.get("clean_address", "") or ""

    suf1 = s1_rec.get("legal_suffix", "") or ""
    suf2 = tgt_rec.get("legal_suffix", "") or ""

    nums1 = s1_rec.get("numeric_tokens", []) or []
    nums2 = tgt_rec.get("numeric_tokens", []) or []

    pc1 = s1_rec.get("postcode", "") or ""
    pc2 = tgt_rec.get("postcode", "") or ""

    c1 = s1_rec.get("country", "") or ""
    c2 = tgt_rec.get("country", "") or ""

    # 1. Name features
    if cn1 and cn2:
        name_ratio = fuzz.ratio(cn1, cn2) / 100.0
        name_token_sort = fuzz.token_sort_ratio(cn1, cn2) / 100.0
        name_token_set = fuzz.token_set_ratio(cn1, cn2) / 100.0
        name_jw = distance.JaroWinkler.similarity(cn1, cn2)
        
        tokens1 = set(w for w in cn1.split() if len(w) >= 2)
        tokens2 = set(w for w in cn2.split() if len(w) >= 2)
        u_name = len(tokens1 | tokens2)
        name_jaccard = (len(tokens1 & tokens2) / u_name) if u_name > 0 else 0.0
        core_exact = 1.0 if cn1 == cn2 else 0.0
    else:
        name_ratio = 0.0
        name_token_sort = 0.0
        name_token_set = 0.0
        name_jw = 0.0
        name_jaccard = 0.0
        core_exact = 0.0

    # Suffix agreement
    if suf1 and suf2:
        suffix_agree = 1.0 if suf1 == suf2 else -1.0
    else:
        suffix_agree = 0.0

    # 2. Address features
    if addr1 and addr2:
        addr_ratio = fuzz.ratio(addr1, addr2) / 100.0
        a_tok1 = set(w for w in addr1.split() if len(w) >= 3 and not w.isdigit())
        a_tok2 = set(w for w in addr2.split() if len(w) >= 3 and not w.isdigit())
        u_addr = len(a_tok1 | a_tok2)
        addr_jaccard = (len(a_tok1 & a_tok2) / u_addr) if u_addr > 0 else 0.0
        common_words = float(len(a_tok1 & a_tok2))
    else:
        addr_ratio = 0.0
        addr_jaccard = 0.0
        common_words = 0.0

    # 3. Numeric agreement & conflicts
    s_num1 = set(nums1)
    s_num2 = set(nums2)
    if s_num1 and s_num2:
        has_match = bool(s_num1 & s_num2)
        house_match = 1.0 if has_match else 0.0
        house_conflict = 0.0 if has_match else 1.0
    else:
        house_match = 0.0
        house_conflict = 0.0

    # Postcode match
    if pc1 and pc2:
        pc_match = 1.0 if pc1 == pc2 else 0.0
    else:
        pc_match = 0.0

    # Country equality
    country_eq = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    return [
        name_ratio,
        name_token_sort,
        name_token_set,
        name_jw,
        name_jaccard,
        core_exact,
        suffix_agree,
        addr_jaccard,
        addr_ratio,
        common_words,
        house_match,
        house_conflict,
        pc_match,
        country_eq,
    ]


def compute_features_for_pairs(
    pair_list: List[Tuple[str, str]],
    s1_dict: Dict[str, Dict[str, Any]],
    target_dict: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """
    Computes feature matrix for a list of (s1_id, target_id) pairs.

    Returns:
        np.ndarray of shape (len(pair_list), len(FEATURE_NAMES))
    """
    n_pairs = len(pair_list)
    n_features = len(FEATURE_NAMES)
    matrix = np.zeros((n_pairs, n_features), dtype=np.float32)

    for i, (s1_id, tgt_id) in enumerate(pair_list):
        s1_rec = s1_dict.get(s1_id, {})
        tgt_rec = target_dict.get(tgt_id, {})
        matrix[i] = compute_pair_features(s1_rec, tgt_rec)

    return matrix


# ---------------------------------------------------------------------------
# Unit Test Suite
# ---------------------------------------------------------------------------

def run_tests() -> bool:
    """Verifies feature extraction outputs on sample test pairs."""
    print("\n--- Running Feature Extraction Unit Tests ---\n")
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

    # Pair 1: Perfect Match
    rec1 = {
        "core_name": "orelee barbershop",
        "legal_suffix": "llc",
        "clean_address": "1795 westchester drive high point nc",
        "numeric_tokens": ["1795"],
        "postcode": "27262",
        "country": "US",
    }
    rec2 = {
        "core_name": "orelee barbershop",
        "legal_suffix": "llc",
        "clean_address": "1795 westchester drive high point nc",
        "numeric_tokens": ["1795"],
        "postcode": "27262",
        "country": "US",
    }
    feats1 = compute_pair_features(rec1, rec2)
    assert_test("Length is 14", len(feats1) == 14)
    assert_test("Exact name ratio is 1.0", feats1[0] == 1.0)
    assert_test("Exact core is 1.0", feats1[5] == 1.0)
    assert_test("Suffix agreement is 1.0", feats1[6] == 1.0)
    assert_test("House number match is 1.0", feats1[10] == 1.0)
    assert_test("House number conflict is 0.0", feats1[11] == 0.0)
    assert_test("Postcode match is 1.0", feats1[12] == 1.0)

    # Pair 2: Different building numbers (conflict)
    rec3 = {
        "core_name": "orelee barbershop",
        "legal_suffix": "inc",
        "clean_address": "2000 main street dallas tx",
        "numeric_tokens": ["2000"],
        "postcode": "75001",
        "country": "US",
    }
    feats2 = compute_pair_features(rec1, rec3)
    assert_test("Suffix conflict is -1.0", feats2[6] == -1.0)
    assert_test("House number match is 0.0", feats2[10] == 0.0)
    assert_test("House number conflict is 1.0", feats2[11] == 1.0)
    assert_test("Postcode match is 0.0", feats2[12] == 0.0)

    # Pair 3: Word order swap (Starbucks Coffee vs Coffee Starbucks)
    rec4 = {"core_name": "starbucks coffee", "clean_address": "123 main st", "country": "US"}
    rec5 = {"core_name": "coffee starbucks", "clean_address": "123 main st", "country": "US"}
    feats3 = compute_pair_features(rec4, rec5)
    assert_test("Token sort handles word order", feats3[1] == 1.0)

    print(f"\n--- Results: {passed} passed, {failed} failed ---")
    return failed == 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  python src/features.py --test   Run feature extraction unit tests")
