"""
Phase 1, Step 5.2: Baseline Blocking and Candidate Capping Module
==================================================================
Builds country-partitioned multi-key inverted indexes across Source 2 and
Source 3 records, retrieves candidates for each Source 1 entity, pre-ranks them
using fast token & number Jaccard heuristics, and caps candidate sets to Top-K (K=20).

Key Capabilities:
  - Country-partitioned inverted indexing (US, India, France, etc.).
  - 5 high-precision blocking keys (exact core name, sorted tokens, token+number,
    token+postcode, 2-token co-occurrence).
  - High-frequency bucket cap (MAX_BUCKET_SIZE=500) to prevent memory bloat.
  - Fast integer-indexed candidate pre-ranking and Top-K capping.
  - Blocking recall and compactness evaluation against ground truth.

Usage:
    # Run test suite:
    python src/blocker.py --test
"""

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Import normalizer functions
from src.normalizer import (
    normalize_general_text,
    extract_core_name,
    get_sorted_tokens,
    normalize_address,
    extract_numeric_tokens,
    extract_postcode_candidate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 20
MAX_BUCKET_SIZE = 500  # Discard or skip buckets larger than this to stop generic floods


# ---------------------------------------------------------------------------
# Key Generation Functions
# ---------------------------------------------------------------------------

def extract_blocking_keys(
    core_name: str,
    sorted_name: str,
    numeric_tokens: List[str],
    postcode: str,
    country: str,
    clean_address: str = "",
) -> List[Tuple[str, str]]:
    """
    Extracts multi-strategy blocking keys for an entity.
    Each key is a tuple: (country, key_string).
    """
    keys = []
    if not country:
        country = "UNKNOWN"

    # 1. Exact core name (if at least 3 chars)
    if len(core_name) >= 3:
        keys.append((country, "cn:" + core_name))

    # 2. Sorted core name (always add if >= 3 chars for word-order invariance)
    if len(sorted_name) >= 3:
        keys.append((country, "sn:" + sorted_name))

    # Extract distinct name tokens (length >= 3 to skip trivial noise)
    name_tokens = [w for w in core_name.split() if len(w) >= 3 and w not in {"the", "and"}]

    if name_tokens:
        t0 = name_tokens[0]

        # 3. First token + Postcode
        if postcode and len(postcode) >= 4:
            keys.append((country, f"np:{t0}_{postcode}"))

        # 4. First token + House / Building numbers (up to 2 numbers)
        for num in numeric_tokens[:2]:
            if len(num) >= 1:
                keys.append((country, f"nh:{t0}_{num}"))

        # 5. First two name tokens (sorted pair: word-order invariant)
        if len(name_tokens) >= 2:
            w1, w2 = sorted([name_tokens[0], name_tokens[1]])
            keys.append((country, f"2t:{w1}_{w2}"))

        # 6. First and third name token (handles inserted middle words)
        if len(name_tokens) >= 3:
            w1, w3 = sorted([name_tokens[0], name_tokens[2]])
            keys.append((country, f"2t:{w1}_{w3}"))

    # 7. Address street token + house/building number (resilience to DBA / trade names)
    if clean_address and numeric_tokens:
        skip_words = {
            "street", "road", "drive", "avenue", "lane", "boulevard", "court",
            "pobox", "unit", "apt", "suite", "floor", "near", "opp", "opposite"
        }
        addr_words = [
            w for w in clean_address.split()
            if len(w) >= 4 and not w.isdigit() and w not in skip_words
        ]
        for s_word in addr_words[:2]:
            for num in numeric_tokens[:2]:
                keys.append((country, f"an:{s_word}_{num}"))

    return keys


# ---------------------------------------------------------------------------
# Inverted Index Class
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Memory-efficient integer-indexed inverted index over target records (S2 + S3).
    """

    def __init__(self, max_bucket_size: int = MAX_BUCKET_SIZE):
        self.max_bucket_size = max_bucket_size
        self.index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        
        # Dense target record storage (parallel arrays indexed by integer pos)
        self.target_ids: List[str] = []
        self.target_name_tokens: List[FrozenSet[str]] = []
        self.target_addr_tokens: List[FrozenSet[str]] = []
        self.target_nums: List[FrozenSet[str]] = []
        self.target_countries: List[str] = []

    def build_from_dataframe(self, df: pd.DataFrame):
        """
        Builds the inverted index from a DataFrame of S2 and/or S3 records.
        Expected columns: entity_id, business_name, business_address, country
        (If pre-normalized columns exist, they will be used directly).
        """
        t0 = time.time()
        n_rows = len(df)
        print(f"  Indexing {n_rows:,} target records...", end=" ", flush=True)

        has_clean = "core_name" in df.columns

        # Pre-allocate lists
        self.target_ids = df["entity_id"].tolist()
        self.target_name_tokens = [None] * n_rows
        self.target_addr_tokens = [None] * n_rows
        self.target_nums = [None] * n_rows
        self.target_countries = [None] * n_rows

        # Iterate rows
        raw_names = df["business_name"].fillna("").values
        raw_addrs = df["business_address"].fillna("").values
        raw_countries = df["country"].fillna("").str.strip().str.upper().values

        if has_clean:
            core_names = df["core_name"].fillna("").values
            sorted_names = df["sorted_name"].fillna("").values
            num_tokens_list = df["numeric_tokens"].values
            postcodes = df["postcode"].fillna("").values
        else:
            core_names = None

        for i in range(n_rows):
            country = raw_countries[i]
            self.target_countries[i] = country

            if has_clean:
                cn = core_names[i]
                sn = sorted_names[i]
                nums = num_tokens_list[i] if isinstance(num_tokens_list[i], list) else []
                pc = postcodes[i]
                addr = df["clean_address"].iloc[i] if "clean_address" in df.columns else normalize_address(raw_addrs[i])
            else:
                clean_name = normalize_general_text(raw_names[i])
                cn, _ = extract_core_name(clean_name)
                sn = get_sorted_tokens(cn)
                addr = normalize_address(raw_addrs[i])
                nums = extract_numeric_tokens(addr)
                pc = extract_postcode_candidate(addr)

            # Store token sets for fast Jaccard pre-scoring
            n_tokens = frozenset(w for w in cn.split() if len(w) >= 2)
            a_tokens = frozenset(w for w in addr.split() if len(w) >= 2)
            num_set = frozenset(nums)

            self.target_name_tokens[i] = n_tokens
            self.target_addr_tokens[i] = a_tokens
            self.target_nums[i] = num_set

            # Extract and store blocking keys
            keys = extract_blocking_keys(cn, sn, nums, pc, country, addr)
            for k in keys:
                self.index[k].append(i)

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s). Total unique keys: {len(self.index):,}")

    def get_candidate_indices_for_keys(self, keys: List[Tuple[str, str]]) -> Set[int]:
        """
        Retrieves union of target record indices matching any of the keys,
        filtering out buckets that exceed max_bucket_size.
        """
        cands = set()
        for k in keys:
            bucket = self.index.get(k)
            if bucket and len(bucket) <= self.max_bucket_size:
                cands.update(bucket)
        return cands


# ---------------------------------------------------------------------------
# Candidate Retrieval & Pre-Ranking
# ---------------------------------------------------------------------------

def pre_score_pair(
    s1_name_tokens: FrozenSet[str],
    s1_addr_tokens: FrozenSet[str],
    s1_nums: FrozenSet[str],
    target_name_tokens: FrozenSet[str],
    target_addr_tokens: FrozenSet[str],
    target_nums: FrozenSet[str],
) -> float:
    """
    Computes a fast deterministic heuristic score between 0.0 and 1.0.
    """
    # Name Jaccard
    name_union = len(s1_name_tokens | target_name_tokens)
    name_jaccard = (len(s1_name_tokens & target_name_tokens) / name_union) if name_union > 0 else 0.0

    # Address Jaccard
    addr_union = len(s1_addr_tokens | target_addr_tokens)
    addr_jaccard = (len(s1_addr_tokens & target_addr_tokens) / addr_union) if addr_union > 0 else 0.0

    # Number match indicator
    has_shared_num = 1.0 if (s1_nums and target_nums and (s1_nums & target_nums)) else 0.0

    return 0.60 * name_jaccard + 0.25 * addr_jaccard + 0.15 * has_shared_num


def retrieve_top_k_candidates_for_s1(
    core_name: str,
    sorted_name: str,
    clean_address: str,
    numeric_tokens: List[str],
    postcode: str,
    country: str,
    index: InvertedIndex,
    top_k: int = DEFAULT_TOP_K,
) -> List[Tuple[str, float]]:
    """
    Retrieves and ranks the top K candidate target IDs for a single S1 entity.

    Returns:
        List of (target_entity_id, pre_score) sorted descending by score.
    """
    keys = extract_blocking_keys(core_name, sorted_name, numeric_tokens, postcode, country, clean_address)
    cand_indices = index.get_candidate_indices_for_keys(keys)

    if not cand_indices:
        return []

    s1_name_tokens = frozenset(w for w in core_name.split() if len(w) >= 2)
    s1_addr_tokens = frozenset(w for w in clean_address.split() if len(w) >= 2)
    s1_nums = frozenset(numeric_tokens)

    # Score each retrieved candidate
    scored = []
    for idx in cand_indices:
        score = pre_score_pair(
            s1_name_tokens,
            s1_addr_tokens,
            s1_nums,
            index.target_name_tokens[idx],
            index.target_addr_tokens[idx],
            index.target_nums[idx],
        )
        scored.append((index.target_ids[idx], score))

    # Sort descending by pre-score and take top-K
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    index: InvertedIndex,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 50000,
) -> Dict[str, List[str]]:
    """
    Generates top-K candidate target IDs for each S1 entity in s1_df.

    Returns:
        dict: { "S1-XXX": ["S2-YYY", "S3-ZZZ", ...], ... }
    """
    t0 = time.time()
    n_s1 = len(s1_df)
    print(f"  Generating candidates for {n_s1:,} S1 entities (Top-K={top_k})...")

    has_clean = "core_name" in s1_df.columns

    s1_ids = s1_df["entity_id"].tolist()
    raw_names = s1_df["business_name"].fillna("").values
    raw_addrs = s1_df["business_address"].fillna("").values
    countries = s1_df["country"].fillna("").str.strip().str.upper().values

    if has_clean:
        core_names = s1_df["core_name"].fillna("").values
        sorted_names = s1_df["sorted_name"].fillna("").values
        clean_addrs = s1_df["clean_address"].fillna("").values
        nums_list = s1_df["numeric_tokens"].values
        postcodes = s1_df["postcode"].fillna("").values
    else:
        core_names = None

    candidates_dict = {}

    for i in range(n_s1):
        if has_clean:
            cn = core_names[i]
            sn = sorted_names[i]
            addr = clean_addrs[i]
            nums = nums_list[i] if isinstance(nums_list[i], list) else []
            pc = postcodes[i]
        else:
            clean_name = normalize_general_text(raw_names[i])
            cn, _ = extract_core_name(clean_name)
            sn = get_sorted_tokens(cn)
            addr = normalize_address(raw_addrs[i])
            nums = extract_numeric_tokens(addr)
            pc = extract_postcode_candidate(addr)

        top_candidates = retrieve_top_k_candidates_for_s1(
            cn, sn, addr, nums, pc, countries[i], index, top_k=top_k
        )
        candidates_dict[s1_ids[i]] = [cand_id for cand_id, _ in top_candidates]

        if (i + 1) % batch_size == 0 or (i + 1) == n_s1:
            rate = (i + 1) / (time.time() - t0)
            print(f"    Processed {i + 1:,} / {n_s1:,} ({rate:,.0f} entities/s)")

    elapsed = time.time() - t0
    print(f"  Candidate generation completed in {elapsed:.1f}s.")
    return candidates_dict


# ---------------------------------------------------------------------------
# Evaluation: Blocking Recall & Compactness
# ---------------------------------------------------------------------------

def evaluate_blocking_recall(
    candidates_dict: Dict[str, List[str]],
    gt_matches: Dict[str, Set[str]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates blocking quality against ground truth:
      - Candidate Recall: % of true match pairs present in candidate lists
      - Average / Maximum candidate pool size
      - Singleton coverage
    """
    total_s1 = len(candidates_dict)
    n_singletons = 0
    n_nonsingletons = 0
    total_true_matches = 0
    retrieved_true_matches = 0

    cand_counts = []

    for s1_id, cands in candidates_dict.items():
        cand_set = set(cands)
        cand_counts.append(len(cands))
        truth = gt_matches.get(s1_id, set())

        if len(truth) == 0:
            n_singletons += 1
        else:
            n_nonsingletons += 1
            total_true_matches += len(truth)
            retrieved_true_matches += len(truth & cand_set)

    recall = (retrieved_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    avg_cands = float(np.mean(cand_counts)) if cand_counts else 0.0
    max_cands = int(np.max(cand_counts)) if cand_counts else 0

    results = {
        "total_s1": total_s1,
        "singletons": n_singletons,
        "nonsingletons": n_nonsingletons,
        "total_true_matches": total_true_matches,
        "retrieved_true_matches": retrieved_true_matches,
        "blocking_recall": recall,
        "avg_candidates_per_s1": avg_cands,
        "max_candidates_per_s1": max_cands,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("  Blocking Evaluation Results")
        print("=" * 60)
        print(f"  Total S1 entities evaluated:     {total_s1:>10,}")
        print(f"  True Singletons (0 matches):      {n_singletons:>10,} ({n_singletons/total_s1*100:.1f}%)")
        print(f"  True Non-Singletons (1+ matches): {n_nonsingletons:>10,} ({n_nonsingletons/total_s1*100:.1f}%)")
        print(f"  Total true match links:           {total_true_matches:>10,}")
        print(f"  True matches retrieved in Top-K:  {retrieved_true_matches:>10,}")
        print(f"  ------------------------------------------------------------")
        print(f"  BLOCKING RECALL:                  {recall*100:>9.2f}%")
        print(f"  Average candidates per S1:        {avg_cands:>10.2f}")
        print(f"  Maximum candidates per S1:        {max_cands:>10}")
        print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Unit Test Suite
# ---------------------------------------------------------------------------

def run_tests() -> bool:
    """Automated unit test suite verifying blocking retrieval and recall."""
    print("\n--- Running Blocker Unit Tests ---\n")
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

    # 1. Mock Target records (S2 + S3)
    target_data = [
        # Match for S1-001 (Orelee's Barbershop, High Point NC, US)
        {"entity_id": "S2-101", "business_name": "Orelee's Barbershop LLC", "business_address": "1795 Westchester Drive, High Point, NC", "country": "US"},
        {"entity_id": "S3-102", "business_name": "Orelee Barbershop", "business_address": "Westchester Dr, 1795, High Point", "country": "US"},
        
        # Match for S1-002 (Word order swapped: Coffee Starbucks, US)
        {"entity_id": "S2-201", "business_name": "Starbucks Coffee Inc", "business_address": "123 Main St, Seattle, WA", "country": "US"},
        
        # Match for S1-003 (French business with accents)
        {"entity_id": "S2-301", "business_name": "Café Étoile SARL", "business_address": "15 Rue de Paris, Lyon", "country": "France"},

        # Distractor / Noise records
        {"entity_id": "S2-901", "business_name": "Random Bakery", "business_address": "456 Elm St, Dallas, TX", "country": "US"},
        {"entity_id": "S3-902", "business_name": "French Flowers EURL", "business_address": "10 Boulevard Sud, Paris", "country": "France"},
    ]
    target_df = pd.DataFrame(target_data)

    # Build index
    idx = InvertedIndex(max_bucket_size=50)
    idx.build_from_dataframe(target_df)

    assert_test("Index built with 6 target records", len(idx.target_ids) == 6)

    # 2. Mock S1 Query records
    s1_data = [
        {"entity_id": "S1-001", "business_name": "Orelee's Barbershop", "business_address": "1795 Westchester Drive, High Point, NC", "country": "US"},
        {"entity_id": "S1-002", "business_name": "Coffee Starbucks", "business_address": "123 Main Street, Seattle", "country": "US"},
        {"entity_id": "S1-003", "business_name": "Cafe Etoile", "business_address": "15 Rue de Paris, Lyon", "country": "France"},
        {"entity_id": "S1-004", "business_name": "Unmatched Store Singleton", "business_address": "999 Nowhere Rd", "country": "US"},
    ]
    s1_df = pd.DataFrame(s1_data)

    # Mock ground truth
    gt = {
        "S1-001": {"S2-101", "S3-102"},
        "S1-002": {"S2-201"},
        "S1-003": {"S2-301"},
        "S1-004": set(),  # singleton
    }

    # Generate candidates with top_k = 5
    cands = generate_candidate_pairs(s1_df, idx, top_k=5)

    # Check S1-001 found both matches
    s1_001_cands = set(cands.get("S1-001", []))
    assert_test("S1-001 retrieved S2-101", "S2-101" in s1_001_cands)
    assert_test("S1-001 retrieved S3-102", "S3-102" in s1_001_cands)

    # Check S1-002 found Starbucks despite word order
    s1_002_cands = set(cands.get("S1-002", []))
    assert_test("S1-002 retrieved Starbucks (word-order invariant)", "S2-201" in s1_002_cands)

    # Check S1-003 found French accented match
    s1_003_cands = set(cands.get("S1-003", []))
    assert_test("S1-003 retrieved Cafe Etoile (accent invariant)", "S2-301" in s1_003_cands)

    # Check S1-004 (singleton) retrieved 0 distractors
    s1_004_cands = cands.get("S1-004", [])
    assert_test("S1-004 singleton candidate count is 0", len(s1_004_cands) == 0)

    # Evaluate blocking recall
    eval_res = evaluate_blocking_recall(cands, gt, verbose=True)
    assert_test("Blocking recall is 100%", eval_res["blocking_recall"] == 1.0)
    assert_test("Singletons count correctly recognized", eval_res["singletons"] == 1)

    print(f"\n--- Results: {passed} passed, {failed} failed ---")
    return failed == 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  python src/blocker.py --test   Run blocker test suite")
