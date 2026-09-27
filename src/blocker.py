"""
Phase 1.2 — Multi-Key Blocking
================================
Builds inverted indexes over S2+S3, then retrieves candidate matches for
every S1 entity using multiple blocking keys. Unions candidates, applies
a cheap pre-score, and caps at top-K per S1.

Blocking keys used:
  1. Exact core name (normalised)
  2. First strong name token + postcode
  3. First strong name token + house number
  4. Name alpha prefix (first 8 chars)
  5. Token overlap (bigrams from name_tokens_str)
  6. Address house number + first addr token

Returns a dict: { s1_id: [(candidate_id, prescore), ...] } — capped at K.
The exact capped set is what goes to candidate_pairs.tsv.

Usage:
    from src.blocker import BlockingIndex
    idx = BlockingIndex(s2_df_norm, s3_df_norm)
    candidates = idx.retrieve(s1_df_norm, K=20)
"""

import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Helper: extract "strong" tokens (non-suffix, len >= 3)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    'the', 'and', 'of', 'in', 'at', 'to', 'for', 'a', 'an',
    'de', 'du', 'la', 'le', 'les', 'et', 'en',  # French
    'ki', 'ka', 'ke', 'aur', 'se',               # Hindi romanised
])


def _strong_tokens(name_tokens_str: str, min_len: int = 3) -> List[str]:
    """Return tokens that are not stopwords and have length >= min_len."""
    tokens = name_tokens_str.split()
    return [t for t in tokens if len(t) >= min_len and t not in _STOPWORDS]


def _bigrams(name_tokens_str: str) -> Set[str]:
    """Character bigrams from the full name_tokens_str (for fuzzy matching)."""
    s = name_tokens_str.replace(' ', '')
    return set(s[i:i+2] for i in range(len(s)-1))


def _token_pair_key(token: str, value: str) -> str:
    return f"{token}|{value}"


# ---------------------------------------------------------------------------
# Pre-scorer: cheap, deterministic, no ML
# ---------------------------------------------------------------------------

def _prescore(s1_row, tgt_row) -> float:
    """
    Cheap pre-score combining:
      - name similarity (token sort ratio, range 0-100)
      - address similarity (partial ratio, range 0-100)
      - country agreement bonus
    Returns a float in [0, 1].
    """
    name_sim = fuzz.token_sort_ratio(
        s1_row['name_punct'], tgt_row['name_punct']
    ) / 100.0

    # address sim — only if both are non-empty
    s1_addr = s1_row.get('addr_punct', '')
    tgt_addr = tgt_row.get('addr_punct', '')
    if s1_addr and tgt_addr:
        addr_sim = fuzz.partial_ratio(s1_addr, tgt_addr) / 100.0
    else:
        addr_sim = 0.3  # neutral if one is blank

    # Country bonus
    country_bonus = 0.05 if (
        s1_row.get('country_norm', '') == tgt_row.get('country_norm', '')
        and s1_row.get('country_norm', '') != ''
    ) else 0.0

    # Weighted sum
    return 0.60 * name_sim + 0.35 * addr_sim + country_bonus


# ---------------------------------------------------------------------------
# Blocking Index
# ---------------------------------------------------------------------------

class BlockingIndex:
    """
    Multi-key inverted index over S2 + S3 normalised DataFrames.

    Parameters
    ----------
    s2_norm : normalised S2 DataFrame
    s3_norm : normalised S3 DataFrame
    """

    def __init__(self, s2_norm: pd.DataFrame, s3_norm: pd.DataFrame):
        t0 = time.time()
        print("  Building blocking index...")

        # Merge S2 + S3 into a single target DataFrame
        self._target_df = pd.concat([s2_norm, s3_norm], ignore_index=True)

        # Build lookup dict AND all indexes in a SINGLE pass via to_dict('records')
        self._target_dict: Dict[str, dict] = {}

        self._idx_core_name: Dict[str, Set[str]] = defaultdict(set)
        self._idx_tok_pc:    Dict[str, Set[str]] = defaultdict(set)
        self._idx_tok_house: Dict[str, Set[str]] = defaultdict(set)
        self._idx_prefix:    Dict[str, Set[str]] = defaultdict(set)
        self._idx_bigrams:   Dict[str, Set[str]] = defaultdict(set)
        self._idx_house_tok: Dict[str, Set[str]] = defaultdict(set)

        records = self._target_df.to_dict('records')
        print(f"  Indexing {len(records):,} target records...", flush=True)

        for row in records:
            eid = row['entity_id']
            self._target_dict[eid] = row

            nc  = row.get('name_core', '') or ''
            na  = row.get('name_alpha', '') or ''
            nts = row.get('name_tokens_str', '') or ''
            pc  = row.get('addr_postcode', '') or ''
            hn  = row.get('addr_house', '') or ''
            at  = row.get('addr_punct', '') or ''

            nc  = nc.strip()
            pc  = pc.strip()
            hn  = hn.strip()

            # Key 1: exact core name
            if nc:
                self._idx_core_name[nc].add(eid)

            # Keys 2 & 3: first strong token + postcode / house
            strong = _strong_tokens(nts)
            if strong:
                first = strong[0]
                if pc:
                    self._idx_tok_pc[_token_pair_key(first, pc)].add(eid)
                if hn:
                    self._idx_tok_house[_token_pair_key(first, hn)].add(eid)

            # Key 4: name alpha prefix (first 8 chars)
            if len(na) >= 4:
                self._idx_prefix[na[:8]].add(eid)

            # Key 5: bigrams over name_tokens_str
            for bg in _bigrams(nts):
                self._idx_bigrams[bg].add(eid)

            # Key 6: house number + first addr token
            if hn and at:
                addr_tokens = at.split()
                if addr_tokens:
                    first_at = addr_tokens[0] if addr_tokens[0] != hn else (addr_tokens[1] if len(addr_tokens) > 1 else '')
                    if first_at:
                        self._idx_house_tok[_token_pair_key(hn, first_at)].add(eid)

        elapsed = time.time() - t0
        print(f"  Blocking index built in {elapsed:.1f}s | "
              f"target records: {len(self._target_dict):,}")

    def _retrieve_candidates(self, s1_row: dict) -> Set[str]:
        """Use all keys to retrieve candidate target IDs for one S1 row."""
        candidates: Set[str] = set()

        nc  = s1_row.get('name_core', '').strip()
        na  = s1_row.get('name_alpha', '')
        nts = s1_row.get('name_tokens_str', '')
        pc  = s1_row.get('addr_postcode', '').strip()
        hn  = s1_row.get('addr_house', '').strip()
        at  = s1_row.get('addr_punct', '')

        # Key 1
        if nc:
            candidates.update(self._idx_core_name.get(nc, set()))

        # Keys 2 & 3
        strong = _strong_tokens(nts)
        if strong:
            first = strong[0]
            if pc:
                candidates.update(self._idx_tok_pc.get(_token_pair_key(first, pc), set()))
            if hn:
                candidates.update(self._idx_tok_house.get(_token_pair_key(first, hn), set()))

        # Key 4: prefix
        if len(na) >= 4:
            candidates.update(self._idx_prefix.get(na[:8], set()))

        # Key 5: bigrams — intersection approach (need ≥2 common bigrams to avoid noise)
        if nts:
            bgs = _bigrams(nts)
            bg_candidates: Dict[str, int] = defaultdict(int)
            for bg in bgs:
                for eid in self._idx_bigrams.get(bg, set()):
                    bg_candidates[eid] += 1
            # Only add candidates with ≥3 bigrams in common
            for eid, count in bg_candidates.items():
                if count >= 3:
                    candidates.add(eid)

        # Key 6: house + addr token
        addr_tokens = at.split()
        if hn and addr_tokens:
            first_at = addr_tokens[0] if addr_tokens[0] != hn else (addr_tokens[1] if len(addr_tokens) > 1 else '')
            if first_at:
                candidates.update(self._idx_house_tok.get(_token_pair_key(hn, first_at), set()))

        return candidates

    def retrieve(
        self,
        s1_norm: pd.DataFrame,
        K: int = 20,
        country_filter: bool = True,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        For every S1 entity, retrieve candidates, pre-score, and cap at K.

        Parameters
        ----------
        s1_norm       : normalised S1 DataFrame
        K             : maximum candidates to retain per S1 entity
        country_filter: if True, prefer same-country candidates

        Returns
        -------
        dict mapping s1_entity_id → list of (target_id, prescore) sorted desc
        """
        t0 = time.time()
        total = len(s1_norm)
        print(f"  Retrieving candidates for {total:,} S1 entities (K={K})...")

        results: Dict[str, List[Tuple[str, float]]] = {}
        report_every = max(1, total // 20)

        s1_records = s1_norm.to_dict('records')

        for i, s1_dict in enumerate(s1_records):
            if i > 0 and i % report_every == 0:
                elapsed = time.time() - t0
                pct = i / total * 100
                print(f"    {i:,}/{total:,} ({pct:.0f}%) — {elapsed:.0f}s elapsed", flush=True)

            s1_id = s1_dict['entity_id']

            raw_candidates = self._retrieve_candidates(s1_dict)

            if not raw_candidates:
                results[s1_id] = []
                continue

            # Country filter: only apply if profiling shows countries are reliable
            if country_filter:
                s1_country = s1_dict.get('country_norm', '')
                filtered = frozenset(
                    cid for cid in raw_candidates
                    if self._target_dict[cid].get('country_norm', '') == s1_country
                )
                # Fallback: if filter removes everything, revert
                if not filtered:
                    filtered = raw_candidates
                raw_candidates = filtered

            # Pre-score all candidates
            scored = [
                (cid, _prescore(s1_dict, self._target_dict[cid]))
                for cid in raw_candidates
            ]

            # Sort descending by prescore, keep top K
            scored.sort(key=lambda x: -x[1])
            results[s1_id] = scored[:K]

        elapsed = time.time() - t0
        avg_cands = sum(len(v) for v in results.values()) / max(1, len(results))
        max_cands = max((len(v) for v in results.values()), default=0)
        print(f"  Retrieval complete in {elapsed:.1f}s | "
              f"avg candidates: {avg_cands:.1f} | max: {max_cands}")

        return results


# ---------------------------------------------------------------------------
# Blocking stats reporter
# ---------------------------------------------------------------------------

def blocking_stats(
    candidates: Dict[str, List[Tuple[str, float]]],
    gt_matches: Dict[str, set],
    s1_ids_in_split: set,
) -> dict:
    """
    Compute blocking recall metrics for a given candidate dict.

    Parameters
    ----------
    candidates       : {s1_id: [(tgt_id, prescore), ...]}
    gt_matches       : {s1_id: set(tgt_id)}  — ground truth
    s1_ids_in_split  : set of S1 IDs in this split

    Returns
    -------
    dict with blocking_recall, avg_candidates, max_candidates, reduction_ratio
    """
    gt_val = {k: v for k, v in gt_matches.items() if k in s1_ids_in_split and v}

    if not gt_val:
        return {'blocking_recall': 0.0, 'avg_candidates': 0, 'max_candidates': 0}

    hits = 0
    total_true = 0
    cand_counts = []

    for s1_id, true_matches in gt_val.items():
        cand_set = set(c[0] for c in candidates.get(s1_id, []))
        matched = len(true_matches & cand_set)
        hits += matched
        total_true += len(true_matches)
        cand_counts.append(len(cand_set))

    blocking_recall = hits / total_true if total_true > 0 else 0.0
    avg_cands = np.mean(cand_counts) if cand_counts else 0
    max_cands = max(cand_counts) if cand_counts else 0

    return {
        'blocking_recall': round(blocking_recall, 4),
        'avg_candidates': round(float(avg_cands), 2),
        'max_candidates': int(max_cands),
    }
