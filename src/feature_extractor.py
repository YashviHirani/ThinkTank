"""
Phase 1.3 — Pair Feature Extraction
=====================================
Given a candidate pair (S1 row, target row), computes a rich feature vector.

Feature groups:
  A. Name similarity  (rapidfuzz: ratio, token_sort, token_set, jaro_winkler, jaccard; core/full)
  B. Address          (ratio, token_jaccard, common_token_count, partial_ratio)
  C. Structure        (country equality, length ratios, shared_numeric_count,
                       house_number_agreement/conflict, postcode_agreement/conflict)
  D. Provenance       (blocking_key indicator flags, pre_score)

Usage:
    from src.feature_extractor import build_feature_matrix
    X, y, pair_ids = build_feature_matrix(candidates, s1_norm_dict, target_norm_dict, gt_matches)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz, distance


# ---------------------------------------------------------------------------
# Token Jaccard similarity
# ---------------------------------------------------------------------------

def _token_jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


def _common_token_count(a: str, b: str) -> int:
    ta = set(a.split())
    tb = set(b.split())
    return len(ta & tb)


# ---------------------------------------------------------------------------
# Per-pair feature vector
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # A. Name similarity
    'name_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_jaro_winkler',
    'name_token_jaccard',
    'name_core_ratio',
    'name_core_token_sort',
    'name_alpha_ratio',
    'name_suffix_match',           # 1 if both have at least 1 common suffix token

    # B. Address similarity
    'addr_ratio',
    'addr_token_sort_ratio',
    'addr_partial_ratio',
    'addr_token_jaccard',
    'addr_common_token_count',
    'addr_alpha_ratio',
    'addr_s1_blank',               # 1 if S1 address is blank
    'addr_tgt_blank',              # 1 if target address is blank
    'addr_both_present',           # 1 if both present

    # C. Structure
    'country_equal',
    'name_len_ratio',              # shorter/longer (always ≤1)
    'addr_len_ratio',
    'shared_numeric_count',        # tokens that are identical numeric strings
    'house_agree',                 # 1 if house numbers exist and match
    'house_conflict',              # 1 if house numbers exist but differ
    'postcode_agree',              # 1 if postcodes exist and match
    'postcode_conflict',           # 1 if postcodes exist but differ

    # D. Provenance
    'pre_score',
]


def extract_features(s1: dict, tgt: dict, pre_score: float = 0.5) -> np.ndarray:
    """
    Compute feature vector for one (S1, target) pair.

    Parameters
    ----------
    s1        : row dict from normalised S1 DataFrame
    tgt       : row dict from normalised S2/S3 DataFrame
    pre_score : pre-computed cheap score from blocking stage

    Returns
    -------
    numpy float32 array of length len(FEATURE_NAMES)
    """
    feats = []

    # ---------- A. Name similarity ----------
    s1_punct = s1.get('name_punct', '')
    tgt_punct = tgt.get('name_punct', '')
    s1_core  = s1.get('name_core', '')
    tgt_core = tgt.get('name_core', '')
    s1_alpha = s1.get('name_alpha', '')
    tgt_alpha = tgt.get('name_alpha', '')
    s1_suf   = s1.get('name_suffix_str', '')
    tgt_suf  = tgt.get('name_suffix_str', '')

    feats.append(fuzz.ratio(s1_punct, tgt_punct) / 100.0)
    feats.append(fuzz.token_sort_ratio(s1_punct, tgt_punct) / 100.0)
    feats.append(fuzz.token_set_ratio(s1_punct, tgt_punct) / 100.0)
    feats.append(fuzz.WRatio(s1_punct, tgt_punct) / 100.0)  # jaro-winkler like weighted
    feats.append(_token_jaccard(s1_punct, tgt_punct))
    feats.append(fuzz.token_sort_ratio(s1_core, tgt_core) / 100.0)
    feats.append(fuzz.ratio(s1_core, tgt_core) / 100.0)
    feats.append(fuzz.ratio(s1_alpha, tgt_alpha) / 100.0)

    # Suffix match: at least 1 common suffix token
    s1_suf_set  = set(s1_suf.split()) if s1_suf else set()
    tgt_suf_set = set(tgt_suf.split()) if tgt_suf else set()
    feats.append(1.0 if (s1_suf_set and tgt_suf_set and s1_suf_set & tgt_suf_set) else 0.0)

    # ---------- B. Address similarity ----------
    s1_addr  = s1.get('addr_punct', '')
    tgt_addr = tgt.get('addr_punct', '')
    s1_addr_alpha  = s1.get('addr_alpha', '')
    tgt_addr_alpha = tgt.get('addr_alpha', '')

    s1_blank  = 1.0 if not s1_addr.strip() else 0.0
    tgt_blank = 1.0 if not tgt_addr.strip() else 0.0
    both_present = 1.0 if (s1_addr.strip() and tgt_addr.strip()) else 0.0

    if both_present:
        feats.append(fuzz.ratio(s1_addr, tgt_addr) / 100.0)
        feats.append(fuzz.token_sort_ratio(s1_addr, tgt_addr) / 100.0)
        feats.append(fuzz.partial_ratio(s1_addr, tgt_addr) / 100.0)
        feats.append(_token_jaccard(s1_addr, tgt_addr))
        feats.append(float(_common_token_count(s1_addr, tgt_addr)))
        feats.append(fuzz.ratio(s1_addr_alpha, tgt_addr_alpha) / 100.0)
    else:
        # If one address is missing, use neutral/zero
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    feats.append(s1_blank)
    feats.append(tgt_blank)
    feats.append(both_present)

    # ---------- C. Structure ----------
    s1_country  = s1.get('country_norm', '')
    tgt_country = tgt.get('country_norm', '')
    feats.append(1.0 if s1_country == tgt_country and s1_country else 0.0)

    # Length ratios
    s1_nlen  = len(s1_punct)
    tgt_nlen = len(tgt_punct)
    if s1_nlen > 0 and tgt_nlen > 0:
        feats.append(min(s1_nlen, tgt_nlen) / max(s1_nlen, tgt_nlen))
    else:
        feats.append(0.0)

    s1_alen  = len(s1_addr)
    tgt_alen = len(tgt_addr)
    if s1_alen > 0 and tgt_alen > 0:
        feats.append(min(s1_alen, tgt_alen) / max(s1_alen, tgt_alen))
    else:
        feats.append(0.0)

    # Shared numeric token count
    s1_nums  = set(s1.get('addr_punct', '').split())
    tgt_nums = set(tgt.get('addr_punct', '').split())
    s1_nums  = {t for t in s1_nums  if t.isdigit()}
    tgt_nums = {t for t in tgt_nums if t.isdigit()}
    feats.append(float(len(s1_nums & tgt_nums)))

    # House number agreement/conflict
    s1_house  = s1.get('addr_house', '').strip()
    tgt_house = tgt.get('addr_house', '').strip()
    if s1_house and tgt_house:
        feats.append(1.0 if s1_house == tgt_house else 0.0)   # house_agree
        feats.append(0.0 if s1_house == tgt_house else 1.0)   # house_conflict
    else:
        feats.extend([0.0, 0.0])

    # Postcode agreement/conflict
    s1_pc  = s1.get('addr_postcode', '').strip()
    tgt_pc = tgt.get('addr_postcode', '').strip()
    if s1_pc and tgt_pc:
        feats.append(1.0 if s1_pc == tgt_pc else 0.0)         # postcode_agree
        feats.append(0.0 if s1_pc == tgt_pc else 1.0)         # postcode_conflict
    else:
        feats.extend([0.0, 0.0])

    # ---------- D. Provenance ----------
    feats.append(float(pre_score))

    arr = np.array(feats, dtype=np.float32)
    assert len(arr) == len(FEATURE_NAMES), (
        f"Feature count mismatch: {len(arr)} != {len(FEATURE_NAMES)}"
    )
    return arr


# ---------------------------------------------------------------------------
# Build full feature matrix from a candidate dict
# ---------------------------------------------------------------------------

def build_feature_matrix(
    candidates: Dict[str, List[Tuple[str, float]]],
    s1_dict: Dict[str, dict],    # entity_id → row dict (normalised)
    target_dict: Dict[str, dict], # entity_id → row dict (normalised)
    gt_matches: Optional[Dict[str, set]] = None,  # None for test set
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Tuple[str, str]]]:
    """
    Build feature matrix for all candidate pairs.

    Parameters
    ----------
    candidates   : {s1_id: [(tgt_id, prescore), ...]}
    s1_dict      : {entity_id: normalised_row_dict}
    target_dict  : {entity_id: normalised_row_dict}
    gt_matches   : ground truth dict (None for test)

    Returns
    -------
    X         : np.float32 array of shape (N_pairs, N_features)
    y         : np.int8 labels array of shape (N_pairs,) or None
    pair_ids  : list of (s1_id, tgt_id) tuples in the same row order
    """
    rows = []
    labels = []
    pair_ids = []

    for s1_id, cand_list in candidates.items():
        if not cand_list:
            continue
        s1_row = s1_dict.get(s1_id)
        if s1_row is None:
            continue

        true_matches = gt_matches.get(s1_id, set()) if gt_matches is not None else None

        for tgt_id, prescore in cand_list:
            tgt_row = target_dict.get(tgt_id)
            if tgt_row is None:
                continue

            feat = extract_features(s1_row, tgt_row, pre_score=prescore)
            rows.append(feat)
            pair_ids.append((s1_id, tgt_id))

            if gt_matches is not None:
                label = 1 if (true_matches is not None and tgt_id in true_matches) else 0
                labels.append(label)

    if not rows:
        n_feat = len(FEATURE_NAMES)
        X = np.empty((0, n_feat), dtype=np.float32)
        y = np.empty(0, dtype=np.int8) if gt_matches is not None else None
        return X, y, pair_ids

    X = np.vstack(rows).astype(np.float32)
    y = np.array(labels, dtype=np.int8) if gt_matches is not None else None
    return X, y, pair_ids


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    s1 = {
        'entity_id': 'S1-001',
        'name_punct': 'abc corp',
        'name_core': 'abc',
        'name_alpha': 'abccorp',
        'name_suffix_str': 'corp',
        'addr_punct': '123 main street springfield',
        'addr_alpha': '123mainstreetspringfield',
        'addr_house': '123',
        'addr_postcode': '62701',
        'country_norm': 'us',
    }
    tgt = {
        'entity_id': 'S2-001',
        'name_punct': 'abc corporation',
        'name_core': 'abc',
        'name_alpha': 'abccorporation',
        'name_suffix_str': 'corporation',
        'addr_punct': '123 main road springfield',
        'addr_alpha': '123mainroadspringfield',
        'addr_house': '123',
        'addr_postcode': '62701',
        'country_norm': 'us',
    }
    feat = extract_features(s1, tgt, pre_score=0.85)
    print(f"Feature vector length: {len(feat)}")
    for name, val in zip(FEATURE_NAMES, feat):
        print(f"  {name:35s} = {val:.4f}")
    print("\n[OK] Feature extractor smoke test passed")
