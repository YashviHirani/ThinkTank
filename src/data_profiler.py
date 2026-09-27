"""
Phase 0, Step 4: Data Profiling
================================
Fast, fully-vectorized profiler. Avoids iterrows entirely.
Writes report to reports/data_profile.md

Usage:
    python src/data_profiler.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "data_profile.md")

# ---------------------------------------------------------------------------
# Fast loading — dtype=str, no default NA so blanks stay as ""
# ---------------------------------------------------------------------------

def load(subdir, filename):
    path = os.path.join(DATASET_DIR, subdir, filename)
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    print(f"  {filename}: {len(df):,} rows ({time.time()-t0:.1f}s)")
    return df


def main():
    t_start = time.time()
    lines = []  # accumulate markdown lines
    L = lines.append  # shorthand

    L("# Data Profile Report\n")
    L(f"Generated at script runtime.\n")

    # ------------------------------------------------------------------
    # Load all files
    # ------------------------------------------------------------------
    print("Loading files...")
    sources = {
        "train_s1": ("train", "train_source1.tsv", "S1"),
        "train_s2": ("train", "train_source2.tsv", "S2"),
        "train_s3": ("train", "train_source3.tsv", "S3"),
        "test_s1":  ("test",  "test_source1.tsv",  "S1"),
        "test_s2":  ("test",  "test_source2.tsv",  "S2"),
        "test_s3":  ("test",  "test_source3.tsv",  "S3"),
    }
    dfs = {}
    for key, (subdir, fname, _) in sources.items():
        dfs[key] = load(subdir, fname)

    print("Loading ground truth...")
    gt = pd.read_csv(
        os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv"),
        sep="\t", dtype=str, keep_default_na=False,
    )
    print(f"  train_ground_truth.tsv: {len(gt):,} rows")

    # ==================================================================
    # Section 1: Row counts & blanks
    # ==================================================================
    L("## 1. Row Counts & Blank Fields\n")
    L("| File | Rows | Blank Name | Blank Addr | Blank Country |")
    L("|---|---|---|---|---|")
    for key, (subdir, fname, prefix) in sources.items():
        df = dfs[key]
        n = len(df)
        bn = (df["business_name"].str.strip() == "").sum()
        ba = (df["business_address"].str.strip() == "").sum()
        bc = (df["country"].str.strip() == "").sum()
        L(f"| {key} | {n:,} | {bn:,} ({bn/n*100:.1f}%) | {ba:,} ({ba/n*100:.1f}%) | {bc:,} ({bc/n*100:.1f}%) |")
    L("")

    # ==================================================================
    # Section 2: Country distribution
    # ==================================================================
    L("## 2. Country Distribution\n")
    for key in sources:
        df = dfs[key]
        vc = df["country"].value_counts()
        n = len(df)
        L(f"**{key}:**\n")
        L("| Country | Count | % |")
        L("|---|---|---|")
        for country, cnt in vc.items():
            L(f"| {country} | {cnt:,} | {cnt/n*100:.1f}% |")
        L("")

    # ==================================================================
    # Section 3: Name length distributions
    # ==================================================================
    L("## 3. Business Name Length Distribution\n")
    L("| Source | Country | Min | P25 | Median | P75 | Max | Mean |")
    L("|---|---|---|---|---|---|---|---|")
    for key in sources:
        df = dfs[key]
        name_len = df["business_name"].str.len()
        for country in sorted(df["country"].unique()):
            mask = df["country"] == country
            lens = name_len[mask]
            if len(lens) == 0:
                continue
            q = np.percentile(lens, [25, 50, 75])
            L(f"| {key} | {country} | {lens.min()} | {q[0]:.0f} | {q[1]:.0f} | {q[2]:.0f} | {lens.max()} | {lens.mean():.1f} |")
    L("")

    # ==================================================================
    # Section 4: Address length distributions (non-blank only)
    # ==================================================================
    L("## 4. Business Address Length Distribution (non-blank only)\n")
    L("| Source | Country | Non-blank | Blank% | Min | P25 | Median | P75 | Max | Mean |")
    L("|---|---|---|---|---|---|---|---|---|---|")
    for key in sources:
        df = dfs[key]
        addr = df["business_address"].str.strip()
        has_addr = addr != ""
        for country in sorted(df["country"].unique()):
            cmask = df["country"] == country
            total_c = cmask.sum()
            valid = has_addr & cmask
            n_valid = valid.sum()
            blank_pct = (1 - n_valid / total_c) * 100 if total_c > 0 else 0
            if n_valid == 0:
                L(f"| {key} | {country} | 0 | {blank_pct:.1f}% | - | - | - | - | - | - |")
                continue
            lens = addr[valid].str.len()
            q = np.percentile(lens, [25, 50, 75])
            L(f"| {key} | {country} | {n_valid:,} | {blank_pct:.1f}% | {lens.min()} | {q[0]:.0f} | {q[1]:.0f} | {q[2]:.0f} | {lens.max()} | {lens.mean():.1f} |")
    L("")

    # ==================================================================
    # Section 5: Postcode & numeric token availability
    # ==================================================================
    L("## 5. Postcode & Numeric Token Availability\n")

    # Precompile regexes
    re_5digit = re.compile(r'\b\d{5}\b')
    re_6digit = re.compile(r'\b\d{6}\b')
    re_any_number = re.compile(r'\d+')

    L("| Source | Country | Has Postcode | Postcode% | Has Any Number | Number% |")
    L("|---|---|---|---|---|---|")
    for key in sources:
        df = dfs[key]
        addr = df["business_address"]
        for country in sorted(df["country"].unique()):
            cmask = df["country"] == country
            addrs = addr[cmask]
            total_c = len(addrs)
            if total_c == 0:
                continue

            # Postcode detection: US/France=5-digit, India=6-digit
            if country in ("US", "France"):
                has_pc = addrs.str.contains(r'\b\d{5}\b', regex=True, na=False).sum()
            elif country == "India":
                has_pc = addrs.str.contains(r'\b\d{6}\b', regex=True, na=False).sum()
            else:
                has_pc = addrs.str.contains(r'\b\d{5,6}\b', regex=True, na=False).sum()

            has_num = addrs.str.contains(r'\d+', regex=True, na=False).sum()
            L(f"| {key} | {country} | {has_pc:,} | {has_pc/total_c*100:.1f}% | {has_num:,} | {has_num/total_c*100:.1f}% |")
    L("")

    # ==================================================================
    # Section 6: S1 match-count distribution (ground truth)
    # ==================================================================
    L("## 6. Ground Truth Match Distribution\n")

    print("Analysing ground truth distribution...")

    # Vectorized: count commas + 1 for non-empty, 0 for empty
    matched_col = gt["matched_entity_ids"]
    is_empty = matched_col.str.strip() == ""
    # For non-empty rows, count matches = number of commas + 1
    match_counts = np.where(
        is_empty,
        0,
        matched_col.str.count(",") + 1
    )
    gt_mc = pd.Series(match_counts, name="match_count")

    total_s1 = len(gt)
    L("### Match count per S1 entity\n")
    L("| Matches | Count | % |")
    L("|---|---|---|")
    for k in range(0, 8):
        c = (gt_mc == k).sum()
        if c > 0:
            label = f"{k} (singleton)" if k == 0 else str(k)
            L(f"| {label} | {c:,} | {c/total_s1*100:.1f}% |")
    c_8plus = (gt_mc >= 8).sum()
    if c_8plus > 0:
        L(f"| 8+ | {c_8plus:,} | {c_8plus/total_s1*100:.1f}% |")
    L("")

    avg_all = gt_mc.mean()
    avg_nonzero = gt_mc[gt_mc > 0].mean() if (gt_mc > 0).any() else 0
    L(f"- **Average matches per S1 (all):** {avg_all:.2f}")
    L(f"- **Average matches per S1 (non-singleton):** {avg_nonzero:.2f}")
    L(f"- **Max matches for one S1:** {gt_mc.max()}")
    L("")

    # S2 vs S3 breakdown — vectorized
    print("Analysing S2 vs S3 breakdown...")
    # For each non-empty row, check if it has S2- and/or S3- IDs
    has_s2 = matched_col.str.contains("S2-", na=False) & ~is_empty
    has_s3 = matched_col.str.contains("S3-", na=False) & ~is_empty
    s2_only = (has_s2 & ~has_s3).sum()
    s3_only = (~has_s2 & has_s3).sum()
    both = (has_s2 & has_s3).sum()
    neither = is_empty.sum()

    L("### S2 vs S3 link breakdown (non-singleton S1 entities)\n")
    L("| Category | Count | % of non-singleton |")
    L("|---|---|---|")
    non_single = total_s1 - neither
    L(f"| S2 matches only | {s2_only:,} | {s2_only/non_single*100:.1f}% |")
    L(f"| S3 matches only | {s3_only:,} | {s3_only/non_single*100:.1f}% |")
    L(f"| Both S2 and S3 | {both:,} | {both/non_single*100:.1f}% |")
    L("")

    # Many-to-many check: are any S2/S3 IDs matched to multiple S1 entities?
    print("Checking many-to-many relationships...")
    # Explode matched IDs — vectorized split
    non_empty_gt = gt[~is_empty][["source1_entity_id", "matched_entity_ids"]].copy()
    exploded = non_empty_gt["matched_entity_ids"].str.split(",", expand=False)
    s1_ids_repeated = non_empty_gt["source1_entity_id"].repeat(exploded.str.len())
    all_match_ids = pd.Series(
        [mid.strip() for sublist in exploded for mid in sublist],
        name="match_id"
    )

    vc = all_match_ids.value_counts()
    multi_assigned = (vc > 1).sum()
    max_assigned = vc.max()

    L("### Many-to-many check\n")
    L(f"- **Total unique S2/S3 IDs in ground truth:** {len(vc):,}")
    s2_unique = all_match_ids.str.startswith("S2-").sum()  # total references
    s3_unique = all_match_ids.str.startswith("S3-").sum()
    s2_uniq_ids = all_match_ids[all_match_ids.str.startswith("S2-")].nunique()
    s3_uniq_ids = all_match_ids[all_match_ids.str.startswith("S3-")].nunique()
    L(f"- **S2 IDs:** {s2_uniq_ids:,} unique (referenced {s2_unique:,} times)")
    L(f"- **S3 IDs:** {s3_uniq_ids:,} unique (referenced {s3_unique:,} times)")
    L(f"- **S2/S3 IDs matched to >1 S1 entity:** {multi_assigned:,}")
    L(f"- **Max S1 entities sharing one S2/S3 ID:** {max_assigned}")
    L("")
    if multi_assigned > 0:
        L("> **NOTE:** Some target IDs match multiple S1 entities. Do NOT use a 1-to-1 assignment constraint.\n")
    else:
        L("> All target IDs match at most one S1 entity. A 1-to-1 constraint could be considered.\n")

    # ==================================================================
    # Section 7: Non-Latin script prevalence
    # ==================================================================
    L("## 7. Non-Latin Script Names\n")
    L("| Source | Country | Non-ASCII Names | % |")
    L("|---|---|---|---|")
    for key in sources:
        df = dfs[key]
        has_nonascii = df["business_name"].str.contains(r'[^\x00-\x7F]', regex=True, na=False)
        for country in sorted(df["country"].unique()):
            cmask = df["country"] == country
            total_c = cmask.sum()
            c = (has_nonascii & cmask).sum()
            if c > 0:
                L(f"| {key} | {country} | {c:,} | {c/total_c*100:.1f}% |")
    L("")

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    elapsed = time.time() - t_start
    print(f"\nReport written to: {REPORT_PATH}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
