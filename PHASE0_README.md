# Phase 0 — Intake, Profiling & Evaluation

## Overview

Phase 0 sets up the foundation before any model work. It loads and validates the raw data, profiles it to understand noise patterns, builds the official scoring metric, and creates a bias-free train/validation split.

**No data is modified or cleaned in Phase 0** — that happens in Phase 1.

---

## What was built

| Step | Script | Purpose | Output |
|---|---|---|---|
| 1 | *(manual)* | Created project folder structure | `dataset/`, `src/`, `experiments/`, `models/`, `output/`, `reports/`, `utils/` |
| 2-3 | `src/data_loader.py` | Load all 7 TSV files, validate schemas, parse ground truth into Python sets, run cross-reference integrity assertions | Console output with all assertions |
| 4 | `src/data_profiler.py` | Profile all data: row counts, blanks, country distribution, name/address lengths, postcode availability, match-count distribution, non-Latin script prevalence | `reports/data_profile.md` |
| 5 | `src/scorer.py` | Grouped macro F0.5 scorer matching competition formula: `F0.5 = (1.25 * P * R) / (0.25 * P + R)` | Importable functions + unit tests |
| 6 | `src/splitter.py` | Bias-free 80/20 train/validation split, stratified on country + match-count bucket | `experiments/train_s1_ids.txt`, `experiments/val_s1_ids.txt`, `experiments/split_metadata.json` |

---

## How to run

### Prerequisites

```bash
pip install pandas numpy scikit-learn
```

### Step 2-3: Data Loading & Validation

```bash
python src/data_loader.py
```

**What it does:**
- Loads all 7 TSV files (`train_source1/2/3.tsv`, `test_source1/2/3.tsv`, `train_ground_truth.tsv`)
- Validates each file has the correct columns
- Checks all entity IDs have the right prefix (`S1-`, `S2-`, `S3-`)
- Checks for duplicate IDs
- Parses `matched_entity_ids` into Python sets (singletons get empty set)
- Cross-references: every S1 in ground truth exists in source1, every match ID exists in source2/source3

**Expected output:**
```
==================================================
  Data Loading & Validation
==================================================

--- Source Files ---
  Loading train_source1.tsv ... 2,206,821 rows
  Loading train_source2.tsv ... 5,034,616 rows
  Loading train_source3.tsv ... 5,285,603 rows
  Loading test_source1.tsv ... 1,732,544 rows
  Loading test_source2.tsv ... 4,887,273 rows
  Loading test_source3.tsv ... 5,082,316 rows

--- Ground Truth ---
  Loading train_ground_truth.tsv ... 2,206,821 rows
  Parsing matched_entity_ids ... done

--- Integrity Checks ---
  [OK] All 2,206,821 ground truth S1 IDs exist in train_source1
  [OK] All 7,638,365 match IDs have S2- or S3- prefix
  [OK] All 3,693,619 S2 match IDs exist in train_source2
  [OK] All 3,944,746 S3 match IDs exist in train_source3
  [OK] Every S1 in train_source1 has a ground truth entry

  All assertions PASSED [OK]
```

**Runtime:** ~3-4 minutes (loading ~2GB of data)

**As a module:**
```python
from src.data_loader import load_all_data
data = load_all_data()
# data["train_s1"], data["train_s2"], data["gt_matches"], etc.
```

---

### Step 4: Data Profiling

```bash
python src/data_profiler.py
```

**What it does:**
- Loads all 6 source files + ground truth
- Computes 7 sections of statistics:
  1. Row counts & blank field rates per source
  2. Country distribution (train vs test — France is new in test)
  3. Business name length distributions per source & country
  4. Address length distributions (non-blank only)
  5. Postcode & numeric token availability per country
  6. Ground truth match-count distribution, S2 vs S3 breakdown, many-to-many check
  7. Non-Latin script name prevalence

**Output:** `reports/data_profile.md`

**Key findings from the profile:**
- **0% blank names**, 2.6-3.4% blank addresses in S2/S3
- **India has 0% postcodes**, US has ~10% — can't rely on postcode blocking for India
- **90%+ addresses have numeric tokens** — useful for blocking
- **Average 3.5 matches per S1**, max 11 — multi-match is the norm
- **Each S2/S3 maps to at most 1 S1** — 1-to-1 constraint on target side is safe
- **28% of India-S2 names are non-ASCII** — needs Unicode normalisation
- **France is 15% of test** with 24% non-ASCII names (accented characters)

**Runtime:** ~2-3 minutes

---

### Step 5: F0.5 Scorer (Unit Tests)

```bash
python src/scorer.py --test
```

**What it does:**
- Runs 8 unit tests verifying the F0.5 scoring formula:
  1. Correct singleton → 1.0
  2. False match on singleton → 0.0
  3. Perfect multi-match → 1.0
  4. One correct + one extra prediction → 0.714 (precision penalty)
  5. Missed all matches → 0.0
  6. Partial match (1 of 2, no extras) → 0.833 (precision rewarded)
  7. All wrong predictions → 0.0
  8. Macro F0.5 integration test → 0.667

**Expected output:**
```
--- Unit Tests ---

  [PASS] Correct singleton: expected=1.0000, got=1.0000
  [PASS] False match on singleton: expected=0.0000, got=0.0000
  [PASS] Perfect multi-match: expected=1.0000, got=1.0000
  [PASS] One correct + one extra: expected=0.7143, got=0.7143
  [PASS] Missed all matches: expected=0.0000, got=0.0000
  [PASS] Partial match (1 of 2, no extras): expected=0.8333, got=0.8333
  [PASS] All wrong predictions: expected=0.0000, got=0.0000

  --- Macro F0.5 integration test ---

  [PASS] Macro F0.5 integration: expected=0.6667, got=0.6667

--- Results: 8 passed, 0 failed ---
```

**Runtime:** Instant (<1 second)

**As a module (used in later phases):**
```python
from src.scorer import macro_f05, f05_single

# Score one entity
score = f05_single(
    truth={"S2-100", "S3-200"},
    preds={"S2-100"}
)
# score = 0.8333

# Score all entities
final_score = macro_f05(gt_matches, predictions, verbose=True)
```

---

### Step 6: Train/Validation Split

```bash
python src/splitter.py
```

**What it does:**
- Loads ground truth + country from `train_source1.tsv`
- Validates merge (one-to-one, no missing countries)
- Computes match counts safely (handles trailing commas, malformed values)
- Buckets: `0 → singleton`, `1-2 → low`, `3-4 → medium`, `5+ → high`
- Builds stratification key: `country + match_bucket` (e.g., `US_singleton`, `India_high`)
- Splits 80/20 with `random_state=42`
- Verifies: zero overlap, complete coverage, no duplicates
- Prints full balance report with 0.5pp difference flagging

**Output files:**
```
experiments/train_s1_ids.txt      # 1,765,456 S1 IDs (one per line, no header)
experiments/val_s1_ids.txt        #   441,365 S1 IDs (one per line, no header)
experiments/split_metadata.json   # seed, counts, buckets, timestamp
```

**Split balance (all perfectly balanced at 0.00pp difference):**

| Dimension | Train | Val |
|---|---|---|
| US | 60.0% | 60.0% |
| India | 40.0% | 40.0% |
| Singleton | 5.6% | 5.6% |
| Low (1-2 matches) | 22.4% | 22.4% |
| Medium (3-4) | 46.0% | 46.0% |
| High (5+) | 26.0% | 26.0% |
| Avg match count | 3.4609 | 3.4626 |

**Runtime:** ~35 seconds

**Loading the split in later scripts:**
```python
train_ids = set(open("experiments/train_s1_ids.txt").read().splitlines())
val_ids = set(open("experiments/val_s1_ids.txt").read().splitlines())
```

---

## Run all of Phase 0

```bash
# Step 2-3: Validate data integrity
python src/data_loader.py

# Step 4: Generate data profile report
python src/data_profiler.py

# Step 5: Run scorer unit tests
python src/scorer.py --test

# Step 6: Create train/validation split
python src/splitter.py
```

Total time: ~7-8 minutes (dominated by data loading in steps 2-4).

---

## Project structure after Phase 0

```
ThinkTank/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv        # 2,206,821 S1 entities
│   │   ├── train_source2.tsv        # 5,034,616 S2 entities
│   │   ├── train_source3.tsv        # 5,285,603 S3 entities
│   │   └── train_ground_truth.tsv   # 2,206,821 label rows
│   └── test/
│       ├── test_source1.tsv         # 1,732,544 S1 entities
│       ├── test_source2.tsv         # 4,887,273 S2 entities
│       └── test_source3.tsv         # 5,082,316 S3 entities
├── src/
│   ├── data_loader.py               # Load & validate all data
│   ├── data_profiler.py             # Generate profile report
│   ├── scorer.py                    # F0.5 scorer + unit tests
│   └── splitter.py                  # Train/val split
├── experiments/
│   ├── train_s1_ids.txt             # 1,765,456 train S1 IDs
│   ├── val_s1_ids.txt               # 441,365 val S1 IDs
│   └── split_metadata.json          # Split parameters
├── reports/
│   └── data_profile.md              # Full data profile report
├── models/                          # (empty — Phase 1)
├── output/                          # (empty — Phase 1)
├── utils/                           # (empty — Phase 1)
├── plan.md                          # Master execution plan
├── mvp.md                           # Original project guide
└── README.md
```

---

## What Phase 0 did NOT do (by design)

- **No data cleaning or normalisation** — raw text stays raw until Phase 1
- **No model training** — no features, no blocking, no predictions
- **No output files** — `matching_results.tsv` and `candidate_pairs.tsv` are Phase 1
- **No external dependencies** beyond pandas, numpy, scikit-learn

Phase 0 is purely: **validate → understand → measure → split**.
