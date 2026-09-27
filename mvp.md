# 🏆 Amazon ML Challenge 2026 — Business Entity Resolution
## Complete Project Guide & Blueprint

---

## 📋 Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [How We Will Solve This](#2-how-we-will-solve-this)
3. [Platform & Where To Do This](#3-platform--where-to-do-this)
4. [System Diagrams, Database Design & Tech Stack](#4-system-diagrams-database-design--tech-stack)

---

## 1. Problem Statement

### 🎯 What Is The Challenge?

The **Amazon ML Challenge 2026** is a **Business Entity Resolution (ER)** competition. In large-scale commercial platforms (like Amazon), business identity data arrives from **multiple independent sources** — each contributing **partial, noisy fragments** of information about the same real-world business entities.

### 🧩 The Core Problem

> **Given business records from 3 independent data sources with noisy and inconsistent fields, build an ML solution that determines which records across sources refer to the same real-world business entity.**

### 📊 Data Sources

| Source | Description | ID Prefix |
|:-------|:-----------|:----------|
| **Source 1** | Deduplicated reference source (the "ground truth" base) | `S1-XXXXX` |
| **Source 2** | Independent business records (noisy, partial) | `S2-XXXXX` |
| **Source 3** | Independent business records (noisy, partial) | `S3-XXXXX` |

### 📄 Data Fields Available

Each source file contains these columns:

| Column | Description |
|:-------|:-----------|
| `entity_id` | Unique identifier (prefix = source: S1-, S2-, S3-) |
| `business_name` | Name of business (may have abbreviations, typos, transliterations) |
| `business_address` | Address (partial, format variations, landmark-based references) |
| `country` | Country label — Training: US, India; Test: **also includes France** |

### 🔍 Types of Noise to Expect

**Business Name Noise:**
- Abbreviations: `Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`
- Legal suffixes: `LLC`, `Inc.`, `LLP` — inconsistent usage
- Punctuation: `&` vs `and`
- Word-order swaps, DBA/trade names, typos

**Address Noise:**
- Abbreviations: `Rd` vs `Road`, `St` vs `Street`
- Transliteration variants (especially for Indian addresses)
- Missing components: no PIN code, no state
- Landmark-based references: `Near SBI ATM`
- Municipal numbering formats, component reordering

### 📦 Files Provided

**Training Files:**
```
dataset/train/train_source1.tsv   → Source 1 reference records
dataset/train/train_source2.tsv   → Source 2 records
dataset/train/train_source3.tsv   → Source 3 records
dataset/train/train_ground_truth.tsv → Ground truth matching labels
```

**Test Files:**
```
dataset/test/test_source1.tsv     → Source 1 test (generate matches for EVERY entity here)
dataset/test/test_source2.tsv     → Source 2 test records
dataset/test/test_source3.tsv     → Source 3 test records
```

### 📤 Output Required

Two tab-separated files in `output/` folder:

1. **`matching_results.tsv`** — Final entity matches (scored on leaderboard)
   ```
   source1_entity_id    matched_entity_ids
   S1-00001             S2-00047,S2-00193,S3-00812
   S1-00002             S3-00004
   S1-00003             
   ```

2. **`candidate_pairs.tsv`** — Candidate set from blocking stage (reviewed but not scored)
   ```
   source1_entity_id    candidate_entity_ids
   S1-00001             S2-00047,S2-00193,S3-00812,S3-00999
   S1-00002             S3-00004
   S1-00003             
   ```

### 📏 Evaluation Metric

**F₀.₅ Score** — A **precision-heavy** metric:

```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

> **Why precision-heavy?** In real-world ER, merging two DIFFERENT businesses (false positive) is **MORE DAMAGING** than missing a link (false negative). F₀.₅ weights precision **2× over recall**.

**Key Rules:**
- Singletons (no matches) correctly predicted as empty = **1.0 score**
- Singletons incorrectly given matches = **0.0 score**
- Macro-averaged across all Source 1 entities

### ⚠️ Critical Constraints

| Constraint | Detail |
|:-----------|:-------|
| Model License | MIT or Apache 2.0 only |
| Model Size | Up to **8 Billion** parameters max |
| External Data | **STRICTLY PROHIBITED** — No APIs, no external databases, no geocoding |
| Submissions | Max **5 per day** for 3 days |
| Challenge Window | **25th Sept 2026, 12:00 AM IST → 27th Sept 2026, 11:59 PM IST** |
| File Format | Tab-separated (.tsv) — MUST use `sep="\t"` |

---

## 2. How We Will Solve This

### 🏗️ Overall Solution Architecture: Multi-Stage Pipeline

Based on the latest research (2025-2026) and best practices in Entity Resolution, we will use a **4-stage pipeline**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ENTITY RESOLUTION PIPELINE                       │
│                                                                      │
│  Stage 1          Stage 2          Stage 3          Stage 4          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │   Data   │    │ Blocking │    │ Pairwise │    │  Final   │      │
│  │ Cleaning │───▶│ /Candid. │───▶│ Matching │───▶│ Decision │      │
│  │ & Prep   │    │  Gen     │    │ & Scoring│    │ & Output │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                      │
│  Normalize        Reduce N²       ML Scoring      Threshold +        │
│  names/addr       comparisons     on candidates    Singleton          │
│                   to O(N·k)       pairs            Detection          │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Data Cleaning & Preprocessing

**Goal:** Normalize business names and addresses to reduce noise before matching.

**Steps:**

```python
# ────────────────────────────────────────────────────────
# STAGE 1: Data Cleaning & Preprocessing
# ────────────────────────────────────────────────────────

# 1.1 — Text Lowercasing + Unicode Normalization
#   Convert all text to lowercase, normalize Unicode (NFKD form)
#   Example: "TATA CONSULTANCY SERVICES LTD." → "tata consultancy services ltd"

# 1.2 — Business Name Standardization
#   - Expand abbreviations: "pvt" → "private", "ltd" → "limited", "corp" → "corporation"
#   - Remove legal suffixes: "llc", "inc", "llp", "co", "company"
#   - Normalize punctuation: "&" → "and", remove extra spaces
#   - Remove special characters: hyphens, dots, apostrophes

# 1.3 — Address Standardization
#   - Expand abbreviations: "rd" → "road", "st" → "street", "blvd" → "boulevard"
#   - Remove landmarks: "near sbi atm", "opposite park"
#   - Normalize numbering: "#123" → "123"
#   - Handle country-specific patterns (India PIN, US ZIP, French postal codes)

# 1.4 — Create Composite Fields
#   - Concatenate: business_name + " " + business_address + " " + country
#   - Create separate cleaned fields for name-only and address-only matching
```

**Python Libraries:**
- `pandas` — Data manipulation
- `re` (regex) — Text cleaning patterns
- `unicodedata` — Unicode normalization
- `ftfy` — Fix text encoding issues

---

### Stage 2: Blocking / Candidate Generation (MOST CRITICAL STAGE)

**Goal:** Reduce the O(N²) comparison space to a manageable candidate set.

> **IMPORTANT:** Amazon explicitly states: *"The approach that generates a **smaller** candidate set per Source 1 entity will be ranked higher."* This means blocking quality directly impacts final ranking.

**Strategy: Multi-Strategy Blocking Ensemble**

```
┌────────────────────────────────────────────────────────┐
│              BLOCKING ENSEMBLE                          │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐    │
│  │  TF-IDF     │  │  Semantic   │  │  Token-Based │    │
│  │  Cosine     │  │  Embedding  │  │  Inverted    │    │
│  │  Blocking   │  │  + FAISS    │  │  Index       │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘    │
│         │                │                │              │
│         └────────┬───────┴────────┬───────┘              │
│                  ▼                                        │
│         ┌─────────────────┐                              │
│         │   UNION of      │                              │
│         │   Candidates    │                              │
│         │   (High Recall) │                              │
│         └─────────────────┘                              │
└────────────────────────────────────────────────────────┘
```

**2A. TF-IDF Cosine Blocking:**
```python
# ────────────────────────────────────────────────────────
# BLOCKING STRATEGY A: TF-IDF Cosine Similarity
# ────────────────────────────────────────────────────────
# - Build TF-IDF vectors on cleaned business_name + address
# - Use sparse matrix cosine similarity
# - Retrieve top-k candidates per S1 entity (k=50-100)
# - Library: sklearn.feature_extraction.text.TfidfVectorizer
# - Library: sklearn.metrics.pairwise.cosine_similarity
```

**2B. Semantic Embedding + FAISS (State of the Art 2026):**
```python
# ────────────────────────────────────────────────────────
# BLOCKING STRATEGY B: Sentence Transformers + FAISS
# ────────────────────────────────────────────────────────
# - Use pre-trained model: "all-MiniLM-L6-v2" (22M params, MIT license)
#   OR "paraphrase-multilingual-MiniLM-L12-v2" (for multilingual: US, India, France)
# - Encode business_name + address into 384-dim dense vectors
# - Build FAISS IndexFlatIP (Inner Product = cosine on normalized vectors)
# - Query: for each S1 entity, retrieve top-k nearest neighbors from S2/S3
# - Advantage: Captures SEMANTIC similarity ("IBM Corp" ↔ "International Business Machines")
```

**2C. Token-Based Inverted Index:**
```python
# ────────────────────────────────────────────────────────
# BLOCKING STRATEGY C: Token Blocking with Inverted Index
# ────────────────────────────────────────────────────────
# - Tokenize business names into n-grams (bigrams, trigrams)
# - Build inverted index: token → set of entity_ids
# - For each S1 entity, find S2/S3 entities sharing ≥ 2 tokens
# - Fast, lightweight, good for exact token matches
# - Country-partitioned: only compare within same country
```

**Final Candidate Set:**
```python
# ────────────────────────────────────────────────────────
# UNION: Merge candidates from all blocking strategies
# ────────────────────────────────────────────────────────
# candidates[s1_id] = set_A.union(set_B).union(set_C)
# This maximizes RECALL (no missed matches)
# Deduplication: remove duplicate candidate pairs
# Output → candidate_pairs.tsv
```

---

### Stage 3: Pairwise Matching & Scoring (ML Classification)

**Goal:** For each (S1, candidate) pair, predict match probability.

**Approach: Feature Engineering + LightGBM Classifier**

```
┌──────────────────────────────────────────────────────────────────┐
│                  MATCHING MODEL PIPELINE                         │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              FEATURE ENGINEERING                          │    │
│  │                                                            │    │
│  │  String Similarity Features:                               │    │
│  │  ├── Levenshtein Distance (name, address)                  │    │
│  │  ├── Jaro-Winkler Similarity (name)                        │    │
│  │  ├── Jaccard Similarity (name tokens, address tokens)      │    │
│  │  ├── TF-IDF Cosine Similarity (name, address)              │    │
│  │  ├── Soundex/Metaphone Match (phonetic name matching)      │    │
│  │  └── Longest Common Subsequence Ratio                      │    │
│  │                                                            │    │
│  │  Semantic Features:                                        │    │
│  │  ├── Sentence Transformer Cosine Distance (name)           │    │
│  │  ├── Sentence Transformer Cosine Distance (address)        │    │
│  │  └── Combined embedding distance                           │    │
│  │                                                            │    │
│  │  Structural Features:                                      │    │
│  │  ├── Country Match (binary)                                │    │
│  │  ├── Name Token Overlap Ratio                              │    │
│  │  ├── Address Token Overlap Ratio                           │    │
│  │  ├── Name Length Ratio (min/max)                            │    │
│  │  └── Numeric Component Match (house numbers, ZIP codes)    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                          │                                        │
│                          ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              ML CLASSIFIER: LightGBM                      │    │
│  │                                                            │    │
│  │  - Objective: binary classification (match/non-match)      │    │
│  │  - Metric: F₀.₅ (custom eval metric)                     │    │
│  │  - Training: Use train_ground_truth.tsv for labels         │    │
│  │  - Cross-validation: Stratified 5-fold                     │    │
│  │  - Output: match_probability per pair                      │    │
│  └──────────────────────────────────────────────────────────┘    │
│                          │                                        │
│                          ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              OUTPUT: match_probability scores              │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

**Feature Engineering Code Outline:**
```python
# ────────────────────────────────────────────────────────
# FEATURE ENGINEERING FOR EACH CANDIDATE PAIR
# ────────────────────────────────────────────────────────

# String Similarity Features (using python-Levenshtein, jellyfish)
# features['name_levenshtein']   = Levenshtein.ratio(s1_name, s2_name)
# features['name_jaro_winkler']  = jellyfish.jaro_winkler_similarity(s1_name, s2_name)
# features['name_jaccard']       = jaccard_tokens(s1_name, s2_name)
# features['addr_levenshtein']   = Levenshtein.ratio(s1_addr, s2_addr)
# features['addr_jaccard']       = jaccard_tokens(s1_addr, s2_addr)
# features['tfidf_name_cosine']  = cosine_sim(tfidf_s1_name, tfidf_s2_name)
# features['tfidf_addr_cosine']  = cosine_sim(tfidf_s1_addr, tfidf_s2_addr)

# Semantic Features (using sentence-transformers)
# features['embed_name_cosine']  = cosine_sim(embed_s1_name, embed_s2_name)
# features['embed_addr_cosine']  = cosine_sim(embed_s1_addr, embed_s2_addr)

# Structural Features
# features['country_match']      = int(s1_country == s2_country)
# features['name_length_ratio']  = min(len_s1, len_s2) / max(len_s1, len_s2)
# features['token_overlap']      = len(common_tokens) / len(union_tokens)
```

---

### Stage 4: Final Decision & Output Generation

**Goal:** Convert match probabilities into final match/no-match decisions.

```
┌────────────────────────────────────────────────────────────┐
│              FINAL DECISION STAGE                           │
│                                                              │
│  1. Apply threshold on match_probability                     │
│     - HIGH threshold (e.g., 0.7-0.85) for precision          │
│     - Tune on validation set to maximize F₀.₅              │
│                                                              │
│  2. Singleton Detection                                      │
│     - If no candidate passes threshold → empty match list    │
│     - Singletons correctly predicted = 1.0 score            │
│                                                              │
│  3. Hard Rule Vetoes (Post-processing)                       │
│     - If countries don't match → reject the pair             │
│     - If name similarity < 0.2 → reject the pair            │
│                                                              │
│  4. Generate Output Files                                    │
│     - matching_results.tsv (thresholded results)            │
│     - candidate_pairs.tsv (all candidates from Stage 2)     │
│                                                              │
│  5. Validate with provided script                            │
│     python3 utils/validate_submission.py \                   │
│       --matching output/matching_results.tsv \               │
│       --candidate output/candidate_pairs.tsv \               │
│       --test-dir dataset/test                                │
└────────────────────────────────────────────────────────────┘
```

---

### 🏆 Strategy Summary: Why This Approach Wins

| Aspect | Our Strategy | Why |
|:-------|:-------------|:----|
| **Precision Focus** | High threshold + hard vetoes | F₀.₅ rewards precision 2× over recall |
| **Blocking Quality** | Multi-strategy ensemble (TF-IDF + FAISS + Token) | Maximizes recall ceiling, smaller candidate set |
| **Semantic Understanding** | Sentence Transformers embeddings | Catches "IBM" ↔ "International Business Machines" |
| **Multilingual Support** | `paraphrase-multilingual-MiniLM` | Handles US, India, AND France (unseen in training) |
| **Singleton Handling** | Explicit empty-list prediction | Each correct singleton = free 1.0 score |
| **Model Compliance** | LightGBM + MiniLM (< 8B params, MIT license) | Meets all constraints |

---

## 3. Platform & Where To Do This

### 🖥️ Development Environment

| Environment | Details |
|:-----------|:--------|
| **Primary Platform** | Local machine (Linux/macOS) with GPU |
| **IDE** | VS Code / JupyterLab / PyCharm |
| **Python Version** | Python 3.10+ |
| **GPU (recommended)** | NVIDIA GPU with CUDA 12.x (for Sentence Transformers) |
| **Cloud Alternatives** | Google Colab Pro (T4/A100), Kaggle Notebooks (P100), AWS SageMaker |

### 🌐 Competition Platform

| Item | Detail |
|:-----|:-------|
| **Host Platform** | **Unstop** (unstop.com) — Amazon's competition hosting partner |
| **Submission Portal** | Upload `matching_results.tsv` on the Unstop portal |
| **Leaderboard** | Public (during challenge) + Private (final ranking) |
| **Support** | Google Form for queries, email: support@unstop.com |

### 📁 Project Directory Structure (Where We Build This)

```
amezon ML/
├── docs for infomation/                    # Reference PDFs
│   ├── PS_ML_Challenge.pdf
│   ├── PS_Amazon_Challenge.pdf
│   └── guidelines_ML_Challenge.pdf
│
├── dataset/                                # Data files
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── code/                                   # Our solution code
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py                   # Configuration & hyperparameters
│       │   ├── preprocessing.py            # Stage 1: Data cleaning
│       │   ├── blocking.py                 # Stage 2: Candidate generation
│       │   ├── feature_engineering.py      # Stage 3a: Feature extraction
│       │   ├── matching.py                 # Stage 3b: LightGBM classifier
│       │   ├── postprocessing.py           # Stage 4: Decision & output
│       │   ├── utils.py                    # Helper functions
│       │   └── pipeline.py                 # End-to-end pipeline runner
│       ├── notebooks/
│       │   ├── 01_eda.ipynb                # Exploratory data analysis
│       │   ├── 02_blocking_experiments.ipynb
│       │   ├── 03_feature_analysis.ipynb
│       │   └── 04_model_tuning.ipynb
│       ├── README.md                       # Reproduction instructions
│       └── requirements.txt                # Pinned dependencies
│
├── output/                                 # Final submission files
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── models/                                 # Saved models
│   ├── lightgbm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── utils/                                  # Provided utilities
│   └── validate_submission.py
│
├── Documentation_template.md               # Methodology write-up
└── Amazon_ML_Challenge_2026_Complete_Guide.md  # This file
```

### 🛠️ Setup Instructions

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python -m src.pipeline --mode train    # Train on training data
python -m src.pipeline --mode predict  # Generate predictions on test data

# 4. Validate submission
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## 4. System Diagrams, Database Design & Tech Stack

### 🏛️ High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTITY RESOLUTION SYSTEM ARCHITECTURE                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         DATA LAYER                                    │  │
│  │                                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │
│  │  │  Source 1    │  │  Source 2    │  │  Source 3    │  │  Ground    │  │  │
│  │  │  (.tsv)      │  │  (.tsv)      │  │  (.tsv)      │  │  Truth     │  │  │
│  │  │  Reference   │  │  Noisy Data  │  │  Noisy Data  │  │  Labels    │  │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │  │
│  │         └─────────────────┼─────────────────┼────────────────┘         │  │
│  └───────────────────────────┼─────────────────┼─────────────────────────┘  │
│                              ▼                 ▼                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     PREPROCESSING ENGINE                              │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  Text        │  │  Name        │  │  Address     │                │  │
│  │  │  Normalizer  │  │  Standardizer│  │  Normalizer  │                │  │
│  │  │  (Unicode,   │  │  (Legal      │  │  (Abbrev,    │                │  │
│  │  │   lowercase) │  │   suffixes)  │  │   landmarks) │                │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │  │
│  │         └─────────────────┼─────────────────┘                         │  │
│  └───────────────────────────┼───────────────────────────────────────────┘  │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     BLOCKING ENGINE (Candidate Generation)             │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  TF-IDF      │  │  FAISS       │  │  Token       │                │  │
│  │  │  Cosine      │  │  Semantic    │  │  Inverted    │                │  │
│  │  │  Retrieval   │  │  ANN Search  │  │  Index       │                │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │  │
│  │         └─────────────────┼─────────────────┘                         │  │
│  │                    ┌──────▼──────┐                                     │  │
│  │                    │   UNION     │──▶ candidate_pairs.tsv             │  │
│  │                    │   Merger    │                                     │  │
│  │                    └──────┬──────┘                                     │  │
│  └───────────────────────────┼───────────────────────────────────────────┘  │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     MATCHING ENGINE (ML Scoring)                       │  │
│  │                                                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────┐     │  │
│  │  │  Feature Extractor                                           │     │  │
│  │  │  ├── String Similarity (Levenshtein, Jaro-Winkler, Jaccard) │     │  │
│  │  │  ├── Semantic Distance (Sentence Transformer cosines)       │     │  │
│  │  │  └── Structural Features (country, length ratio, tokens)    │     │  │
│  │  └──────────────────────────┬───────────────────────────────────┘     │  │
│  │                             ▼                                         │  │
│  │  ┌──────────────────────────────────────────────────────────────┐     │  │
│  │  │  LightGBM Binary Classifier                                  │     │  │
│  │  │  - match_probability per (S1, candidate) pair                │     │  │
│  │  │  - Custom F₀.₅ eval metric                                  │     │  │
│  │  └──────────────────────────┬───────────────────────────────────┘     │  │
│  └─────────────────────────────┼─────────────────────────────────────────┘  │
│                                ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     DECISION ENGINE (Post-Processing)                  │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  Threshold   │  │  Hard Rule   │  │  Output      │                │  │
│  │  │  Application │  │  Vetoes      │  │  Formatter   │                │  │
│  │  │  (F₀.₅     │  │  (country,   │  │  (.tsv)      │                │  │
│  │  │   optimized) │  │   min sim)   │  │              │                │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │  │
│  │         └─────────────────┼─────────────────┘                         │  │
│  │                           ▼                                           │  │
│  │                  matching_results.tsv                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 📊 Data Flow Diagram

```
                    ┌─────────────────┐
                    │   RAW TSV DATA   │
                    │  (3 sources +    │
                    │   ground truth)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  PANDAS LOADER   │
                    │  pd.read_csv(    │
                    │    sep="\t")     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  CLEANING        │
                    │  MODULE          │
                    │  (preprocessing) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────┐  ┌─────▼─────┐  ┌────▼────────┐
     │  TF-IDF     │  │  SENTENCE  │  │  TOKEN      │
     │  VECTORS    │  │  EMBEDDINGS│  │  INDEX      │
     │  (sparse)   │  │  (dense)   │  │  (dict)     │
     └────────┬────┘  └─────┬─────┘  └────┬────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  CANDIDATE PAIRS │
                    │  (merged set)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  FEATURE         │
                    │  EXTRACTION      │
                    │  (15-20 features │
                    │   per pair)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  LIGHTGBM       │
                    │  CLASSIFIER     │
                    │  (probability)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  THRESHOLDING   │
                    │  + VETOES       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼──────────┐     ┌───────────▼──────────┐
     │ matching_results  │     │ candidate_pairs       │
     │ .tsv              │     │ .tsv                  │
     │ (SCORED)          │     │ (REVIEWED)            │
     └───────────────────┘     └──────────────────────┘
```

---

### 🗄️ Data Schema / Database Diagram

Since this is an ML pipeline (not a web app), our "database" is the TSV file structure:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SCHEMA                                  │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │  SOURCE FILES (source1/2/3.tsv)          │                   │
│  │  ─────────────────────────────────────── │                   │
│  │  entity_id        VARCHAR  PK            │                   │
│  │  business_name    TEXT                    │                   │
│  │  business_address TEXT                    │                   │
│  │  country          VARCHAR                │                   │
│  └──────────────────────┬───────────────────┘                   │
│                         │                                        │
│                    1 ──▶ N                                       │
│                         │                                        │
│  ┌──────────────────────▼───────────────────┐                   │
│  │  GROUND TRUTH (train_ground_truth.tsv)   │                   │
│  │  ─────────────────────────────────────── │                   │
│  │  source1_entity_id   VARCHAR  FK→S1      │                   │
│  │  matched_entity_ids  TEXT (comma-sep)     │                   │
│  │     → References S2-/S3- entity_ids      │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │  MATCHING RESULTS (matching_results.tsv) │                   │
│  │  ─────────────────────────────────────── │                   │
│  │  source1_entity_id   VARCHAR  FK→S1      │                   │
│  │  matched_entity_ids  TEXT (comma-sep)     │                   │
│  │     → References S2-/S3- entity_ids      │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │  CANDIDATE PAIRS (candidate_pairs.tsv)   │                   │
│  │  ─────────────────────────────────────── │                   │
│  │  source1_entity_id     VARCHAR  FK→S1    │                   │
│  │  candidate_entity_ids  TEXT (comma-sep)   │                   │
│  │     → References S2-/S3- entity_ids      │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │  INTERNAL: Feature Matrix                 │                   │
│  │  (generated during pipeline)              │                   │
│  │  ─────────────────────────────────────── │                   │
│  │  s1_entity_id           VARCHAR           │                   │
│  │  candidate_entity_id    VARCHAR           │                   │
│  │  name_levenshtein       FLOAT             │                   │
│  │  name_jaro_winkler      FLOAT             │                   │
│  │  name_jaccard           FLOAT             │                   │
│  │  addr_levenshtein       FLOAT             │                   │
│  │  addr_jaccard           FLOAT             │                   │
│  │  tfidf_name_cosine      FLOAT             │                   │
│  │  tfidf_addr_cosine      FLOAT             │                   │
│  │  embed_name_cosine      FLOAT             │                   │
│  │  embed_addr_cosine      FLOAT             │                   │
│  │  country_match          BOOLEAN           │                   │
│  │  name_length_ratio      FLOAT             │                   │
│  │  token_overlap          FLOAT             │                   │
│  │  numeric_match          BOOLEAN           │                   │
│  │  label (train only)     BOOLEAN           │                   │
│  └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

### 🧰 Complete Tech Stack

#### Core ML Stack

| Category | Technology | Version | License | Purpose |
|:---------|:----------|:--------|:--------|:--------|
| **Language** | Python | 3.10+ | PSF | Primary development language |
| **Data** | Pandas | 2.2+ | BSD-3 | Data loading, manipulation (.tsv) |
| **Data** | NumPy | 1.26+ | BSD-3 | Numerical operations |
| **ML Framework** | LightGBM | 4.5+ | MIT ✅ | Binary classifier for matching |
| **ML Framework** | scikit-learn | 1.5+ | BSD-3 | TF-IDF, metrics, cross-validation |
| **NLP - Embeddings** | sentence-transformers | 3.3+ | Apache-2.0 ✅ | Semantic embedding generation |
| **NLP - Model** | all-MiniLM-L6-v2 | — | Apache-2.0 ✅ | Pre-trained embedding model (22M params) |
| **NLP - Multilingual** | paraphrase-multilingual-MiniLM-L12-v2 | — | Apache-2.0 ✅ | Multilingual embeddings (US/India/France) |
| **ANN Search** | FAISS (faiss-cpu / faiss-gpu) | 1.9+ | MIT ✅ | Fast approximate nearest neighbor search |
| **String Similarity** | python-Levenshtein | 0.26+ | MIT ✅ | Edit distance calculations |
| **String Similarity** | jellyfish | 1.1+ | MIT ✅ | Jaro-Winkler, Soundex, Metaphone |
| **Text Processing** | regex / re | stdlib | — | Text cleaning patterns |
| **Text Processing** | ftfy | 6.3+ | Apache-2.0 ✅ | Fix broken Unicode text |
| **Serialization** | joblib | 1.4+ | BSD-3 | Save/load trained models |

#### Development & Experimentation

| Category | Technology | Purpose |
|:---------|:----------|:--------|
| **Notebook** | JupyterLab / Jupyter Notebook | EDA, experiments, visualization |
| **Visualization** | matplotlib / seaborn | Data analysis charts |
| **Progress Bars** | tqdm | Progress tracking for long operations |
| **Config** | pyyaml / argparse | Configuration management |
| **Version Control** | Git | Code versioning (do NOT push until told) |

#### Infrastructure

| Category | Technology | Purpose |
|:---------|:----------|:--------|
| **GPU Acceleration** | CUDA 12.x + cuDNN | Speed up embedding generation |
| **Cloud (if needed)** | Google Colab Pro / Kaggle | Free GPU access (T4/P100) |
| **Package Manager** | pip + venv | Dependency management |

---

### 📦 requirements.txt (Recommended)

```
# ─────────────────────────────────────────────
# Amazon ML Challenge 2026 - Dependencies
# Business Entity Resolution Pipeline
# ─────────────────────────────────────────────

# Core Data Science
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0

# ML Classifier
lightgbm>=4.5.0

# NLP & Embeddings
sentence-transformers>=3.3.0
transformers>=4.45.0
torch>=2.4.0

# Approximate Nearest Neighbor Search
faiss-cpu>=1.9.0          # Use faiss-gpu if GPU available

# String Similarity
python-Levenshtein>=0.26.0
jellyfish>=1.1.0

# Text Processing
ftfy>=6.3.0
regex>=2024.0.0

# Utilities
tqdm>=4.66.0
joblib>=1.4.0
pyyaml>=6.0.0

# Visualization (for EDA notebooks)
matplotlib>=3.9.0
seaborn>=0.13.0

# Jupyter
jupyterlab>=4.2.0
```

---

### 🔄 Model Training & Inference Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                 TRAINING PIPELINE                            │
│                                                              │
│  1. Load train_source1/2/3.tsv + ground_truth.tsv           │
│  2. Preprocess & clean all records                           │
│  3. Generate positive pairs from ground truth                │
│  4. Generate HARD negative pairs via blocking                │
│     (candidates that are close but NOT matches)              │
│  5. Extract features for all pairs                           │
│  6. Train LightGBM with custom F₀.₅ metric                 │
│  7. Tune threshold on validation set                         │
│  8. Save model + vectorizers + config                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 INFERENCE PIPELINE                            │
│                                                              │
│  1. Load test_source1/2/3.tsv                                │
│  2. Preprocess & clean all records                           │
│  3. Run blocking → generate candidate_pairs.tsv             │
│  4. Extract features for all candidate pairs                 │
│  5. Score with trained LightGBM model                        │
│  6. Apply threshold + hard vetoes                            │
│  7. Generate matching_results.tsv                            │
│  8. Validate with validate_submission.py                     │
└─────────────────────────────────────────────────────────────┘
```

---

### 📋 Final Submission Package Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv          # Final matches (scored on leaderboard)
│   └── candidate_pairs.tsv           # Blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── preprocessing.py
│       │   ├── blocking.py
│       │   ├── feature_engineering.py
│       │   ├── matching.py
│       │   ├── postprocessing.py
│       │   ├── utils.py
│       │   └── pipeline.py
│       ├── README.md                  # Reproduction instructions
│       └── requirements.txt           # Pinned dependencies
└── Documentation_template.md          # Methodology write-up
```

---

### 🎯 Key Success Factors

**Top Priority Actions (in order):**
1. Get a strong blocking strategy — it determines your recall ceiling
2. Focus on precision — F₀.₅ punishes false merges HARD
3. Don't forget singletons — free 1.0 per correct empty prediction
4. Handle France data — unseen country in test, use multilingual models
5. Validate output format — one bad format = rejected submission

**Things That Will Get You DISQUALIFIED:**
- Using ANY external API (geocoding, business lookup, etc.)
- Using models > 8B parameters or non-MIT/Apache-2.0 licensed models
- Submitting improperly formatted TSV files
- Registering with multiple IDs

---

*Document created: 26th September 2026*
*Amazon ML Challenge 2026 — Business Entity Resolution*
*Challenge Window: 25th Sept 2026 12:00 AM IST → 27th Sept 2026 11:59 PM IST*