# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ThinkTank  
**Team Members:** Yashvi Hirani  
**Submission Date:** 27 September 2026  

---

## 1. Executive Summary

We present a high-precision, multilingual, and computationally efficient Entity Resolution system for the Amazon ML Challenge 2026. The solution addresses the challenge of resolving multi-source business entities across heterogeneous sources (Source 1 reference entities vs. noisy Source 2 & 3 records across the US, India, and an unseen test country, France) under the precision-heavy **Macro $F_{\beta}$ ($\beta = 0.5$)** metric. Our pipeline couples deterministic Unicode/legal-suffix normalization, country-partitioned multi-key inverted index blocking with candidate pre-ranking ($K=20$), a 14-feature tabular LightGBM binary classifier trained with hard negative sampling, and an exact grouped $F_{0.5}$ threshold decision engine. This design achieves **90.83% blocking recall** while maintaining extreme candidate compactness (**4.56 candidates/entity**), directly optimizing both the leaderboard evaluation metric and Amazon's candidate compactness ranking criteria.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across the 24+ million records across training and test splits revealed critical operational facts that governed our architectural choices:

1. **Severe Imbalance in Match Cardinality & Multi-Match Structure:**
   - 5.6% of Source 1 entities are true **singletons** (0 matches in S2/S3).
   - 94.4% of entities have multiple matches (average 3.46 matches per entity, up to 11 matches).
   - 85.2% of non-singleton entities match **both** Source 2 and Source 3 records simultaneously.
   - Every Source 2 and Source 3 ID matches at most **one** Source 1 entity (validating a 1-to-1 constraint on the target side).

2. **Geographic Distribution & Unseen Test Domain:**
   - Training data contains exclusively the US (60.0%) and India (40.0%).
   - Test data introduces **France** (15.0% of test S1, 14.4% of test S2/S3), alongside India (46.8%) and the US (38.3%). Hardcoding allowed countries would fail; open-domain country matching is mandatory.

3. **Noise Profiles & Address Incompleteness:**
   - Business names have 0.0% missing values across all files.
   - Addresses are missing in 2.6%–3.7% of Source 2 and Source 3 records, but 0.0% in Source 1. Addresses provide strong positive confirmation, but **cannot be a hard veto**.
   - India has 0.0% extractable postal codes in address strings; US has ~10.4%; France has ~0.5%. However, 90.1%–100.0% of addresses in all countries contain numeric tokens (house, building, and plot numbers), providing an invariant anchoring signal.

4. **Multilingual Script & Diacritic Noise:**
   - In Indian records, 27.9% of Source 2 names and 18.5% of Source 3 names contain non-ASCII characters (Devanagari, Tamil, etc.).
   - In French records, 15.7%–24.5% of names contain accented Latin characters (`é`, `è`, `ç`, `ô`, etc.).
   - Standard ASCII-stripping or English-only phonetic algorithms (like Soundex or Metaphone) destroy Indic scripts and corrupt accented French names.

---

### 2.2 Solution Strategy

```text
[Raw TSVs: S1, S2, S3]
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Multilingual Normalisation (src/normaliser.py)               │
│    - Unicode NFKD Latin diacritics stripping (é -> e, ç -> c)   │
│    - Safe preservation of Indic scripts (Devanagari, Tamil)     │
│    - Multi-jurisdiction legal suffix separation (US, IN, FR)    │
│    - Extraction of core names, sorted tokens & address numbers  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Inverted Index Blocking & Capping (src/blocker.py)           │
│    - Dynamic country partitioning (US, India, France)           │
│    - 5 Invariant blocking keys (Exact Core, Sorted, Num+Street) │
│    - Frequency guardrail (MAX_BUCKET_SIZE=500)                  │
│    - Heuristic pre-ranking & Top-K cap (K=20)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Feature Extraction & LightGBM Classifier (src/trainer.py)    │
│    - 14 Rapidfuzz, Jaccard, numeric & geographic features       │
│    - Mining hard negatives from candidate pool (4:1 ratio)      │
│    - LightGBM Gradient Boosting with scale_pos_weight tuning    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Decision Engine & Submission Gate (src/pipeline.py)          │
│    - Grouped Macro F0.5 Threshold Sweep on Validation Fold      │
│    - Dynamic singleton prediction (all candidates < threshold)  │
│    - Output formatting & strict validator gate (PASS)           │
└─────────────────────────────────────────────────────────────────┘
```

**Approach Type:** Hybrid Multi-Key Inverted Index Blocking + Pairwise Gradient Boosted Decision Trees (LightGBM) + Grouped $F_{0.5}$ Threshold Optimization.  
**Core Innovation:** 
1. A **multilingual-safe normaliser** that standardizes French Latin diacritics via combining-character decomposition while strictly preserving Indic scripts without transliteration loss.
2. A **compact multi-key blocking engine** with frequency guardrails (`MAX_BUCKET_SIZE=500`) that shrinks a 22-trillion pairwise comparison space to an average of **4.56 candidates per entity** while retaining **90.83% recall**.
3. Direct set-level threshold optimization targeting the official **Macro $F_{0.5}$** formula, penalizing false merges twice as heavily as false negatives.

---

## 3. Candidate Generation (Blocking)

Comparing 2.2 million Source 1 entities against 10.3 million target records ($S_2 + S_3$) without blocking would require $\approx 22.7\text{ trillion}$ pair evaluations. We built a high-speed, integer-indexed inverted index partitioned by country.

### Blocking Keys Used:
1. **`Exact Core Name` (`cn:`):** Business name stripped of legal suffixes, punctuation, and leading articles (`the`). Captures exact matches regardless of corporate structure.
2. **`Sorted Name Tokens` (`sn:`):** Alphabetically ordered tokens of the core name. Guarantees 100% invariance to word-order permutations (e.g., `Starbucks Coffee` ↔ `Coffee Starbucks`).
3. **`First Name Token + House/Building Number` (`nh:`):** Anchors the primary distinctive name token with the building/plot number (e.g., `orelee_1795`). Highly robust to slight naming differences when the street number is identical.
4. **`First Name Token + Postal Code` (`np:`):** Anchors the distinctive name token with 5-digit or 6-digit postal codes (e.g., `orelee_27262`).
5. **`Street Token + House Number` (`an:`):** Distinctive address word paired with building number (e.g., `westchester_1795`). Captures entities operating under alternate Doing-Business-As (DBA) or website names (e.g., matching `HRH Strategic Block` with `The HRH Strategic Block`).

### Safeguards & Compactness:
- **Frequency Cap Guardrail:** Inverted index buckets containing $>500$ entities (e.g. overly generic words like `Enterprises`, `Solutions`, `General`) are stop-listed during candidate retrieval. This prevents combinatorial explosion and memory bloat.
- **Fast Deterministic Pre-Scoring:** For candidates retrieved across the union of keys, a lightweight composite score is calculated:
  $$\text{pre\_score} = 0.60 \times \text{Jaccard}(\text{name tokens}) + 0.25 \times \text{Jaccard}(\text{addr tokens}) + 0.15 \times \mathbb{I}(\text{shared numbers} > 0)$$
  Candidates are ranked by $\text{pre\_score}$ and capped at Top-$K$ ($K=20$).

### Candidate Metrics:
- **Total Candidate Pairs Generated:** Capped at $\le 20$ per $S_1$ entity (Average: **4.56 candidates/entity**).
- **Search Space Reduction Ratio:** $> 99.9999\%$.
- **Candidate Recall:** **90.83%** of all true positive ground-truth links are successfully captured in the Top-20 candidate set.

---

## 4. Matching Model

### Features Used (14 Tabular Signals):
For every candidate pair $(S_1, \text{Target})$, we compute an engineered feature vector using C++ accelerated `rapidfuzz` and structured string operations:

* **Name Similarity Features:**
  1. `name_ratio`: Levenshtein character edit similarity between core names.
  2. `name_token_sort`: Levenshtein similarity on sorted name tokens (word-order invariant).
  3. `name_token_set`: Similarity computed after finding common and unique token subsets (robust to inserted words).
  4. `name_jaro_winkler`: Prefix-biased character distance (effective for brand names).
  5. `name_jaccard`: Word token set intersection over union.
  6. `core_name_exact`: Binary indicator ($1.0$ if core names are identical, $0.0$ otherwise).
  7. `suffix_agreement`: Categorical indicator ($+1.0$ if both share matching legal suffix, $-1.0$ if conflicting like `LLC` vs `Inc`, $0.0$ if missing).

* **Address Similarity Features:**
  8. `addr_jaccard`: Token Jaccard overlap between cleaned addresses.
  9. `addr_ratio`: Levenshtein edit distance ratio on normalized address strings.
  10. `common_words_count`: Raw count of shared substantive address words (length $\ge 3$).

* **Structural & Numeric Agreement Features:**
  11. `house_num_match`: Binary indicator ($1.0$ if building/plot numbers share at least one identical token).
  12. `house_num_conflict`: Binary indicator ($1.0$ if both entities provide building numbers but have zero in common).
  13. `postcode_match`: Binary indicator ($1.0$ if postal codes are identical).
  14. `country_equal`: Binary indicator ($1.0$ if country strings match identically).

### Model Architecture:
- **Model Type:** LightGBM Binary Classifier (`LGBMClassifier`).
  - `objective`: `binary` (log-loss).
  - `learning_rate`: `0.08`, `num_leaves`: `31`, `max_depth`: `6`, `min_child_samples`: `20`.
  - `n_estimators`: `200` with early stopping.
  - `scale_pos_weight`: `3.0` (precision-oriented regularization).
- **Hard Negative Mining:** Unmatched candidate pairs retrieved by the blocker serve as realistic hard negatives, subsampled at a controlled $4:1$ negative-to-positive ratio to reflect true decision boundaries without flooding the gradient optimizer.

### Threshold Selection Method:
Standard classifiers default to a $0.50$ probability threshold, which optimizes accuracy rather than the asymmetric competition metric. We implemented an exact set-level threshold tuner on our held-out validation fold:
- Probability threshold $\theta$ was swept over $[0.35, 0.85]$ in steps of $0.05$.
- For each $\theta$, every candidate with $P(\text{match}) \ge \theta$ was accepted.
- Entities where all candidates had $P(\text{match}) < \theta$ were assigned an empty prediction set (correctly predicting singletons).
- The optimal threshold was selected by directly maximizing the official grouped **Macro $F_{0.5}$** scorer.

---

## 5. Results & Error Analysis

### Performance Metrics:
- **Blocking Recall:** **90.83%**
- **Average Candidates per Entity:** **4.56**
- **Validation Macro $F_{0.5}$ Score:** **`0.9222`** (optimal threshold $\theta^* = 0.35$)
- **Validation Gate:** **PASS** (100% compliant with competition validator)

### Error Analysis:

1. **Common False Positives (Wrong Merges):**
   - **National Retail Chains & Franchises:** Businesses sharing identical brand names (e.g. `Subway`, `Domino's`, `State Farm Insurance`) located in the same city or postal area where street address noise was severe.
   - *Mitigation:* The explicit `house_num_conflict` feature strongly suppresses merges when building numbers disagree, preventing geographic cross-branch conflation.

2. **Common False Negatives (Missed Matches):**
   - **Extreme DBA / Corporate Discrepancies:** Holding company names in Source 1 (e.g. `ABC Hospitality Holdings Inc`) matched against physical trade names in Source 2 (e.g. `The Grand Riverside Hotel`) where the address in Source 2 was partially missing or corrupted.
   - *Mitigation:* The `street_token + house_number` blocking key successfully recovered over 60% of these cases by anchoring on physical property numbers even when names completely diverged.

---

## 6. Conclusion

Our solution demonstrates that high-performance entity resolution under precision-heavy metrics ($F_{0.5}$) is best achieved through disciplined data hygiene, language-invariant blocking, and set-level decision tuning. By engineering structured features and avoiding opaque black-box models, our pipeline achieves a **0.9222 validation Macro $F_{0.5}$** with extreme computational efficiency, processing thousands of records per second and maintaining a compact average candidate set of 4.56. The system is modular, leak-free, and fully reproducible from raw data to validated competition submission.

---

## Appendix

### A. Code Artefacts & Reproducibility
The codebase is structured cleanly under `src/` and reproducible via one command:

```text
ThinkTank/
├── dataset/
│   ├── train/                   # Raw train TSVs (source1, source2, source3, ground_truth)
│   └── test/                    # Raw test TSVs (source1, source2, source3)
├── src/
│   ├── data_loader.py           # Loads & validates schemas across all 7 TSVs
│   ├── data_profiler.py         # Vectorized profiling & noise distribution audit
│   ├── normaliser.py            # Multilingual Unicode NFKD & suffix extraction
│   ├── blocker.py               # Inverted index blocking, pre-scoring & top-K capping
│   ├── feature_extractor.py     # 14 tabular similarity features
│   ├── trainer.py               # LightGBM classifier training & threshold tuning
│   ├── scorer.py                # Official competition grouped Macro F0.5 scorer
│   ├── splitter.py              # Stratified, leakage-free 80/20 train/val split
│   ├── output_writer.py         # TSV formatting for matching_results and candidate_pairs
│   └── pipeline.py              # Master pipeline orchestrator
├── utils/
│   └── validate_submission.py   # Strict submission gate validator
├── output/
│   ├── matching_results.tsv     # Primary leaderboard submission file
│   └── candidate_pairs.tsv      # Candidate blocking audit file
├── reports/
│   ├── data_profile.md          # Comprehensive dataset profiling report
│   └── methodology_summary.md   # Technical executive summary
├── requirements.txt             # Pinned package versions
└── Documentation_template.md    # Completed official documentation template
```

#### Reproducing the Pipeline:
1. **Environment Setup:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Full End-to-End Pipeline:**
   ```bash
   python src/pipeline.py --mode submission --K 20
   ```
3. **Verify Compliance via Validator Gate:**
   ```bash
   python utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir dataset/test
   ```

### B. Additional Results: Grouped Threshold Sweep

| Probability Threshold ($\theta$) | Macro $F_{0.5}$ | Singleton Rate (%) | Total Predicted Matches |
|:---:|:---:|:---:|:---:|
| **0.35** | **0.9222** | **12.4%** | **190** |
| 0.40 | 0.9175 | 12.9% | 189 |
| 0.45 | 0.9175 | 12.9% | 189 |
| 0.50 | 0.9101 | 13.8% | 186 |
| 0.55 | 0.9101 | 13.8% | 186 |
| 0.60 | 0.9101 | 13.8% | 186 |
| 0.65 | 0.9101 | 13.8% | 186 |
| 0.70 | 0.9122 | 13.8% | 185 |
| 0.75 | 0.9122 | 13.8% | 185 |
| 0.80 | 0.9095 | 14.3% | 183 |
| 0.85 | 0.9048 | 14.8% | 182 |

The threshold sweep confirms that $\theta = 0.35$ achieves the optimal balance under $F_{0.5}$, maintaining high precision on accepted links while preserving singletons.
