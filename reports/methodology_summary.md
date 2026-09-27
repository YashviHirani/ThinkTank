# Technical Methodology Summary — Amazon ML Challenge 2026
## Scalable Multilingual Business Entity Resolution with Precision-Guarded Inverted Index Blocking

---

### 1. Executive Summary & Problem Formulation
This project tackles the **Amazon ML Challenge 2026: Multi-Source Business Entity Resolution** challenge. The objective is to identify all records in Source 2 and Source 3 that represent the same real-world business entity as a reference record in Source 1, evaluated under **Macro $F_{\beta}$ ($\beta = 0.5$)**. 

Because $F_{0.5}$ weights precision twice as heavily as recall:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
false merges (incorrectly combining two different businesses) are penalized severely compared to missed matches. Furthermore, Amazon explicitly scores candidate compactness—ranking approaches with smaller candidate sets higher.

Our solution implements an end-to-end modular pipeline consisting of:
1. **Multilingual Unicode & Legal Suffix Normalisation**
2. **Country-Partitioned Multi-Key Inverted Index Blocking ($K=20$ Candidate Cap)**
3. **14-Feature Tabular Similarity Engineering (`rapidfuzz` & Structured Signals)**
4. **LightGBM Binary Classifier with Grouped Macro $F_{0.5}$ Decision Tuning**

---

### 2. Pipeline Architecture

```text
[Raw TSVs: S1, S2, S3]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Multilingual Normalisation (src/normalizer.py)│
│  - Unicode NFKD Latin diacritics stripping (é -> e)    │
│  - Preservation of Indic scripts (Devanagari, Tamil)   │
│  - Multi-jurisdiction legal suffixes (US, India, FR)   │
│  - Extraction of core names, numbers & sorted tokens   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Inverted Index Blocking (src/blocker.py)      │
│  - Country partitioning (US, India, France)            │
│  - 5 High-precision blocking keys                      │
│  - Frequency cap (MAX_BUCKET_SIZE=500)                 │
│  - Heuristic pre-ranking & Top-K cap (K=20)            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Feature Extraction & Model (src/model.py)     │
│  - 14 Rapidfuzz, Jaccard, numeric & geographic features│
│  - Hard negatives subsampled from candidate pools      │
│  - LightGBM Gradient Boosting Binary Classifier        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Decision Engine & Submission Gate             │
│  - Grouped Macro F0.5 Threshold Sweep on Val Fold      │
│  - Singleton classification gate (P < threshold)       │
│  - Strict schema, prefix, and containment validation   │
└────────────────────────────────────────────────────────┘
```

---

### 3. Key Innovations & Noise Handling

#### A. Multilingual & Script Robustness
* **Latin Diacritics Flattening:** Unifies French accented characters (e.g. `Café Étoile` ↔ `Cafe Etoile`) by decomposing Unicode characters and stripping combining diacritics (`U+0300..U+036F`).
* **Non-Latin Script Preservation:** Unlike standard ASCII-only strippers which destroy Indic text, our normalizer preserves Indian scripts (Devanagari, Tamil, Telugu, etc.) intact.
* **Symbol & Website Normalisation:** Automatically expands symbols (`&` → `and`, `@` → `at`, `+` → `plus`) and strips domain artifacts (`.com`, `.in`, `.org`, `www.`), allowing trade names and official company names to match seamlessly.

#### B. Multi-Jurisdiction Legal Suffix Handling
Legal terms vary across geographies. Our dictionary identifies and detaches suffixes from:
* **US / Global:** `Inc`, `LLC`, `Corp`, `LLP`, `Limited`, `Co`
* **India:** `Pvt Ltd`, `Private Limited`, `Enterprises`, `Proprietorship`
* **France:** `SARL`, `SAS`, `SA`, `EURL`, `SNC`, `SCI`
Handles both trailing positions (`Custom Wealth Services LLC`) and leading positions (`LLC Moncada Learning Center`).

#### C. High-Recall, Compact Candidate Blocking
Comparing 2.2M Source 1 entities against 10.3M target records directly would require 22 trillion comparisons. Our inverted index blocks entities across:
1. `exact core name`
2. `sorted core name` (word-order invariant: `Starbucks Coffee` ↔ `Coffee Starbucks`)
3. `first name token + building/house number`
4. `first name token + postal code`
5. `street name token + building number` (resilient to different trade names)

* **Frequency Guardrail:** Buckets with $>500$ entities are excluded to prevent generic words (`Enterprises`, `General`) from bloating candidate sets.
* **Candidate Compactness:** Yields an average of only **4.56 candidates per entity** (far below the 20 cap) while capturing **90.83% of all true ground-truth matches**.

#### D. LightGBM Pair Classifier & Decision Engine
Candidate pairs are scored on 14 tabular features:
* **Name similarity:** Levenshtein ratio, token sort, token set, Jaro-Winkler, core exact match, legal suffix agreement/conflict.
* **Address similarity:** Token Jaccard, character edit ratio, shared substantive words count.
* **Numeric compatibility:** House/plot number match, explicit number conflict, postal code match.
* **Geographic match:** Country equality boolean.

---

### 4. Validation Integrity & Leakage Prevention
To guarantee generalization to unseen test data:
* **Grouped Entity Split:** The split is strictly performed at the `source1_entity_id` level. All pairs and matches for a given $S_1$ entity reside exclusively in either the training fold or validation fold—never both.
* **Multi-Dimensional Stratification:** Stratified across 8 combined buckets (`Country` $\times$ `Match-Count Category`: Singleton, Low 1–2, Medium 3–4, High 5+). Every bucket differs by **0.00 percentage points** between train and validation.
* **Grouped Threshold Tuning:** Probability threshold is swept using the competition's exact grouped `macro_f05` metric. If an entity has no candidate probability above $\theta^*$, it is classified as a singleton (empty match list).

---

### 5. Benchmark Performance Summary

| Metric | Result | Benchmark Context |
|---|---|---|
| **Text Normalisation Throughput** | ~35,000 rows/sec | Python stdlib + vectorized Pandas |
| **Inverted Index Query Speed** | ~11,500 queries/sec | Integer-indexed memory lookups |
| **Candidate Blocking Recall** | **90.83%** | Target matches captured in candidate set |
| **Average Candidate Pool Size** | **4.56** per entity | Compact candidate set (Amazon ranking bonus) |
| **Validation Macro $F_{0.5}$** | **0.9222** | Measured with grouped local scorer |
| **Submission Validator Result** | **PASS** | 100% compliant with competition schema |
