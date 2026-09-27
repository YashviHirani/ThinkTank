# Amazon ML Challenge 2026 — Practical Execution Plan

## Purpose and decision

This is the single working plan for the Business Entity Resolution challenge. It consolidates the supplied problem statements, guidelines, original guide, improved blueprint, and review notes. It is designed to produce a valid submission early, then improve it using measured experiments.

**Current repository status (reviewed 26 September 2026):** this workspace contains reference documents only. There is no `dataset/`, pipeline code, output directory, or version-control repository yet. Start at Phase 0.

The main decision is to use a **precision-oriented lexical and structured baseline first**:

```text
TSV data → profile/normalise → multi-key blocking → pre-rank and cap candidates
        → pair features → LightGBM → grouped F0.5 tuning → outputs → validation
```

Sentence-transformer/FAISS embeddings are optional experiments after this baseline works and improves local validation. They are not a prerequisite for a submission.

---

## 1. Facts that govern the solution

| Area | Required interpretation |
|---|---|
| Matching task | For every Source 1 (`S1-`) entity, find zero, one, or many matching Source 2/3 records. Do **not** assume one match per S1. |
| Training/test countries | Training has US and India; test additionally has France. Treat `country` as an open text field—never hard-code an allowed-country list. |
| Score | Macro F0.5 is computed per S1 entity. Precision matters twice as much as recall. A true singleton is worth 1.0 only when its predicted list is empty. |
| Leaderboard output | `matching_results.tsv`, tab-separated, with exactly one row for every test S1 entity. |
| Blocking audit | `candidate_pairs.tsv` is the final candidate set actually scored by the model, not an earlier raw retrieval set. Every predicted match must be in it. Small, high-recall candidate sets help final review. |
| Integrity | Use only supplied data and compliant offline libraries/models. No lookups, geocoding, business databases, web augmentation, or entity-resolution APIs. |
| Model rule | Any model used must be MIT/Apache-2.0 licensed and no larger than 8B parameters. Record the exact version and licence in the final documentation. |
| Submission limits | The supplied guidelines say five submissions per day. Use local validation and save every submission configuration. |

The supplied guidelines mention a 1–2 page summary, while the detailed problem statement asks for the completed `Documentation_template.md` and says there is no page limit. Prepare both: a concise 1–2 page overview plus the full methodology document.

The supplied documents state the challenge window ends at **27 September 2026, 11:59 PM IST**. Confirm the portal's displayed deadline and final-package upload requirement directly before relying on it.

---

## 2. Corrections to the original blueprint

The original guide is a useful architecture reference, but this plan intentionally changes the following parts.

| Original idea | Adopted decision | Reason |
|---|---|---|
| Raw union of TF-IDF, embeddings, and token candidates | Union for recall experiments, then cheap pre-ranking and a tuned top-*K* cap for the model and `candidate_pairs.tsv` | The competition reviews candidate compactness. A raw union can be unnecessarily large. |
| LightGBM custom pairwise F0.5 metric | Train with binary log loss; evaluate thresholds using a separate exact grouped macro-F0.5 scorer | The official metric is based on an S1 entity's full set of predictions, not independent pair rows. |
| Stratified pair-level CV | Split by `source1_entity_id` using `GroupKFold` or a grouped holdout | Pair-level folds leak the same S1 entity into both train and validation. |
| Embeddings early in the build | Add only after a reproducible lexical/structured baseline | Avoids model-download, inference, and dependency risk before a valid submission exists. |
| `python-Levenshtein` as a core dependency | Use `rapidfuzz` | It is fast, permissively licensed, and provides the needed ratios. |
| Aggressive legal-suffix deletion | Preserve full and core name versions; use suffix agreement/conflict as features | Legal suffixes can be signal and should not be thrown away. |

Important guardrail: score-gap logic can support singleton confidence, but must be tuned with the exact set-level metric. It must not accidentally suppress legitimate multiple matches for one S1 entity. Similarly, do not impose one-to-one target assignment unless the supplied data proves that a Source 2/3 record can match only one Source 1 record.

---

## 3. Delivery milestones

| Milestone | Deliverable | Exit condition |
|---|---|---|
| M0 — Data and scorer | Schema report, parsed ground truth, exact local scorer | Train labels and empty singleton lists parse correctly; scorer is tested on hand-worked cases. |
| M1 — Safe baseline | First two TSV outputs and a run log | Provided validator returns `PASS`; submit this baseline promptly. |
| M2 — Measured candidate generator | Blocking report and final capped candidate set | Candidate recall, average/maximum candidates, and runtime are recorded for each blocking setting. |
| M3 — Validated model | Out-of-fold probabilities, model, tuned decisions | Grouped macro F0.5 improves over the baseline without format regressions. |
| M4 — Final package | Reproducible ZIP, documents, saved configuration | A clean run recreates both output files and validation passes. |

---

## 4. Phase 0 — Intake, profiling, and evaluation (30–60 minutes)

1. Create the project structure below and preserve the supplied validator/template unchanged.

   ```text
   dataset/{train,test}/
   src/
   experiments/
   models/
   output/
   reports/
   ```

2. Load every `.tsv` with `sep="\t"`, preserve IDs as strings, and assert the exact columns:

   ```text
   entity_id, business_name, business_address, country
   source1_entity_id, matched_entity_ids  # ground truth
   ```

3. Parse `matched_entity_ids` into `set[str]`, treating missing/empty values as `set()`. Assert that every referenced training ID exists and is `S2-` or `S3-`.

4. Write `reports/data_profile.md` and measure, by source and country:

   - row counts, duplicate IDs, null/blank name and address rates;
   - country values (without restricting them);
   - name/address-length distributions;
   - postcode and numeric-token availability;
   - S1 match-count distribution, singleton share, and S1→S2/S1→S3 link counts.

5. Implement and unit-test the official set-level metric. For every S1 ID, calculate precision and recall over its entire predicted ID set, then F0.5; score true singletons as 1 only if their predicted set is empty; macro-average the result across S1 IDs. Test at least:

   - correct singleton → 1.0;
   - false match on a singleton → 0.0;
   - exact multi-match set → 1.0;
   - one correct and one extra match → the expected precision-penalised score;
   - no predicted matches for a non-singleton → 0.0.

6. Make a deterministic grouped holdout or `GroupKFold` split on `source1_entity_id`. Cache the split seed. Candidate generation may see the target source records because those are available at inference, but labels and threshold/model selection must use only the training fold.

---

## 5. Phase 1 — Build and submit a minimum viable baseline (2–3 hours)

The goal is a correct, reproducible first submission—not peak score.

### 5.1 Normalisation

Keep raw fields and create multiple deterministic views. Never overwrite the raw text.

| Field | Views / extracted data |
|---|---|
| Business name | lowercased Unicode-normalised string; punctuation-normalised form; compact alphanumeric form; token set; sorted tokens; core name with legal suffixes separated; full normalised name |
| Address | normalised text; compact form; token set; numeric tokens; likely house/building number; likely postcode; lightweight city/state candidates where reliably extractable |
| Country | normalised text plus `country_equal`; maintain as an open category |

Normalise common abbreviations cautiously (for example `pvt`, `ltd`, `corp`, road/street forms and `&`/`and`). Include French legal forms such as `SARL`, `SAS`, `SA`, and `EURL` as recognised suffixes. Do not use an external address parser, gazetteer, or API. If transliteration support is added, keep it a small, transparent, offline rule set and benchmark it.

### 5.2 Baseline blocking and cap

Build country-compatible multi-key inverted indexes across S2 + S3, with an explicit fallback only if profiling shows countries are unreliable. Initial keys:

- exact core name;
- first strong name token + postcode;
- first strong name token + house number;
- name token overlap;
- number + street token;
- char n-gram/name-prefix retrieval for typo resilience.

Union IDs returned by the keys, deduplicate them, then calculate a cheap deterministic pre-score from name similarity, address similarity, number compatibility, postcode compatibility, and country compatibility. Keep the top *K* candidates for each S1 (start with *K* = 20; tune later). The **capped** set is the input to the model and is exactly what is written to `candidate_pairs.tsv`.

### 5.3 Baseline pair model and decisions

For every retained pair, calculate a small feature set:

- name: `rapidfuzz` ratio, token-sort ratio, token-set ratio, Jaro-Winkler, token Jaccard, core/full-name similarity;
- address: ratio, token Jaccard, number of common tokens;
- structure: country equality, length ratios, shared numeric count, house-number agreement/conflict, postcode agreement/conflict;
- provenance: blocking key(s) that retrieved the pair and pre-score.

Label every candidate pair from ground truth. Train a simple LightGBM binary classifier (or regularised logistic regression only if LightGBM is unavailable). Use binary log loss, fixed seed, and a tunable class weight / `scale_pos_weight`. Generate holdout predictions and choose the threshold using the exact grouped local scorer.

Apply only evidence-based vetoes. A missing address component is not a conflict. Any country, postcode, or number veto must first demonstrate improved grouped F0.5 on training validation.

### 5.4 Submission gate

Generate both files for **all** test S1 records, with a row even when no candidate/match exists. Run:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Only upload once it prints `PASS`. Save the output files, code/config hash, threshold, candidate cap, local metrics, timestamp, and leaderboard result in `experiments/submission_log.csv`.

---

## 6. Phase 2 — Measure and improve the highest-value pieces (3–5 hours)

### 6.1 Blocking experiment matrix

On the grouped validation fold, evaluate each strategy alone and in combinations. Report:

```text
pair completeness / blocking recall
average candidates per S1
maximum candidates per S1
reduction ratio
candidate-generation runtime
final macro F0.5 after the same downstream model
```

Run these in order:

1. exact core-name and name-plus-structured-field keys;
2. token-overlap and character n-gram retrieval;
3. TF-IDF character/word n-gram retrieval only if it fixes observed misses;
4. cap sweep: K = 10, 15, 20, 25, 30.

Choose the smallest cap that preserves nearly all useful candidate recall and maximises final grouped F0.5. Do not choose `K` by intuition or raw retrieval recall alone.

### 6.2 Stronger model training

1. Build positives from the training ground truth.
2. Use unmatched records from the same blocking buckets as hard negatives; they resemble real mistakes. Include a controlled number of random easy negatives if necessary.
3. Start with about 3–10 hard negatives per positive, then tune it. Never let labels from held-out S1 entities enter model fitting.
4. Add feature groups one at a time and retain a group only when it improves grouped validation:

   - full/core/legal-suffix name comparisons;
   - address and character n-gram comparisons;
   - numeric and postal/city agreement and conflicts;
   - cross-field interactions such as `name_similarity * address_similarity`;
   - pre-ranking/blocking provenance.

5. Compare class weights and, if beneficial, probability calibration. Track run configuration and out-of-fold macro F0.5 for every experiment.

### 6.3 Decision tuning

Use out-of-fold pair probabilities reconstructed into complete prediction sets per S1. Sweep:

- global score threshold;
- separate S1→S2 and S1→S3 thresholds, only if validation supports it;
- narrowly scoped high-confidence acceptance rules;
- narrowly scoped contradiction vetoes;
- optional singleton confidence signals (best score, candidate count, and score gap).

For each setting, report total macro F0.5, singleton accuracy, non-singleton F0.5, predicted-match count, and false-positive count. Select the setting by the official local score, not pair accuracy, AUC, or LightGBM's internal metric.

---

## 7. Phase 3 — Optional experiments, only after M3

Run these one at a time and keep them only if they give a repeatable grouped-validation gain.

1. TF-IDF cosine retrieval/features on separate name, address, and combined fields.
2. Character n-gram enhancements where the error analysis shows typos or token reordering.
3. A compliant offline MiniLM embedding feature/retriever, after documenting the exact model licence, parameter count, download provenance, and reproducibility. Check whether it improves France-proxy performance rather than assuming it will.
4. Leave-one-country-out validation (US versus India) as a proxy for generalisation to unseen France. This is diagnostic, not a replacement for grouped validation.
5. Target-level conflict resolution only if data inspection proves a target record cannot legally match multiple S1 records.

Stop optional work immediately if it risks the validated submission, worsens the candidate cap, or cannot be reproduced before the deadline.

---

## 8. Error analysis loop

After every meaningful run, inspect sampled examples from each category:

| Category | Question to answer |
|---|---|
| Blocking false negative | Which missing key or normalisation variant would retrieve it without inflating the pool? |
| Model false positive | Which conflicting number/postcode/city/name evidence was missing or underweighted? |
| Model false negative | Was it filtered at K, underrepresented in hard-negative training, or rejected by an over-strict threshold/veto? |
| Singleton false positive | Is the threshold too permissive, or are generic names creating poor candidates? |
| France proxy failure | Is the issue suffix normalisation, address handling, or a feature tied too closely to US/India conventions? |

Changes must follow an observed error pattern and be confirmed with the grouped scorer. Avoid adding a large untested rule collection.

---

## 9. Finalisation checklist

### Output correctness

- [ ] Both files are TSV, with the exact required headers.
- [ ] Every test S1 ID appears exactly once in both files.
- [ ] Empty lists are genuinely empty cells, not `[]`, `NaN`, or `None`.
- [ ] All listed IDs exist in test S2/S3, contain no duplicates, and are never S1 IDs.
- [ ] Every predicted match is in the final capped candidate list.
- [ ] The supplied validator prints `PASS` on the exact files to upload.

### Reproducibility and package

- [ ] `src/pipeline.py` (or equivalent) recreates both outputs from supplied data with one documented command.
- [ ] `requirements.txt` pins every dependency; the model artefact/config/seed is included or recreated.
- [ ] `README.md` contains setup, training, prediction, validation, and expected-output instructions.
- [ ] `Documentation_template.md` describes data handling, blocking, candidate cap, features, model, validation, experiments, final parameters, and rule compliance.
- [ ] A 1–2 page summary exists for the separate guideline artefact.
- [ ] The package is created from a clean copy and its outputs validate again.

### Submission discipline

- [ ] `submission_log.csv` records each upload, its configuration, local metrics, portal score, and reason for the next change.
- [ ] Keep at least one known-valid submission untouched until a newer candidate passes local validation.
- [ ] Reserve final submissions for the best independently validated variants, rather than probing basic file-format errors.

---

## 10. Priority order when time is short

1. Exact local scorer, grouped split, and validator integration.
2. A valid lexical/structured baseline and an early upload.
3. Measure blocking recall and tune a compact top-*K* candidate set.
4. Hard negatives, numeric/address conflict features, threshold tuning.
5. Normalisation improvements including French suffixes and transparent transliteration rules.
6. TF-IDF, embeddings, source-specific thresholds, and global consistency experiments.

The operating principle is simple: **protect precision, but never sacrifice unmeasured blocking recall or submission validity.**