# ThinkTank — Amazon ML Challenge 2026: Business Entity Resolution

## Setup

```bash
pip install -r requirements.txt
# macOS only: ensure libomp is installed for LightGBM
brew install libomp
ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /opt/homebrew/lib/libomp.dylib
```

## Project Structure

```
dataset/
  train/  — train_source1/2/3.tsv, train_ground_truth.tsv
  test/   — test_source1/2/3.tsv
src/
  normaliser.py        — Multi-view text normalisation
  blocker.py           — Multi-key inverted index blocking
  feature_extractor.py — Pair feature computation (27 features)
  trainer.py           — LightGBM training + threshold tuning
  output_writer.py     — TSV output with spec compliance checks
  pipeline.py          — Main orchestration (train_val / submission)
  phase2_experiments.py — Phase 2 experiment suite
  data_loader.py       — Data loading + validation (Phase 0)
  data_profiler.py     — Data profiling (Phase 0)
  scorer.py            — Official grouped macro-F0.5 scorer
  splitter.py          — Stratified 80/20 grouped split
utils/
  validate_submission.py — Official submission validator (unchanged)
experiments/
  train_s1_ids.txt     — 80% train S1 IDs
  val_s1_ids.txt       — 20% val S1 IDs
  split_metadata.json  — Split configuration
  submission_log.csv   — All experiment runs logged here
models/
  lgbm_model.pkl       — Trained LightGBM model
  threshold.json       — Tuned decision threshold + metrics
output/
  matching_results.tsv — Final submission file
  candidate_pairs.tsv  — Blocking stage candidates
reports/
  data_profile.md      — Data profiling report (Phase 0)
  blocking_experiment.md — Phase 2 blocking K-sweep results
```

## Running the Pipeline

### Step 1: Validate scorer (smoke test)
```bash
python3 src/scorer.py --test
```

### Step 2: Local train/val validation
```bash
python3 src/pipeline.py --mode train_val --K 20 --spw 3.0
```
Outputs: local grouped macro-F0.5 score + saves model to `models/`.

### Step 3: Phase 2 experiments
```bash
# Blocking K sweep
python3 src/phase2_experiments.py --exp blocking

# Model + SPW sweep
python3 src/phase2_experiments.py --exp model --K 20

# Detailed threshold analysis
python3 src/phase2_experiments.py --exp threshold --K 20

# All experiments in sequence
python3 src/phase2_experiments.py --exp all
```

### Step 4: Generate submission
```bash
python3 src/pipeline.py --mode submission --K 20 --spw 3.0
```

### Step 5: Validate before upload
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
Only upload if it prints **PASS**.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Precision-first (F0.5) | β=0.5 weights precision 4× recall; wrong predictions hurt more |
| Binary log-loss (not custom metric) | Official metric is set-level; threshold-tune separately |
| GroupKFold by S1 ID | Prevents leakage across the same entity into train+val |
| rapidfuzz (not python-Levenshtein) | Fast, permissive license, comprehensive |
| Legal suffixes as features, not deleted | Suffix agreement/conflict is signal |
| Country-aware blocking | Countries are reliable (0% blank in all sources) |
| K cap on candidates | Keeps candidate_pairs.tsv compact as reviewed |

## Model

- **Algorithm**: LightGBM binary classifier (log-loss)
- **License**: MIT
- **Parameters**: 8B limit — LightGBM uses tree models, far below any neural limit
- **Features**: 27 per pair across name similarity, address similarity, structural signals, provenance

## Metric

Macro-averaged F0.5 per S1 entity:
```
F0.5 = (1.25 × P × R) / (0.25 × P + R)
Singleton: score = 1.0 iff predicted set is empty
```