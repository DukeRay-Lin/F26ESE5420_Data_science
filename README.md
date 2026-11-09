# ESE 5420 — Final Project

## Research question and dataset

**Can one-hot encoding make linear regression match an MLP?** A/B compare numerical Linear Regression against Ridge with one-hot features. MLP comparisons are still planned; the research question is not yet answered.

UCI Bike Sharing `hour.csv` contains 17,379 hourly records from 2011–2012. Predict `cnt` using `season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed`. Exclude `instant, dteday, casual, registered` from inputs. Preserve `dteday, timestamp, source_row_index` as evaluation metadata. Three weather-category 4 rows are mapped to 3 in derived data; no rows are deleted and 165 absent hours are not imputed. Raw data remain unchanged. The supplied `day.csv` is unused by A/B.

## Progress and validation results

- **Phase 1 complete:** checked dimensions, duplicates, ordering, target consistency and file identity; corrected proposal split counts.
- **Phase 2 complete:** saved chronological partitions and shared five-fold expanding CV.
- **Phase 3A complete:** trained A/B, searched the fixed Ridge grid, saved predictions/hourly MAE and computed the day-block CI. Earlier execution verified repeatability and passed four regression tests.
- **Phase 3B:** [Paper draft](Checkpoint.md) and [plain-text checkpoint](checkpoint/baseline.txt) prepared.

Errors below are **CV Validation MAE in bikes/hour**, not Final Test MAE.

| Model                                 | Mean fold MAE | Sample SD | Pooled OOF MAE | 95% bootstrap CI   |
| ------------------------------------- | ------------: | --------: | -------------: | ------------------ |
| A: scaled numerical Linear Regression |      108.7206 |   25.9760 |       108.7474 | Not computed       |
| B: one-hot + scaled numerical Ridge   |       80.1569 |   18.5738 |        80.1873 | [78.2162, 82.2004] |

B encodes `hr, weekday, mnth, season, weathersit`, ignores unseen categories, and scales the other seven predictors. Alpha=0.0001 minimizes mean fold MAE over 25 logarithmic values from 10^-4 to 10^4. Each fold/alpha gets a new Pipeline. OOF MAE pools 11,608 hourly predictions per model; mean fold MAE weights the five folds equally. SD uses ddof=1. The CI uses 5,000 whole-day resamples stratified by fold, seed=42, and percentile endpoints. Predictions are not clipped.

## Split and shared CV

The fixed boundary is **2012-08-07**: 13,891 training rows before it and 3,488 final held-out test rows from it onward. The proposal's 13,903/3,476 counts correspond to a different within-day boundary. No shuffling is used; final test data remain isolated and unevaluated.

| Fold | Training dates         | Validation dates       | Train rows | Validation rows |
| ---- | ---------------------- | ---------------------- | ---------: | --------------: |
| 1    | 2011-01-01–2011-04-09 | 2011-04-10–2011-07-15 |      2,283 |           2,327 |
| 2    | 2011-01-01–2011-07-15 | 2011-07-16–2011-10-20 |      4,610 |           2,311 |
| 3    | 2011-01-01–2011-10-20 | 2011-10-21–2012-01-25 |      6,921 |           2,321 |
| 4    | 2011-01-01–2012-01-25 | 2012-01-26–2012-05-01 |      9,242 |           2,321 |
| 5    | 2011-01-01–2012-05-01 | 2012-05-02–2012-08-06 |     11,563 |           2,328 |

Use `src.cv_splits.load_cv_splits()` and `src.prepare_data.FEATURES`. Indices are zero-based positions in train.csv for `iloc`. All models must reuse these folds; fit learned preprocessing only on each fold's training subset. Never include metadata in X.

## Project files

All current files are listed below, excluding Git's internal `.git/` directory. Unneeded macOS metadata was removed; `.gitignore` prevents it from being added accidentally.

```text
.
├── README.md                         # Overview and reproduction instructions
├── Checkpoint.md                     # Evolving Paper draft, not official template
├── requirements-phase3a.txt          # Pinned experiment dependencies
├── .gitignore                       # Ignore macOS filesystem metadata
├── checkpoint/
│   └── baseline.txt                  # English plain-text 11/9 checkpoint
├── data/
│   ├── raw/
│   │   ├── hour.csv                  # Original hourly data; do not edit
│   │   └── day.csv                   # Daily data; unused now, future use unconfirmed
│   └── processed/
│       ├── train.csv                 # Development data, target and metadata
│       ├── test.csv                  # Isolated final holdout; not for tuning
│       └── cv_folds.json             # Shared indices, dates, counts and train hash
├── src/
│   ├── prepare_data.py               # Checks, weather mapping and date split
│   ├── cv_splits.py                  # Fold creation and validated loading
│   ├── model_a.py                    # StandardScaler + LinearRegression
│   ├── model_b.py                    # ColumnTransformer + Ridge and alpha grid
│   └── evaluate.py                   # CV, tuning, bootstrap and safe output
├── tests/
│   └── test_phase3a.py               # Bootstrap, preprocessing and safety checks
└── results/
    ├── baseline_metrics.json         # Metrics, CI, seed, settings and versions
    ├── model_a_cv.csv                # A fold MAEs, dates and counts
    ├── model_b_cv.csv                # B fold MAEs, dates, counts and alpha
    ├── model_b_tuning.csv            # All alpha candidates and fold scores
    ├── validation_predictions.csv   # 23,216 predictions, dates, hours and folds
    └── validation_mae_by_hour.csv   # A/B hourly MAE and counts; 48 rows
```

### File retention

| Classification                     | Files / decision                                                                                                                                                      |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Necessary                          | Project documentation, submission text, requirements, .gitignore, original hour.csv, all src/ and tests/ files.                                                       |
| Regenerable but required to retain | All three processed artifacts and six results files: preserve the exact shared folds and published baseline evidence.                                                 |
| Unrelated / unnecessary            | The 0-byte Test.py placeholder had no code, import or runtime file dependency and was removed. No caches, logs, obsolete reports or empty directories remained.       |
| Uncertain; retained                | data/raw/day.csv (57,569 bytes): no current code/test dependency, but possible future use is unconfirmed. It is not a duplicate of hour.csv and has not been deleted. |

The inventory contains 22 files, excluding .git/. The missing .gitignore was recreated with the six macOS metadata rules. No model, data, result, Paper draft or submission text was changed. Narrative mentions of Test.py in the existing report-generator text are historical descriptions, not file-read dependencies; model source remains unchanged.

Redundant reports, preparation summary and bootstrap draws are not retained. Their absence does not break evaluation or tests. Existing generation commands recreate their respective outputs; conflicting existing content triggers an error rather than silent replacement.

## Reproduction

Use Python 3.12 (recorded run: 3.12.14), from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase3a.txt

# Trains A/B using the existing train.csv and saved folds.
python -B -m src.evaluate --verify-repeat

# Regression tests use synthetic examples, not the final test data.
python -B -m unittest discover -s tests -p 'test_phase3a.py' -v
```

To preserve the cleaned results directory on a future rerun, use `python -B -m src.evaluate --verify-repeat --output-dir results/rerun-phase3a` with a fresh output directory. Four tests passed during the preceding cleanup; this documentation update checks command entry points without rerunning training.

Only when deliberately regenerating Phase 2 artifacts, run `python -B -m src.prepare_data` followed by `python -B -m src.cv_splits`. Do not rebuild separate folds for different models. Evaluation validates training/fold hashes and blocks opening test.csv.

## Limitations and next steps

Same-CV alpha selection and scoring can introduce optimism. The CI is conditional on the selected alpha and fitted models; it excludes selection uncertainty and dependence across days or overlapping training windows. Alpha is at the grid's lower endpoint. Missing hours, unseen early-fold categories and negative predictions warrant analysis. A/B change both representation and regularization.

Per instructor feedback supplied by the team, add **C+** and include **B+** in the main comparison. **B+, C and C+ are not implemented or evaluated**; exact specifications remain unconfirmed. Before the **11/16 Early Paper Draft**, plan to complete these models, extend the existing A/B hourly comparisons and write error analysis. The Final Paper will document methods, model comparison, uncertainty and limitations. Final-test evaluation must wait until model selection is complete and separately authorized.
