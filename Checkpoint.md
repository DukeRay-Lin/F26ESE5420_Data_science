# F26ESE5420_Data_Science_FinalProject

## Paper working draft

**Research question:** Can one-hot encoding make linear regression match an MLP?

This is the team's evolving paper draft, not an official submission template. The November 9 baseline submission is [checkpoint/baseline.txt](checkpoint/baseline.txt).

Abstract in progress

We study how feature representation affects hourly bike-demand prediction. On shared chronological validation folds, numerical-feature Linear Regression (A) achieves pooled out-of-fold MAE of 108.7474 bikes/hour, while Ridge with one-hot categorical features (B) achieves 80.1873. B's fold-stratified whole-day bootstrap 95% interval is [78.2162, 82.2004]. These are development-set validation results; the final test set has not been evaluated. The planned MLP comparisons remain incomplete, so the research question is not yet answered. The A/B comparison changes both representation and regularization and does not isolate their causal contributions.

### Data and experimental design

The source is the UCI Bike Sharing hourly dataset, `data/raw/hour.csv`: 17,379 records spanning 2011-01-01 through 2012-12-31. Its original bytes are retained. No rows were deleted; three weather-category 4 records are mapped to 3 only in derived data. The 165 absent hourly timestamps are not imputed and are not assumed to represent zero rentals.

The target is `cnt`. The 12 inputs are `season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed`. `instant, dteday, casual, registered` are excluded from inputs; `dteday, timestamp, source_row_index` remain evaluation metadata.

The boundary is fixed at 2012-08-07: 13,891 training records before the boundary and 3,488 held-out test records on or after it. The earlier proposal counts of 13,903/3,476 correspond to a different within-day row boundary and are incorrect for the chosen date rule. The date boundary has not moved.

All models use the saved five expanding-window folds with no shuffling. Each validation window contains 97 calendar days; the initial training window contains 99 days. Every fold's training dates strictly precede its validation dates. The initial 2,283 training rows have no validation predictions; the remaining 11,608 training rows each receive one OOF prediction per model.

### Completed baseline methods

- **A:** StandardScaler followed by sklearn LinearRegression on all 12 numerical features.
- **B:** OneHotEncoder with `handle_unknown="ignore"` for `hr, weekday, mnth, season, weathersit`; StandardScaler for the seven remaining features; ColumnTransformer and Ridge inside a Pipeline. Every fold and alpha receives an independently fitted pipeline. Metadata and the target never enter the predictors.
- **Tuning:** `np.logspace(-4, 4, 25)`, shared folds, minimum unweighted mean fold MAE. The selected alpha is 0.0001, the lower grid endpoint. Predictions are not clipped or rounded.

### Validation results

All MAEs below are in bikes/hour. SD is the sample standard deviation across five folds (`ddof=1`); pooled OOF MAE weights hourly observations equally rather than weighting folds equally.

| Metric           |      Model A |            Model B |
| ---------------- | -----------: | -----------------: |
| Fold 1 MAE       |     100.7524 |            85.0015 |
| Fold 2 MAE       |     100.7988 |            67.5361 |
| Fold 3 MAE       |      78.2528 |            60.1917 |
| Fold 4 MAE       |     115.0180 |            79.6439 |
| Fold 5 MAE       |     148.7808 |           108.4114 |
| Mean fold MAE    |     108.7206 |            80.1569 |
| Sample SD        |      25.9760 |            18.5738 |
| Pooled OOF MAE   |     108.7474 |            80.1873 |
| Bootstrap 95% CI | Not computed | [78.2162, 82.2004] |

The interval uses 5,000 resamples with seed=42. Within each fold, complete observed dates are sampled with replacement, retaining all observed hours in each chosen date. Each resample pools absolute-error sums and observation counts; endpoints are the 2.5th and 97.5th percentiles. The interval is conditional on the selected alpha and fitted fold models, not a final-test interval.

Numeric evidence is retained in [baseline_metrics.json](results/baseline_metrics.json), [A fold metrics](results/model_a_cv.csv), [B fold metrics](results/model_b_cv.csv), [tuning results](results/model_b_tuning.csv), [validation predictions](results/validation_predictions.csv), and [hourly MAE](results/validation_mae_by_hour.csv). The discarded bootstrap draw file is not needed to reconstruct the interval from the saved predictions and code.

### Limitations and planned extensions

Same-CV tuning and scoring can make B's reported performance optimistic. The bootstrap does not rerun model fitting or alpha selection and does not capture dependence across days or overlapping training windows. Missing hours, unseen early-fold categories, and changing error levels across periods warrant further analysis. The best alpha being at the grid boundary does not establish an optimum outside that grid. Both models can make negative predictions; no postprocessing comparison has been performed.

Per the instructor feedback supplied by the team, C+ is to be added and B+ is to join the main comparison. **B+, C, and C+ are planned, not trained.** Their exact specifications remain to be confirmed. Before the November 16 Early Paper Draft, plan to implement these models on the same folds, extend hourly comparisons beyond the existing A/B tables, and develop error analysis. For the Final Paper, document model specifications, selection procedures, uncertainty, and limitations. Evaluate the final test set only after model selection and separate authorization.
