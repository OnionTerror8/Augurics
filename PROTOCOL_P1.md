## 1. The Pre-Registered Gate Protocol (commit this file first)

Registered before any model is fitted. This section is append-only thereafter.

### Protocol P1 — "M5 Retail Gate"

- **Data:** M5, all series with ≥ 2 years active history (active span = first nonzero sale → dataset end). Report PLs kept and **share of volume retained** (expect sparse-series exclusions; per E3, filters use train data only).
- **Coverage diagnostic first (E1):** month-by-month profile of event/price coverage and mean sales BEFORE any fit. Findings recorded in `experiments.coverage_report`. Any cliff bounds the training window and is documented as a label-integrity decision.
- **Backtest:** rolling origin, 4 folds, 28-day holdout per fold, expanding window. Metrics: WMAPE (primary), bias, MASE, pinball@0.9 (where quantiles exist).
- **Aggregation:** volume-weighted across series; also reported per category and per horizon-week.

### Gates (immutable)

| Gate | Challenger vs comparator | Pass condition |
|---|---|---|
| **G1** | Best Rung 2 model vs **Seasonal naive** | WMAPE improvement ≥ 15% |
| **G2** | Best Rung 2 model vs **ETS (AutoETS)** | WMAPE improvement ≥ 8% |
| **G3** | Best Rung 2 model vs **Bayesian comparator (§3)** | WMAPE improvement > 0 with fold-consistent sign (wins ≥ 3 of 4 folds) |

Comparator corrections are permitted with written justification (E2); thresholds are not. A failed gate is reported as failed.
