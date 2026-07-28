# GATE_REPORT (Protocol P1)

## Gates (Protocol P1, immutable thresholds)

- **G1** (vs Seasonal Naive, >= 15% improvement): **PENDING** -- no Rung 2 challenger result yet -- task 2.3 training deferred by a verified memory-infeasibility check (KNOWN_ISSUES.md: fold-1 scale estimate 5.13GB vs this laptop's available RAM), awaiting the VM in PROVISIONING.md. Not a false pass/fail; the evaluation below runs the moment `lgbm_promo_v1` results exist for all four folds.
- **G2** (vs AutoETS, >= 8% improvement): **PENDING** -- no Rung 2 challenger result yet -- task 2.3 training deferred by a verified memory-infeasibility check (KNOWN_ISSUES.md: fold-1 scale estimate 5.13GB vs this laptop's available RAM), awaiting the VM in PROVISIONING.md. Not a false pass/fail; the evaluation below runs the moment `lgbm_promo_v1` results exist for all four folds.
- **G3** (vs Bayesian comparator, >0% improvement, wins >= 3/4 folds): **PENDING** -- no Rung 2 challenger result yet -- task 2.3 training deferred by a verified memory-infeasibility check (KNOWN_ISSUES.md: fold-1 scale estimate 5.13GB vs this laptop's available RAM), awaiting the VM in PROVISIONING.md. Not a false pass/fail; the evaluation below runs the moment `lgbm_promo_v1` results exist for all four folds.

## Per-fold, volume-weighted WMAPE

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---|---|---|---|---|
| auto_ets | 0.7302 | 0.7300 | 0.7389 | 0.7105 | 0.7274 |
| naive | 0.9832 | 0.9965 | 0.9658 | 0.9386 | 0.9710 |
| seasonal_naive | 0.8572 | 0.8656 | 0.8778 | 0.8516 | 0.8630 |
| theta | 0.7338 | 0.7352 | 0.7427 | 0.7144 | 0.7315 |

## Per-fold population (series kept, volume retained)

| Fold | Series kept | Series total | Share kept | Volume kept | Volume total | Share of volume retained |
|---|---|---|---|---|---|---|
| 1 | 26,304 | 30,490 | 86.3% | 60,324,492 | 62,160,048 | 97.0% |
| 2 | 26,799 | 30,490 | 87.9% | 61,743,488 | 63,335,716 | 97.5% |
| 3 | 27,366 | 30,490 | 89.8% | 63,230,200 | 64,511,783 | 98.0% |
| 4 | 27,684 | 30,490 | 90.8% | 64,504,065 | 65,695,409 | 98.2% |

## Per-category WMAPE

_Not available for this run: the cached fold frames used by the Rung 1 baseline batch predate category-aware extraction. `score_model_by_category` is implemented and tested (tests/test_score_by_category.py) and will populate this section once fold frames are rebuilt with the `category` column, or for any future run that uses `run_baselines.score_model_by_category` directly._