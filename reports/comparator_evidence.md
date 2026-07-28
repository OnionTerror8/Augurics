# Bayesian Comparator Evidence — `bayes_glm_pooled` (task 2.1)

This evidence fit (182 series, fold-1 train sample) demonstrates the comparator's construction only. No number in this document is a gate result. Gate results come exclusively from full per-fold populations through the validated harness path.

Sample: 182 series drawn from fold 1's train-fold population (`ds <= 2016-01-31`), spanning all three M5 categories (FOODS, HOBBIES, HOUSEHOLD), and including two series chosen specifically to demonstrate the active-span offset: `FOODS_1_130::CA_1` (first nonzero sale 2011-01-29, the first day of the dataset) and `FOODS_3_033::WI_1` (first nonzero sale 2014-01-31, near the boundary of the 730-day active-span filter as of fold 1's train cutoff).

## 1. Family printed from the fitted object, with overdispersion

```
family (from fitted object): NegBinomialRV
overdispersion_ratio: 7.497617273705005
```

`family` is read from `model.named_vars["y_obs"].owner.op.__class__.__name__` — the fitted PyMC graph node's own class, never a config string.

The observed variance-to-mean ratio is **~7.50**, i.e. counts are roughly 7.5x more dispersed than a Poisson model would assume (Poisson requires variance = mean, ratio = 1). Since a Poisson standard error scales with `sqrt(variance)`, fitting Poisson here would understate standard errors by a factor of **`sqrt(7.50) ≈ 2.74x`** — a true effect at 2.7 standard errors would appear to be at ~7.5 standard errors under Poisson, i.e. badly overconfident, spuriously "significant" coefficients. The Negative Binomial family is demonstrated **necessary** by this data, not a stylistic preference.

## 2. Active-span offset — two example series

```
EARLY-listed (first sale 2011-01-29): FOODS_1_130::CA_1
  full-window rows: 1829, active-span rows: 1829
  full-window mean y: 1.6763 -> log = 0.5166
  active-span mean y: 1.6763 -> log = 0.5166
  model offset used: 0.5166

LATE-listed (first sale 2014-01-31): FOODS_3_033::WI_1
  full-window rows: 1829, active-span rows: 731
  full-window mean y: 0.1509 -> log = -1.8911
  active-span mean y: 0.3776 -> log = -0.9740
  model offset used: -0.9740
```

`FOODS_1_130::CA_1` sold from day one of the dataset, so its full-window mean and active-span mean are identical (1829 rows either way) — the offset choice is a no-op for this series. `FOODS_3_033::WI_1` is the case E3 exists for: it has 1829 total rows in the panel but only 731 of them are within its active span (from first sale onward). Its full-window mean (0.1509, dragged down by ~1098 structural pre-listing zero rows) gives `log = -1.8911`, while its active-span mean (0.3776) gives `log = -0.9740` — the model uses **-0.9740**, the active-span value, confirming the offset is never computed from the full-window mean.

## 3. Pooling structure

```
mu_beta shape (population-pooled coefficient vector): (23,)
sigma_beta shape (shared pooling scale): ()
beta_offset shape (per-category deviation, n_categories x n_features): (3, 23)
categories pooled over: {'FOODS': 0, 'HOBBIES': 1, 'HOUSEHOLD': 2}
```

Non-centred hierarchical parameterisation: `beta = mu_beta + sigma_beta * beta_offset[category]`. `mu_beta` (23 regressor coefficients: event_flag, snap_flag, log_price, trend, 7 day-of-week dummies, 12 month dummies) is the **population-pooled** mean, shared across all three categories; `sigma_beta` is a single shared pooling scale; `beta_offset` holds each category's deviation from that population mean. Both `mu_beta` and `beta_offset` carry `Normal(0, ·)` priors — zero-centred, mirroring the platform-class regulariser.

**Per-series parameters: none of the regressor coefficients are per-series.** The only per-series quantity is the active-span offset (§2), which is a deterministic value computed directly from each series' own data, not a fitted/pooled parameter. Every regressor coefficient pools at the category level; there is no series-level free parameter for the model to overfit to an individual product/store's noise.

## 4. Co-firing check

```
'event_flag/snap_flag separability OK'
```

In this real 182-series sample, `event_flag` and `snap_flag` are not collinear (SNAP days and calendar events don't coincide often enough to be perfectly co-firing), so the check reports separability is OK, verbatim. The alternative branch — reporting **"NOT separately identifiable"** verbatim when the two streams do co-fire perfectly — is exercised and passing in `tests/test_comparators.py::test_identifiability_check_detects_perfectly_co_firing_streams`, which constructs a synthetic panel where the two flags are always identical and asserts the exact string appears in both `fit_result.diagnostics["identifiability_note"]` and the model's internal state.

## 5. Coefficient plausibility at observed exposures

Evaluated at each feature's own median and p90 observed value in this sample, not naively at feature = 1.0:

| feature | beta (pooled) | median exposure | effect @ median | p90 exposure | effect @ p90 | feature=1.0 attainable | effect @ 1.0 |
|---|---|---|---|---|---|---|---|
| event_flag | -0.0434 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.9575 |
| snap_flag | 0.0823 | 0.000 | 1.0000 | 1.000 | 1.0858 | True | 1.0858 |
| log_price | 0.4588 | 1.1878 | 1.7246 | 2.1838 | 2.7236 | True | 1.5822 |
| trend | 0.0052 | 30.4667 | 1.1714 | 54.8667 | 1.3296 | True | 1.0052 |
| dow_0 | -0.4092 | 0.000 | 1.0000 | 1.000 | 0.6642 | True | 0.6642 |
| dow_1 | -0.4844 | 0.000 | 1.0000 | 1.000 | 0.6161 | True | 0.6161 |
| dow_2 | -0.4677 | 0.000 | 1.0000 | 1.000 | 0.6264 | True | 0.6264 |
| dow_3 | -0.4627 | 0.000 | 1.0000 | 1.000 | 0.6296 | True | 0.6296 |
| dow_4 | -0.3679 | 0.000 | 1.0000 | 1.000 | 0.6922 | True | 0.6922 |
| dow_5 | -0.1371 | 0.000 | 1.0000 | 1.000 | 0.8719 | True | 0.8719 |
| dow_6 | -0.2210 | 0.000 | 1.0000 | 1.000 | 0.8017 | True | 0.8017 |
| month_1 | -0.3574 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.6995 |
| month_2 | -0.1969 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.8213 |
| month_3 | -0.3904 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.6768 |
| month_4 | -0.4169 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.6591 |
| month_5 | -0.4147 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.6605 |
| month_6 | -0.3844 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.6808 |
| month_7 | -0.4106 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.6633 |
| month_8 | -0.3819 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.6826 |
| month_9 | -0.3834 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.6815 |
| month_10 | -0.3355 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.7150 |
| month_11 | -0.3497 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.7049 |
| month_12 | -0.2649 | 0.000 | 1.0000 | 0.000 | 1.0000 | True | 0.7673 |

All 23 features in this sample happen to be attainable at 1.0 (the binary flags/dummies do take value 1, and `log_price`/`trend` both range through 1.0 in this data), so the "not attainable" branch isn't exercised by this particular real fit. It is exercised and tested in `tests/test_comparators.py::test_coefficient_plausibility_report_flags_unattainable_feature_eq_1`, which constructs a series with a constant price far from `log_price = 1.0` and asserts that column is reported as `"not attainable"` rather than a number computed off the support of the data.

## 6. ADVI/NUTS decision

```
fit_method: ADVI (2000 iterations)
nuts_spot_check: {'ran': True, 'category': 'FOODS', 'mean_abs_coef_diff_advi_vs_nuts': 0.1706269399816926}
```

ADVI (2000 iterations, Adam optimiser, learning rate 0.01, eta clipped to [-10, 10]) was used for the full fit, for speed. A NUTS spot-check ran on the **FOODS** category alone (200 tune + 200 draw, 1 chain) and its posterior mean coefficient vector was compared against ADVI's mean coefficient vector for that same category: **mean absolute difference of 0.171** across the 23 regressor coefficients. This is a real, non-trivial gap (not exact agreement) — documented here rather than asserted away, consistent with kickoff §3's "ADVI acceptable for speed with a NUTS spot-check... documented either way." A gap of this size on a 182-series/2000-iteration ADVI fit is unsurprising; it is not evidence the fit is wrong, but it is evidence ADVI and NUTS are not interchangeable here, and any future decision to lean more heavily on ADVI point estimates should account for it.

### Addendum: the NUTS spot-check is now bounded, independent of this evidence run

At this 182-series scale, the FOODS subset was 113,398 rows and the spot-check ran in 332s. Before preparing the real 4-fold gate-relevant run, we timed what the *unbounded* spot-check would cost at real fold-1 scale: fold 1's FOODS category alone is 22,432,685 rows (12,265 series) — scaling the measured per-row cost gives an estimated **~18.2 hours** for one fold's spot-check alone. This is a design catch, not a retraction of the evidence above: `engine.models.comparators.NUTS_SPOT_CHECK_MAX_ROWS` now caps the spot-check to a bounded, seeded, whole-series subsample (~100k rows) regardless of population size, and the batch orchestrator runs it once (fold 1 only), not per fold. See `KNOWN_ISSUES.md` for the full numbers and `tests/test_comparators.py` for the cap tests. The 0.171 mean-absolute-coefficient-difference figure above remains the evidence-fit's result; the bounded full-scale confirmation on real fold-1 data is a separate, later step gated behind this review.

## Observations (not defects)

Two properties of the fitted model worth flagging explicitly, so they aren't later misread as bugs:

**(a) The day-of-week and month dummy blocks are mutually collinear with the absent intercept.** There is no explicit intercept term in this model (the active-span offset plays that role per series); with no intercept to anchor them, the `dow_*`/`month_*` dummy coefficients (§5's table) partly absorb the general level rather than isolating pure seasonal deviation. Their individual values should not be read as "this day/month causes a +/-X% seasonal effect" in isolation — under the zero-centred, non-centred hierarchical priors, this does not bias **predictions** (the level and seasonal components net out consistently at fit time), but it does mean the per-dummy coefficients in the plausibility table are not directly interpretable as standalone seasonal effect sizes.

**(b) `log_price` at +0.46 (§5) reflects cross-series composition under pooling with a level offset, not within-series price elasticity.** Because price varies far more *between* series (a $2 item vs a $20 item) than *within* a single series over time in this sample, and the model pools the price coefficient across all series in a category, the fitted coefficient is dominated by the cross-sectional price/volume relationship (higher-priced items in a category tend to sell at different absolute volumes) rather than "if this specific item's price rises 1%, its own demand falls/rises by 0.46%." This is an acceptable and expected property of the pooled comparator as specified — it is not attempting to estimate within-series elasticity — but it should be flagged for contrast when the Rung 2 challenger's price features (task 2.3, which will likely include within-series price and price-change features per the feature library) are reviewed, so the two models' price coefficients are not compared as if they measured the same thing.
