# Population Report — Active-Span Filter (Protocol P1, task 1.4)

Filter: active span (first nonzero sale -> dataset end) >= 730 days (2 years), per PROTOCOL_P1.md. Declared before this report was generated; not adjusted after seeing these numbers.

## Overall

- Series (product x location) kept: **28,000 / 30,490** (91.8%)
- Volume retained: **65,831,450 / 66,927,173** units (98.4%)

## Per category

| Category | Series kept | Series total | Share kept | Volume kept | Volume total | Share of volume retained |
|---|---|---|---|---|---|---|
| FOODS | 12,959 | 14,370 | 90.2% | 45,216,066 | 45,922,427 | 98.5% |
| HOBBIES | 5,400 | 5,650 | 95.6% | 6,168,358 | 6,240,656 | 98.8% |
| HOUSEHOLD | 9,641 | 10,470 | 92.1% | 14,447,026 | 14,764,090 | 97.9% |

Note: this filter is computed on the full ingested history (2011-01-29 to dataset end). Per Protocol P1 / E3, the backtest harness (task 1.5) must recompute active span from **train-fold data only** for each rolling-origin fold, to avoid using future information to decide which series enter training — the figures here are a whole-history population view, not the per-fold filter that will actually gate model fitting.

---

## Per-fold population (train-fold data only, Protocol P1 / E3)

Each fold's eligible population is computed from ONLY that fold's training data (`ds <= fold.train_end`) — active span uses the fold's own train cutoff as "dataset end", not the global dataset end. This avoids leaking a series' future activity into today's inclusion decision. The eligible population, and therefore volume retained, differs per fold.

| Fold | Train end | Holdout | Series kept | Series total | Share kept | Volume kept | Volume total | Share of volume retained |
|---|---|---|---|---|---|---|---|---|
| 1 | 2016-01-31 | 2016-02-01 to 2016-02-28 | 26,304 | 30,490 | 86.3% | 60,324,492 | 62,160,048 | 97.0% |
| 2 | 2016-02-28 | 2016-02-29 to 2016-03-27 | 26,799 | 30,490 | 87.9% | 61,743,488 | 63,335,716 | 97.5% |
| 3 | 2016-03-27 | 2016-03-28 to 2016-04-24 | 27,366 | 30,490 | 89.8% | 63,230,200 | 64,511,783 | 98.0% |
| 4 | 2016-04-24 | 2016-04-25 to 2016-05-22 | 27,684 | 30,490 | 90.8% | 64,504,065 | 65,695,409 | 98.2% |