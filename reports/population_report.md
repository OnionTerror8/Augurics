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