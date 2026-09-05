# MVP Implementation Spec — PM2.5 Next-Hour Model

This turns `datathon_plan.docx` into a concrete, buildable spec. It's been implemented and run end-to-end (`pipeline.py`, attached) so every number below is real, not projected. Treat this as the thing you build first — the "Phase 2" section at the end is explicitly deferred, not forgotten.

---

## 1. Definition of done for the MVP

A single script that, given `train.csv`:
1. Cleans + engineers features identically for train and test (no leakage).
2. Assigns each row to one of 5 chronological CV blocks.
3. Runs the **inner loop** (rolling-origin CV) for any given model and returns one averaged RMSE.
4. Can run the **outer loop**: swap in Model A (baseline) vs Model B (LightGBM) and compare.
5. Logs every run to a research log (schema below).
6. Once a model is chosen, retrains on 100% of `train.csv` and produces a `submission.csv` against `test_1_.csv`.

That's the whole MVP. Everything else (extra features, tuning, ensembling) is Phase 2.

---

## 2. CV block definition (exact, implemented)

Per the plan: 5 chronological blocks, each end-exclusive, spanning the full training range (`2013-03-01` → `2016-08-31`):

| Block | Start | End (excl.) | Rows | ~Months |
|---|---|---|---|---|
| 1 | 2013-03-01 | 2013-12-01 | 78,014 | 9 |
| 2 | 2013-12-01 | 2014-08-01 | 68,419 | 8 |
| 3 | 2014-08-01 | 2015-04-01 | 68,338 | 8 |
| 4 | 2015-04-01 | 2015-12-01 | 68,614 | 8 |
| 5 | 2015-12-01 | 2016-09-01 | 77,569 | 9 |

Balanced within ~13% of each other — no block dominates. `assign_block()` in `pipeline.py` implements this as a simple date-range lookup; reuse it everywhere so block membership is never computed two different ways.

**Known, accepted limitation (see §8):** Block 5 — the block used as final validation in Round 4 — ends in Aug 2016 (summer), which is the *easy* season, while the real test set is autumn/winter. This is the plan's own documented trade-off. MVP ships with it anyway; Phase 2 swaps in a winter-aware validation block.

---

## 3. Feature engineering spec

Every feature is derivable from a single row's own `station`/`observation_timestamp`/raw readings — nothing here looks across rows in a way that could leak the target. Grouped in order of expected importance (confirmed empirically in §7):

| Feature | Type | Derivation | Why (from EDA / external-factors research) |
|---|---|---|---|
| `PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM` | numeric | raw, imputed (§4) | PM10 is the strongest single predictor (physically a superset of PM2.5); DEWP/TEMP/PRES jointly proxy inversion conditions |
| `year_frac` | numeric | `year + day_of_year/365` | Captures the 2013→2016 secular decline from the Clean Air Action Plan (SO2 −59%, PM2.5 −20% over the period) — a tree model needs this as an explicit monotonic-ish signal, it won't infer it from weather alone |
| `hour_sin/cos`, `month_sin/cos` | numeric | cyclical encoding | Raw `hour`/`month` breaks the wrap-around (hour 23 and hour 0 are adjacent); captures rush-hour + midday-O3 + winter/summer cycles |
| `heating_season` | binary | 1 if date in Nov15–Mar15 | Direct flag for the single strongest seasonal driver (coal heating) |
| `wd_clean_dirty_rank` | ordinal | wd mapped NW=0 (cleanest) → ESE=15 (dirtiest), per measured PM2.5-by-wind-direction ordering | Wind direction is one of the strongest features found in EDA (~2x PM2.5 swing, S/SE vs N/NW) — giving it an explicit clean→dirty order helps a tree split more efficiently than an unordered category |
| `wd_cat`, `station_cat` | categorical | native LightGBM categoricals | Station is a genuine proxy for urban-core vs. rural-background baseline (Dongsi ~84 vs Dingling ~65) |
| `is_weekend` | binary | dow >= 5 | Ozone-weekend-effect / traffic pattern proxy |
| `pm10_x_spring` | numeric | `PM10 × 1[month in {3,4,5}]` | Lets the model treat a given PM10 reading differently in dust-storm season vs. not |
| `pm10_capped`, `co_capped` | binary | 1 if reading at sensor ceiling (999 / 10,000) | Tells the model "this reading is a floor, not the truth" instead of trusting it as exact |
| `is_cny_window` | binary | ±2 days around each Lunar New Year date (2013–2017) | Sharp, calendar-driven SO2/PM spike no weather variable explains — **and it fires in `test_1_.csv` too** (CNY 2017-01-28 falls inside the test period) |
| `is_apec_blue`, `is_parade_blue` | binary | fixed date windows, 2014/2015 only | Real, huge effects in training data. **These will always be 0 in `test_1_.csv`** (both predate the test window) — expected and fine; they help the model learn "government shutdown ⇒ pollution collapses" as a concept without ever needing to fire at test time. |
| `is_red_alert` | binary | Dec 2015 + Dec2016/Jan2017 windows | Covers both the training-period red alerts **and** the severe Dec2016/Jan2017 haze episode that sits inside `test_1_.csv` (PM10 hit 411–496 on those days) |

**Deliberately excluded from MVP:** lag features (previous-hour/24h PM2.5 per station), rolling means, and per-station target-encoding. These are natural next steps (Phase 2) but each introduces its own leakage risk (must be computed strictly causally, respecting block boundaries) that's not worth the complexity for a first working model.

---

## 4. Missing-value imputation spec

Per the plan's own finding — gaps clump by station and era (CO worst, concentrated in 2013), so a single global fill is wrong. Implemented as a 3-stage fallback, in this exact order:

1. **Forward-fill, capped at 3 consecutive hours**, computed *within each station's own time-ordered series* (never crosses station boundaries). Handles short sensor blips using the strong hour-to-hour autocorrelation the plan calls out.
2. **Station + month + hour-of-day group mean** for anything still missing (e.g. CO's multi-day 2013 outages). Preserves the diurnal + seasonal shape instead of flattening it.
3. **Station overall mean**, then **global mean**, as a final safety net for the rare station+month+hour combination with zero observations.

Verified: zero NaNs remain in any of the 10 raw numeric columns after this runs on the full training set.

---

## 5. Outlier / sensor-cap handling

Per the plan: don't silently trust `PM10 == 999` (official monitoring-scale ceiling) or `CO == 10000` (sensor cap) as exact readings, but don't discard them either — they're real extreme-pollution rows, just censored. MVP handles this with `pm10_capped`/`co_capped` flag columns (§3) and otherwise leaves the raw (capped) value in place, since removing these rows would specifically remove the hardest, highest-value cases RMSE punishes most.

The single `PM2_5_next_hour == 999` row (target, not a feature) is left as-is — one row, not a systemic cap, not worth special-casing.

---

## 6. Model spec

### Baseline (Model A)
Per-station mean of the target, computed on the training portion of each round only, applied by station lookup to validation rows. This is `baseline_predict()` in `pipeline.py` — no fitting, no hyperparameters, ~5 lines of code. This is the number every subsequent model must beat.

### First real model (Model B): LightGBM
Chosen over linear/distance-based models because:
- Handles the mixed numeric + categorical (`station`, `wd`) feature set natively — no one-hot/scaling needed.
- Tree-based models don't require feature scaling (confirmed not needed per the workshop slides).
- Fast to iterate (seconds per fold on this data size), which matters for the outer-loop tweak cycle.
- Robust to the remaining skew in the raw features without needing a log-transform first.

Starting hyperparameters (deliberately conservative, not tuned — tuning is an outer-loop activity, not part of the MVP):

```python
dict(objective='regression', metric='rmse', n_estimators=500,
     learning_rate=0.05, num_leaves=63, min_child_samples=30,
     subsample=0.8, colsample_bytree=0.8, random_state=42)
```

---

## 7. Validated results — this spec actually runs

Ran both models through the exact inner loop specified in the plan (Rounds 1–4, train-on-past/validate-on-next-block):

| Model | Round1 (Val Blk2) | Round2 (Val Blk3) | Round3 (Val Blk4) | Round4 (Val Blk5) | **Avg RMSE** |
|---|---|---|---|---|---|
| A: per-station mean (baseline) | 83.78 | 81.76 | 68.14 | 82.82 | **79.13** |
| B: LightGBM, MVP feature set | 42.01 | 30.60 | 33.09 | 33.21 | **34.73** |

**56% improvement over baseline** on the first real model, using only the features in §3 — before any tuning. This is your number to beat going into the outer loop.

Feature importance (split count, full-training-set fit) confirms the EDA-driven feature choices are pulling weight rather than sitting idle:

```
year_frac              5346   <- secular trend feature
DEWP                   3447
PM10                   3074   <- the physically-expected top predictor
station_cat            2327
CO                      2073
...
pm10_x_spring            918
month_sin/cos, hour_sin/cos   (each 700-900)
is_cny_window              72
is_apec_blue / is_parade_blue / is_red_alert / capped-flags   single digits or 0
```

Note the calendar-event flags rank low by split-count — expected, since they fire on a small fraction of rows and a gradient-boosted tree naturally favors continuous features with more split points. Don't read this as "they don't matter"; it just means split-count isn't the right lens for rare-but-large-effect binary flags. Worth re-checking with SHAP or gain-based importance in Phase 2 rather than dropping them on this evidence alone.

---

## 8. Research log schema

One row per experiment, exactly per the plan:

| Column | Notes |
|---|---|
| `exp_id` | sequential int |
| `timestamp` | when it was run |
| `change` | the ONE thing changed vs. the previous row |
| `cv_avg_rmse` | inner-loop average |
| `block_rmses` | list of 4 per-block scores (spot instability) |
| `train_rmse` | (Phase 2, §9) — training-set RMSE, to catch the overfitting gap |
| `kaggle_public` | only filled in on submission rounds |
| `keep_y_n` | did it beat current best |
| `notes` | anything odd |

Suggested starter rows:

| exp_id | change | cv_avg_rmse | block_rmses | keep? |
|---|---|---|---|---|
| 1 | Baseline: per-station mean | 79.13 | [83.78, 81.76, 68.14, 82.82] | — (reference) |
| 2 | LightGBM, MVP feature set (§3) | 34.73 | [42.01, 30.60, 33.09, 33.21] | Y |

---

## 9. Explicitly deferred to Phase 2 (don't build these into the MVP)

- **Winter-aware validation block** — the plan itself flags this as "an upgrade for later, not day one." Once the MVP loop works, swap block 5 for a previous autumn/winter slice (e.g. Sep2015–Feb2016) so your final validation round mimics the real exam's hard season instead of flattering you with the summer tail.
- **Train-vs-validation RMSE gap tracking** (§8 `train_rmse` column) — needed to catch overfitting per the plan's own "(a) train-vs-validation gap" warning, but not required to get a first working model out the door.
- **Lag/rolling features** (previous-hour PM2.5, 24h rolling mean per station) — high expected value, but must be built causally (only using data strictly before the prediction row, respecting block boundaries) to avoid leakage. Build and test this in isolation once the MVP loop is trustworthy.
- **Log-transform of the target** — the plan notes the target is right-skewed (skew ≈ 2.0); worth testing `log1p(target)` with a matching `expm1` on prediction, compared against the raw-scale model on the same inner loop, rather than assumed to help.
- **Hyperparameter tuning** — the §6 params are a reasonable starting point, not a tuned result. Tuning belongs in the outer loop, after the MVP is proven to run cleanly end-to-end.

---

## 10. Files delivered

- `pipeline.py` — the module implementing everything above (`engineer_features`, `impute_missing`, `assign_block`, `inner_loop`, `baseline_predict`, `make_lgbm_fit_predict`). Import it directly; don't copy-paste pieces out, so block logic and feature logic stay in one place as you iterate.
