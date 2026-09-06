# Research Log

## Iteration 1

- Date/time: 2026-09-05 22:47:51
- Model: baseline_station_mean
- Description: Baseline: predict each station's own training-set mean, no features.
- CV avg RMSE: 79.1611
- Per-block RMSEs: 83.1 / 80.6 / 74.8 / 78.1
- Keep?: 
- Notes:

## Iteration 2

- Date/time: 2026-09-05 23:54:23
- Model: lightgbm_v1
- Description: LightGBM on pollutants (PM10,SO2,NO2,CO,O3) + weather (TEMP,PRES,DEWP,RAIN,WSPM) + engineered: is_heating_season, time_index, wind_is_clean, hour_sin/cos, month_sin/cos, is_lunar_new_year, raw hour/month, station (categorical). Leak-safe chronological inner early stopping.
- CV avg RMSE: 30.1057
- Per-block RMSEs: 40.5 / 27.0 / 24.7 / 28.2
- Keep?: 
- Notes: 
  - **Round 1 is the outlier, and Cursor's explanation is right.** With only block 0 (Mar–Nov 2013, less than a full year), the model has never seen a deep winter, so it predicts block 1's winter poorly. Gap of 14.3. This isn't overfitting in the dangerous sense — it's *too little history*, which fixes itself as blocks accumulate
  - **The one Cursor undersold — round 4 matters most and it's not your best.** Round 4 validates on block 4, which is **Dec 2015 → Aug 2016** — the most recent, most test-like slice. Its val RMSE is 28.2, worse than round 3's 24.7. That's a mild warning that on the *latest* data (closest to your real test) you're doing a touch worse than mid-record.
  - Unusual values -> caps weren't handled.

## Iteration 2A

- Date/time: 2026-09-06 00:14:47
- Model: lightgbm_v2
- Description: v1 + PM10_was_capped + CO_was_capped flags (only change vs v1).
- CV avg RMSE: 30.1057
- Per-block RMSEs: 40.5 / 27.0 / 24.7 / 28.2
- Keep?: NO (REVERT)
- Notes:
  - Extend make_features (or make a make_features_v2 that calls the original) to add TWO row-local flag columns: PM10_was_capped : 1 if PM10 == 999, else 0 CO_was_capped : 1 if CO == 10000, else 0
  - NASA's Earth Observatory, describing Beijing's official monitoring, notes PM10 readings hitting 999 — the maximum threshold for the indicator.
  - v2's RMSE is identical to v1's, to 4 decimals (mean 30.1057). That's not a bug in the wiring — it's because the flags almost never fire:
    - `PM10 == 999`: 2 rows / 360,954
    - `CO == 10000`: 43 rows / 360,954
  - Revert back to before

## Iteration 3

- Date/time: 2026-09-06 01:10:10
- Model: lightgbm_v3
- Description: v1 + 4 interaction features: PM10_per_WSPM, CO_per_WSPM, temp_dewp_spread, PM10_x_wind_clean (only change vs v1)
- CV avg RMSE: 29.0426
- Per-block RMSEs: 37.1 / 27.0 / 24.3 / 27.7
- Keep?: YES
- Notes:
  - PM10/WPSM
    - PM10 measures coarse dust and combustion particles. When this ratio is high, it means high particulate volume is paired with dead-calm air. It identifies stagnant smog domes where local dust and soot are trapped near ground level.
  - CO/WPSM
    - Carbon Monoxide (CO) is a direct, non-reactive tracer of primary combustion (traffic exhaust and coal-heating boilers). Because CO dilutes directly with wind speed, dividing CO by WPSM measures how quickly local combustion emissions are diluting into fresh air. A high value means emissions are overwhelming the ventilation capacity of the city.
  - TEMP - DEWP
    - Direct proxy for relative humidity/moisture; small spread means near 100% humidity, which fuels secondary particle growth (fog/smog formation).
  - PM10 x wind_is_clean 
    - An interaction between your best feature and your geography flag. The story: the *same* PM10 reading might mean different things depending on whether clean or dirty air is blowing in. This combines pollution with direction — different domains, which is the good kind.

## Iteration 4

- Date/time: 2026-09-06 01:19:59
- Model: lightgbm_v4
- Description: v3 minus near-dead features; testing if leaner holds ~29
- CV avg RMSE: 28.9637
- Per-block RMSEs: 37.1 / 27.0 / 24.2 / 27.5
- Keep?: YES
- Notes: Removed some noise from dead features and such

## Iteration 5

- Date/time: 2026-09-06 01:22:04
- Model: lightgbm_lr003
- Description: champion recipe, learning_rate 0.05 -> 0.03 (only change)
- CV avg RMSE: 28.8369
- Per-block RMSEs: 36.8 / 26.9 / 24.3 / 27.4
- Keep?: Only learnign rate = 0.03
- Notes:


## Iteration 6
- Date/time: 2026-09-06 01:41:23
- Model: lightgbm_v5
- Description: v4 + per-station backward lags/rolls of PM10/CO/NO2/WSPM (lag1-3, roll3); computed per-slice inside fold (fold-safe), predictors only, never target
- CV avg RMSE: 28.5064
- Per-block RMSEs: 37.3 / 26.2 / 23.5 / 27.0
- Keep?: 
- Notes: 
