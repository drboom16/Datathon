# Methodology report

**Task.** Forecast next-hour fine particulate matter (`PM2_5_next_hour`) at each of 12 monitoring stations. The competition score is RMSE.

**Final model.** A 50/50 average of two LightGBM regressors trained on the same feature set:

1. **Direct head (C)** — predicts next-hour PM2.5 in one step.
2. **Nowcast + change** — predicts the current level and the one-hour change, then adds them.

**Leaderboard file.** `submission.csv` in this folder. That is the exact prediction file under review.

**Local validation.** Average RMSE **27.61** on two chronological winter holdouts inside the labelled period (section 4).

## 1. Approach

Each row gives the pollutants, weather, wind, station, and clock available at the observation hour. Current PM2.5 is not among the released predictors.

Next-hour PM2.5 is mostly a slowly changing *level*, plus a smaller one-hour *change*. PM10 and CO, together with the same-hour readings from the other stations, carry most of the level. We therefore train:

- a direct model for the target, and
- a two-part model that first estimates the current level (training label = the previous hour’s known target on labelled rows) and then the change.

The two predictions are averaged. Every feature uses information available at time *t* or earlier.

## 2. Data cleaning and preprocessing

- Official `train.csv` (360,954 labelled station-hours; 2013-03-01 00:00 to 2016-08-31 22:00) and `test(1).csv` (51,063 station-hours; 2016-08-31 23:00 to 2017-02-28 22:00).
- Concatenate and sort by `station`, `observation_timestamp` so lags stay inside one station’s timeline.
- Store `station` and `wd` as pandas categoricals.
- Leave missing pollutant and weather values as missing. LightGBM handles them natively. Filling a station’s missing PM10 or CO from the network average helped one training winter and hurt the other, so it is not used.
- A small number of hours have PM10 missing at every station. Those predictions are filled from the same station’s nearest valid hour (forward fill, then backward fill), then from the direct head if still empty.

No rows are dropped.

## 3. Feature engineering

**Raw released columns.** PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, hour, month, station, wd.

**Time.** Hour and month sine/cosine, day of week, day of year, and a flag for the colder months (November–March), when particulate levels tend to run higher.

**Own-station dynamics.** Lags 1, 2, 3 for every pollutant (plus 6, 12, 24 for PM10 and CO); 3-hour tendency; 6- and 24-hour rolling mean/std/max and a 6-hour EWMA on PM10 and CO; first differences; PM10 acceleration.

**Ratios and weather.** CO/PM10, PM10 / (wind speed + 0.5), temperature minus dew point, 3-hour dew-point and pressure changes, 6-hour mean wind speed.

**Spatial (same timestamp, all 12 stations).** Network mean/max/std of each pollutant; leave-one-out network PM10; station minus network PM10/CO; one-hour change in the network PM10/CO means.

Tried during search and **not** used in the shipped model: network-average imputation, missingness flags, extra calendar-event flags, seasonal sample weights, and raw neighbour-station columns.

## 4. Validation strategy

The official split is chronological, so random cross-validation would look too optimistic. We score later winters *inside the labelled period*:

| Fold | Train | Validate |
|---|---|---|
| w14 | timestamps `< 2014-09-01` | 2014-09-01 → 2015-03-01 |
| w15 | timestamps `< 2015-09-01` | 2015-09-01 → 2016-03-01 |

Keep rule used while searching: both folds improve, or the two-winter average drops by at least 0.15 RMSE and neither fold gets worse.

**Holdout RMSE (labelled data only)**

| Head | w14 | w15 | average |
|---|---|---|---|
| Direct (C-style) | 26.81 | 28.85 | 27.83 |
| Nowcast + change | 26.75 | 28.84 | 27.79 |
| 50/50 blend (locked on this holdout) | — | — | **27.61** |

50/50 was the best blend weight on this holdout among `{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0}`.

`python train_submit.py --validate` reruns the same two folds with one seed. Those single-seed numbers will sit close to the table, not match it exactly.

## 5. Models tested and final selection

| Model | Holdout / note | Shipped? |
|---|---|---|
| Global mean | much worse | no |
| Released columns, no spatial features | worse than C | no |
| LightGBM C: spatial + deeper PM10/CO, 5 seeds | avg 27.83 | yes, head 1 |
| Nowcast + change, 3 seeds | avg 27.79 | yes, head 2 |
| **0.5 C + 0.5 (nowcast + change)** | **27.61** | **yes** |
| CatBoost / XGBoost / Huber loss | worse holdout | no |
| Winter-only training | worse holdout | no |
| Quantile 0.6 objective | worse holdout | no |
| Extra event flags and seasonal sample weights | mixed; did not generalise across both winters | no |
| Hand-written event multipliers | worse holdout | no |

Final selection: the 50/50 ensemble. The two heads are correlated but not identical, and the average was the best holdout score.

## 6. Ensembling and post-processing

- **Ensemble.** Arithmetic mean, weight 0.5 / 0.5. No stacking.
- **Post-processing.** Clip at 0. Station-wise forward/backward fill on hours where PM10 is missing at every station. Any remaining hole uses the direct head, then the prediction mean.
- No manual row edits.

## 7. Key results and observations

- Own-station PM10 is the strongest single released signal; the network PM10 field is close behind. Trees beat a simple multiple of PM10 by using the spatial field, short history, and weather.
- Spatial features and the deeper PM10/CO block improved both winter holdouts.
- Gadgets that only helped one training winter usually failed on the other. The shipped feature set is the one that held up on both folds.
- RMSE is dominated by the highest-PM10 hours.

## 8. Limitations

- Current PM2.5 is not released, so the model cannot use the most direct persistence signal.
- The one-hour change is only weakly related to one-hour changes in the other pollutants.
- A later winter can still differ from the two holdout winters.
- Retraining LightGBM can change the CSV at floating-point noise. The reviewed file is the frozen `submission.csv`.

## 9. Processed data

No extra processed dataset is required. Features are built in memory from the official CSVs. Optional intermediate direct-head predictions from the frozen blend are stored at `artifacts/submission_C.csv` (`train_submit.py` retrains that head).

## 10. Final model settings

| Item | Value |
|---|---|
| Library | LightGBM `LGBMRegressor`, trained from scratch |
| Direct head | 5 seeds `42, 7, 2024, 99, 2026`; `n_estimators=250` |
| Nowcast / change heads | 3 seeds `42, 7, 2024`; 2000 and 764 trees |
| Shared hyperparameters | `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=30`, `subsample=0.8`, `subsample_freq=1`, `colsample_bytree=0.8` |
| Ensemble | `0.5 * direct + 0.5 * (nowcast + change)` |

## 11. Official leaderboard score

Not recorded in this package. Please add the score shown on the competition site for `submission.csv` if you have it.
