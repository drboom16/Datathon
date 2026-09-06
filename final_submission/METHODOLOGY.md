# Methodology report

**Task.** Forecast next-hour fine particulate matter (`PM2_5_next_hour`) at each of 12 monitoring stations. The competition score is RMSE.

**Final model.** A 50/50 average of two LightGBM regressors trained on the same feature set:

1. **Direct head (C)** — predicts next-hour PM2.5 in one step.
2. **Nowcast + change** — predicts the current level and the one-hour change, then adds them.

**Leaderboard file.** `submission.csv` in this folder. That is the exact prediction file under review.

**Local validation.** Average RMSE **27.61** on two chronological winter holdouts (section 4). Full holdout / leaderboard log: [`SCORES.md`](SCORES.md).

## 1. Approach

Each row gives the pollutants, weather, wind, station, and clock available at the observation hour. Current PM2.5 is not among the released predictors.

External research and EDA pointed at the same picture. Next-hour PM2.5 is mostly a slowly changing *level* (city-wide haze, heating, inversions), plus a smaller one-hour *change*. PM10 is physically a superset of PM2.5; CO traces combustion. Same-hour readings from the other eleven stations say whether an event is regional or local. Weather and calendar set the background (coal heating, clean NW vs dirty SE wind, Lunar New Year fireworks).

We therefore train:

- a direct model for the target, and
- a two-part model that first estimates the current level (training label = the previous hour’s known target on labelled rows) and then the change.

The two predictions are averaged. Every feature uses information available at time *t* or earlier. No current-hour PM2.5 is used as a model input.

## 2. Data cleaning and preprocessing

- Official `train.csv` (360,954 labelled station-hours; 2013-03-01 00:00 to 2016-08-31 22:00) and `test(1).csv` (51,063 station-hours; 2016-08-31 23:00 to 2017-02-28 22:00).
- Concatenate and sort by `station`, `observation_timestamp` so lags stay inside one station’s timeline.
- Store `station` and `wd` as pandas categoricals.
- No rows are dropped. The target is right-skewed (mean 78, median 55, skew 2.01, 99th percentile 354). RMSE is owned by that tail, so removing extremes would throw away the hours that matter most.

**Missing values (EDA).** Eleven columns have gaps. CO is worst (15,832 rows, 4.4%), then O3, NO2, SO2, PM10 (0.5%). Weather gaps are rare (~0.05%). CO missingness hits all 12 stations but is not random: Dongsi alone holds ~17.6% of CO gaps, and the top five year-months — all in 2013 — hold 54.6% of them. Gaps clump by station and era.

That argued against a global mean fill. A station/month/hour average would respect the clumps; a short forward-fill (2–3 hours) would use hour-to-hour autocorrelation. We tried filling missing PM10/CO from the network average. It helped one training winter and **hurt the other**, so it is not used. Missing values stay missing. LightGBM handles them natively.

**Sensor ceilings (EDA + external notes).** PM10 tops out at 999 — NASA’s Earth Observatory describes 999 as the official monitoring-scale ceiling, not a true reading. CO tops out at 10,000 (sensor cap). The target hits 999 only once (February 2016). We tried 0/1 cap flags. They almost never fire (2 PM10 rows, 43 CO rows) and did not change RMSE, so they are not in the shipped model. The raw capped values stay in; trees can split on them if needed.

A small number of hours have PM10 missing at *every* station. Those predictions are filled from the same station’s nearest valid hour (forward fill, then backward fill), then from the direct head if still empty.

## 3. Feature engineering

All features come from released columns only. The list below is what we shipped, and *why* — from EDA, Beijing-specific research, or both. Tried-and-dropped items are at the end of this section.

### 3.1 Raw released columns

PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, hour, month, station, wd.

Linear correlation missed the cyclical hour and month patterns (midday low ~72, evening peak ~86; winter months ~94–96 vs August ~53). Trees can split on raw hour/month; we still add sine/cosine so 23:00 sits next to 00:00 and December next to January.

### 3.2 Spatial field (same hour, all 12 stations)

This was the single largest modelling gain.

Research: Beijing sits on a plain, mountains to the north and west, industrial Hebei to the south and east. Winter high pressure often lids the basin (temperature inversion). A dirty hour is usually a *city* event, not one rogue sensor.

EDA: central urban sites run highest (Dongsi 83.7, Nongzhanguan 82.7); northern suburban/background sites run lowest (Dingling 65.2, Huairou 68.9). The gap is geography, not noise.

For each pollutant we compute, at each timestamp: network mean, max, and standard deviation; a leave-one-out city mean; the station-minus-city gap; and the one-hour change in the city mean. That tells the model whether the hour is city-wide or local.

Own-station PM10 remains the strongest single released column. `city_PM10_mean` and `city_CO_mean` sit just behind it. On an earlier expanding-window search, adding this block was the jump from about **28.5 → 26.2**. Both winter holdouts in section 4 also improved. In plain English: next-hour PM2.5 at one station is mostly “how filthy is the city this hour.”

### 3.3 Own-station dynamics (backward only)

Current PM2.5 is not released, so the next-best persistence signal is the recent path of *released* pollutants — especially PM10 (coarse + fine mass) and CO (combustion / heating / traffic).

Per station, time-sorted, `shift` then roll (never the future, never the target):

- lags 1, 2, 3 on every pollutant; 6, 12, 24 on PM10 and CO
- 3-hour tendency; first differences (`PM10_d1`, `CO_d1`) and PM10 acceleration
- 6- and 24-hour rolling mean / std / max and a 6-hour EWMA on PM10 and CO

These help. They are **not** a substitute for current PM2.5. The nowcast head (below) is how we use the missing level as a *label*, not as a leaked input.

### 3.4 Weather, humidity, and wind

Research and the data agreed on the mechanisms; they disagreed on how much linear correlation you should expect.

- **Wind speed.** Low wind lets pollution pile up; strong wind vents the city. Literature treats WSPM as a dominant meteorological factor. In our table, WSPM correlated **−0.28** with the target. We keep raw WSPM, `PM10 / (WSPM + 0.5)`, and a 6-hour wind-speed mean.
- **Humidity / stagnation.** `TEMP − DEWP` (`tempdew_depr`) is a near-saturation proxy. Small spread means moist, still air — the physical setting for secondary particle growth and haze. We also keep 3-hour dew-point and pressure changes (inversion / system passage).
- **Rain.** Washout is real, but rain is rare and brief here. Correlation was **−0.03**. We keep RAIN and do not build a family of rain features around a tiny average effect.
- **Wind direction.** `wd` is not 16 arbitrary labels. NW / NNW air crosses the mountains and is the cleanest; S / SE draws in the industrial corridor. We pass `wd` as a native LightGBM categorical so the tree can learn that order without forcing a linear rank.

### 3.5 Calendar and secular trend

- **Heating.** Northern China’s centralised heating is roughly mid-November to mid-March and was historically coal-fired. The test window (Sep 2016 → Feb 2017) sits mostly inside that season. EDA already showed winter months ~40 µg/m³ dirtier than August. We ship a November–March flag plus month sine/cosine — the trees can still split on raw month if the flag is too coarse.
- **Diurnal cycle.** Evening/late-night peak, afternoon trough. Hour sine/cosine.
- **2013–2017 Clean Air Campaign.** The January 2013 “Airpocalypse” (PM2.5 past 1,000 µg/m³) triggered factory shutdowns, boiler upgrades, and tighter vehicle standards. 2013 air is dirtier than 2016 air for reasons that are not weather. The test set is the cleaner end of the record. Day-of-year (and the year implicit in the time index) lets a tree represent that slow drift.
- **Day of week.** Traffic / weekend-ozone proxy.

**Lunar New Year (researched, not shipped as a flag).** Fireworks on Lunar New Year’s Eve can take PM2.5 from healthy to hazardous in a few hours (2017 peak reported ~647). Dates in span: 10 Feb 2013, 31 Jan 2014, 19 Feb 2015, 8 Feb 2016, and **28 Jan 2017 (inside the test)**. That spike is cultural, not meteorological. A 0/1 flag is ~30 hours per year; trees ignored it (same fate as the heating flag and a missing-PM10 flag). Hand-written multipliers on the full 16-day legal window also failed: days 1–13 after New Year are already well calibrated, and the 2017 Lantern Festival looks clean in the released sensors. Those flags are not in the shipped model.

**Other calendar interventions (not shipped).** APEC Blue (Nov 2014) and Parade Blue (Sep 2015) are real, huge, and *only in train*. Red alerts (Dec 2015 in train; Dec 2016 / Jan 2017 in test) shut factories and take cars off the road — they can *lower* the next hour relative to a naive “filthy city → go higher” rule. Extra event flags and seasonal sample weights did not hold on both holdout winters.

**Spring dust (researched, not a separate shipped block).** Gobi storms (mainly March–April) inflate PM10 more than PM2.5 (PM10 at the 999 cap, PM2.5 far lower). That is why a blanket “high city PM10 → raise the forecast” rule is unsafe: some high-PM10 hours are mineral dust, not haze. The spatial field plus CO/PM10 already lets the tree see “high PM10, low combustion.”

### 3.6 Station as a categorical

EDA ranks match the map: inner-city traffic sites (Dongsi, Nongzhanguan, Wanshouxigong, Tiantan, Guanyuan) sit above western/industrial (Gucheng, Wanliu) and well above the northern background sites (Dingling, Huairou, Changping, Shunyi). `station` is a native categorical so each site can have its own baseline without us hard-coding the ranking.

### 3.7 Nowcast labels (training only)

The nowcast head’s targets — a “current level” and an hour-to-hour “delta” — are the competition label shifted by one hour *inside the training set*. At test time both heads see only released features. No test-set target is read. That is how we use the missing current PM2.5 without leaking it as an input.

### 3.8 Tried during search and not shipped

Network-average imputation; missingness flags; Lunar New Year / red-alert / APEC / Parade flags; seasonal sample weights; raw neighbour-station columns; log1p and Tweedie objectives; winter-only training.

## 4. Validation strategy

The official split is chronological, so random cross-validation would look too optimistic. The live test is autumn–winter 2016/17 — the same season as Beijing’s heating cycle and the documented Dec 2016 / Jan 2017 haze episodes — so we score later *winters inside the labelled period*:

| Fold | Train | Validate |
|---|---|---|
| w14 | timestamps `< 2014-09-01` | 2014-09-01 → 2015-03-01 |
| w15 | timestamps `< 2015-09-01` | 2015-09-01 → 2016-03-01 |

Keep rule used while searching: both folds improve, or the two-winter average drops by at least 0.15 RMSE and neither fold gets worse. Gadgets that only helped one winter were discarded (section 7).

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
| Tweedie / log1p(target) | worse; log1p fought the tail RMSE is scored on | no |
| Extra event flags and seasonal sample weights | mixed; did not generalise across both winters | no |
| Hand-written event multipliers (LNY window, dirty-night floors) | worse holdout / failed to generalise | no |

Final selection: the 50/50 ensemble. The two heads are correlated but not identical, and the average was the best holdout score.

## 6. Ensembling and post-processing

- **Ensemble.** Arithmetic mean, weight 0.5 / 0.5. No stacking. Seed averages inside each head (5 seeds on C, 3 on nowcast + change) reduce LightGBM jitter.
- **Post-processing.** Clip at 0. Station-wise forward/backward fill on hours where PM10 is missing at every station. Any remaining hole uses the direct head, then the prediction mean.
- No manual row edits.

## 7. Key results and observations

RMSE is dominated by the highest-PM10 / highest-true-PM2.5 hours. Typical hours are already well calibrated; the score is a fat tail.

### What worked

- **City-wide PM10/CO** was the real jump (about 28.51 → 26.24 on the expanding-window search; both winter holdouts also moved). One station’s next-hour PM2.5 is mostly how filthy the city is this hour.
- **Lags and rolls of released pollutants** help. They are not a substitute for current PM2.5.
- **Nowcast + Δ** is a clean way to use the missing current PM2.5 as a *label*, not a leak.
- **Seed averaging** knocks down LightGBM run-to-run jitter.
- **Leaving NaNs alone** is correct. Forcing a city-mean into missing-PM10 rows hurt a later fold (round 4 +1.08 on that protocol) and failed the “both winters” keep rule here.

### What failed, and stayed out

- **Winter sample weights.** Better on some CV numbers, worse on the public leaderboard — the model over-chased the tail and gave up ordinary hours.
- **Winter-only training.** Worse on winter *and* on the rest of the year. Summer hours still teach the PM10 / CO → PM2.5 mapping.
- **Tweedie and log1p.** The metric is raw-scale RMSE; transforming the target made the tail worse.
- **More leaves / heavier row-column sampling** without a hold on the later winter. Mean could dip while the test-like fold rose.
- **Tree flags** for Lunar New Year, heating, and missing PM10. Rare bits get ignored. The heating *season* is already in month and the Nov–Mar flag; a 30-hour LNY bit is not.
- **Flooring every filthy winter hour** to `k × city_PM10`. Wrecks the fold that contains spring-like high-PM10 / low-PM2.5 dust. High PM10 is not always haze.
- **Boosting the whole 16-day legal firework window.** Days 1–13 after New Year are already calibrated; the 2017 Lantern Festival is clean in the test sensors. The cultural spike is a few midnight hours, not a fortnight.

Gadgets that only helped one training winter usually failed on the other. The shipped feature set is the one that held up on both folds.

## 8. Limitations

- Current PM2.5 is not released, so the model cannot use the most direct persistence signal.
- The one-hour change is only weakly related to one-hour changes in the other pollutants.
- A later winter can still differ from the two holdout winters. The live test includes documented red-alert / New Year haze (Dec 2016 – Jan 2017) and Lunar New Year (28 Jan 2017). Those episodes show up in the *released* PM10/CO field; we do not hard-code their dates.
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

Public RMSE for the frozen file `submission.csv`: **23.89161** (uploaded as `submission.csv`, account andrew_c105).

Earlier legal uploads on the same board (see `SCORES.md` for the full log):

| Upload | Public RMSE |
|---|---|
| `submission.csv` (this package) | **23.89161** |
| `submission_cm.csv` (C only) | 24.23269 |
| `submission_v8.csv` | 26.94941 |
| `submission_v8_events.csv` | 27.31382 |
| `submission_v8_part2.csv` | 27.46496 |
