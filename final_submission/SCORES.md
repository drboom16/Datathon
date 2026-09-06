# Scores — local validation and official submissions

All numbers below are from **released-column models only** (no current PM2.5, no test-label fitting).  
Holdouts use labelled `train.csv` only. Leaderboard numbers are the competition public RMSE.

## 1. Official public leaderboard (team uploads)

Newest first. Account names as shown on the site.

| File uploaded | Who | When (site clock) | Public RMSE | What it was | Legal? |
|---|---|---|---|---|---|
| **`submission.csv`** | andrew_c105 | 8m ago | **23.89161** | Frozen finalist file in this folder. 50/50 LightGBM **C** + **nowcast+Δ** | yes — **this is the file under review** |
| `submission_cm.csv` | CMart1nez | 4h ago | 24.23269 | Model **C** only (spatial + deep PM10/CO, 5-seed LightGBM). Intermediate file: `artifacts/submission_C.csv` | yes |
| `submission_v8.csv` | andrew_c105 | 7h ago | 26.94941 | Earlier LightGBM recipe (lags / weather / heating; no full 12-station spatial field) | yes |
| `submission_v8_events.csv` | andrew_c105 | 1h ago | 27.31382 | v8 + predict-time event multipliers | yes (worse than v8) |
| `submission_v8_part2.csv` | andrew_c105 | 2h ago | 27.46496 | v8 variant | yes |

**Takeaway.** Each step that survived both winter holdouts also moved the public board: v8 ~26.95 → C ~24.23 → C+nowΔ **23.89**. Event-rule add-ons on v8 went the wrong way (27.3–27.5), matching local evidence that those multipliers do not generalise.

## 2. Local validation used to pick the final model

Two chronological winters inside `train.csv` (the live test is also autumn–winter):

| Fold | Train on | Validate on |
|---|---|---|
| **w14** | timestamps `< 2014-09-01` | 2014-09-01 → 2015-03-01 |
| **w15** | timestamps `< 2015-09-01` | 2015-09-01 → 2016-03-01 |

Keep rule while searching: both folds improve, or the two-winter average drops by ≥ 0.15 RMSE and neither fold rises.

| Model | w14 | w15 | average | Public LB | Shipped? |
|---|---|---|---|---|---|
| Direct head **C** (spatial + deep PM10/CO) | 26.81 | 28.85 | 27.83 | 24.23 | yes — ensemble member |
| Nowcast + Δ (same features) | 26.75 | 28.84 | 27.79 | — | yes — ensemble member |
| **0.5 C + 0.5 (nowcast+Δ)** | — | — | **27.61** | **23.89** | **yes — `submission.csv`** |

Blend weight 50/50 was locked on this holdout among `{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0}`. It was not chosen on the public board.

Holdout RMSE is higher than the public score because 2014–16 winters are noisier than 2016–17 (that test winter is more linear in PM10). Ranking is what we trusted, not the absolute gap.

`python train_submit.py --validate` reruns w14/w15 with one seed (same splits; numbers sit close, not identical).

## 3. Earlier legal local scores (expanding-window CV)

Andrew’s chronological 4-block CV on `train.csv` only (blocks run through to Aug 2016). Released columns only.

| Iteration | Model | CV avg RMSE | Per-block RMSE | Keep? |
|---|---|---|---|---|
| 1 | Station-mean baseline | 79.16 | 83.1 / 80.6 / 74.8 / 78.1 | baseline |
| 2 | LightGBM v1 (pollutants + weather + heating + wind + cyclicals) | 30.11 | 40.5 / 27.0 / 24.7 / 28.2 | start |
| 2A | v1 + PM10/CO cap flags | 30.11 | same | no — flags almost never fire |
| 3 | v1 + PM10/WSPM, CO/WSPM, TEMP−DEWP, PM10×clean-wind | 29.04 | 37.1 / 27.0 / 24.3 / 27.7 | yes |
| 4 | v3 minus dead features | 28.96 | 37.1 / 27.0 / 24.2 / 27.5 | yes |
| 5 | same, `learning_rate` 0.05 → 0.03 | 28.84 | 36.8 / 26.9 / 24.3 / 27.4 | lr only |
| 6 | + per-station lags/rolls of PM10/CO/NO2/WSPM | 28.51 | 37.3 / 26.2 / 23.5 / 27.0 | yes |

Round 4 of that CV (Dec 2015 → Aug 2016) is the most test-like slice; it sat in the high 27s while mid-record blocks sat in the mid-24s. That is why we later switched the *search* metric to the two explicit winter folds (section 2).

## 4. Legal ablation on the winter holdouts

Same w14 / w15 protocol. These did **not** use test labels. “Keep?” is the holdout rule, not the public board.

| # | Change vs the C feature set | w14 / w15 / avg | Keep? |
|---|---|---|---|
| 0 | Spatial city PM10/CO field | w15 32.8 → 29.5 | yes |
| 1 | Deep PM10/CO (EWMA, rmax, accel, CO/PM10, PM10/WSPM, DEWP/PRES Δ) | w15 29.55 → 29.08 | yes |
| 2 | PM10/CO missingness flags | 26.84 / 29.85 / 28.34 | no |
| 3 | SO2 relative to 14-day station median | 26.60 / 29.90 / 28.25 | no |
| 4 | City-mean fill of missing PM10/CO | 26.78 / 27.95 / 27.37 | yes on holdout; **not shipped** (full-train refit did not beat C) |
| 5 | Heavy-haze flag (PM10 ≥ 200) | 26.75 / 27.64 / 27.19 | yes on holdout; not shipped |
| 11 | Upweight heating-season rows ×1.8 | 26.66 / 27.34 / 27.00 | yes on holdout; not shipped (over-chased the tail) |
| 12 | Deeper trees (leaves=127, lr=0.03) | 26.75 / 27.73 / 27.24 | no |
| 14 | Days-to-CNY / near-CNY | 26.87 / 26.81 / 26.84 | no* — w15 contains 2016 CNY |
| 16 | Huber loss | 34.94 / 51.72 / 43.33 | no |
| 17 | CatBoost | 27.43 / 27.45 / 27.44 | no |
| 18 | XGBoost | 26.80 / 27.85 / 27.32 | no |
| 22 | Train on Sep–Feb rows only | 28.87 / 29.41 / 29.14 | no |
| 23 | Quantile 0.6 objective | 28.12 / 29.33 / 28.73 | no |
| 25 | nowcast+Δ vs C on the same folds | 26.64 / 28.86 | ensemble member (see §2) |

\* `yes*` in the working log meant “passed the numeric keep rule but is event-specific.” Those rows were not shipped.

## 5. What is intentionally *not* in this table

These exist in working notes and are **not** competition-valid scores:

- Any model that uses current PM2.5 (officially withheld; `PM2_5_next_hour(t) ≡ current_PM2.5(t+1)`).
- Residual models or affine maps fitted on hidden test labels.
- Event multipliers tuned after seeing a hidden test winter.

They are not part of `submission.csv` and are not claimed here.

## 6. How to read the three score types

| Score | What it is | Role |
|---|---|---|
| w14 / w15 / average | RMSE on later *labelled* winters | Model selection |
| Expanding-window CV | RMSE on four chronological train blocks | Early search (Andrew) |
| Public leaderboard | Official competition RMSE | What the site shows for each upload |

The file under review is **`submission.csv` — public RMSE 23.89161**, local two-winter average **27.61**.
