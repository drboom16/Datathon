# Final submission — next-hour PM2.5

This folder is the package required by *Expected Submission Materials* (deadline 6 September 2026). It is the pipeline that produced the leaderboard file being reviewed.

Upload **this folder only**. It is self-contained.

## What to upload to the competition

**`submission.csv`** — exact prediction file (`id`, `PM2_5_next_hour`), 51,063 rows. This is the frozen leaderboard file. Do not replace it with a retrain unless you clearly say so.

**Official public RMSE for this file: 23.89161.**  
Local two-winter holdout average: **27.61**. Full tables (every legal upload + ablations): [`SCORES.md`](SCORES.md).

## Files in this folder

| File | Role |
|---|---|
| `SCORES.md` | Local winter holdouts, expanding-window CV, official leaderboard log |
| `DISCLOSURE.md` | External data, code, AI tools, post-processing |
| `train_submit.py` | Validation (optional), training, inference, submission file |
| `features.py` | Data load + feature generation |
| `submission.csv` | Exact leaderboard predictions |
| `artifacts/submission_C.csv` | Optional intermediate direct-head predictions |
| `requirements.txt` | Python packages |

Competition CSVs are not copied here. Put `train.csv`, `test(1).csv` (or `test.csv`), and `sample_submission.csv` in `final_submission/data/`, or pass `--data-dir`.

## How to reproduce

```bash
cd final_submission
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Frozen leaderboard file (use this unless you intend to retrain):
#   submission.csv

# Optional: time-aware winter holdouts on the labelled data
python train_submit.py --validate --out submission_reproduced.csv

# Retrain from the official files (minutes; small float noise vs the frozen file)
python train_submit.py --out submission_reproduced.csv
```

`--data-dir /path/to/official/csvs` overrides the data location.

Python 3.10+ is recommended.

### Execution order

1. Load `train.csv` + `test(1).csv`
2. Feature generation (`features.add_features`)
3. Optional: two chronological winter holdouts (`--validate`)
4. Train the direct head (5-seed LightGBM, 250 trees)
5. Train nowcast + change (3-seed LightGBM, 2000 / 764 trees)
6. Blend 50/50
7. Station-wise fill on hours with no PM10 anywhere in the network
8. Write predictions

### Seeds and settings

- Direct-head seeds: `42, 7, 2024, 99, 2026` — `n_estimators=250`
- Nowcast + change seeds: `42, 7, 2024` — nowcast `2000` trees, change `764` trees
- LightGBM: `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=30`, `subsample=0.8`, `colsample_bytree=0.8`
- Ensemble weight: **0.5 direct + 0.5 (nowcast + change)**

## Final model (short)

Two LightGBM regressors on the same released-column feature set, averaged:

1. **Direct** — predict `PM2_5_next_hour`
2. **Nowcast + change** — predict the current level (training label = previous row’s target) plus the one-hour change, then add

Details and holdout numbers are in `METHODOLOGY.md`.
