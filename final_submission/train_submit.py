"""
Train the final model and write a submission CSV.

Steps
  1. Load the official train and test files
  2. Build features from the columns released at each hour
  3. Optionally score two chronological winter holdouts on the labelled data
  4. Train two LightGBM heads and average them 50/50
  5. Fill any remaining holes and write the prediction file

`submission.csv` in this folder is the exact file uploaded to the leaderboard.
A retrain can differ by a little floating-point noise.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from features import Y, add_features, find_data_dir, load_competition_frames

C_SEEDS = [42, 7, 2024, 99, 2026]
ND_SEEDS = [42, 7, 2024]
C_TREES = 250
NOWCAST_TREES = 2000
DELTA_TREES = 764
BLEND_C = 0.5

# Shared LightGBM settings for every head and seed.
LGB = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    n_jobs=-1,
    verbose=-1,
)

# Later winters inside the labelled period — closer in spirit to a chronological test.
HOLDOUTS = [
    ("w14", "2014-09-01", "2015-03-01"),
    ("w15", "2015-09-01", "2016-03-01"),
]


def fit_lgb(X, y, n_estimators, seed):
    model = lgb.LGBMRegressor(**{**LGB, "n_estimators": n_estimators, "random_state": seed})
    model.fit(X, y)
    return model


def rmse(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def predict_heads(train, test, feat, c_seeds, nd_seeds):
    """Fit the direct head and the nowcast+change head, then return both test predictions."""
    preds_c = []
    for seed in c_seeds:
        print(f"  direct head, seed {seed}")
        model = fit_lgb(train[feat], train[Y].values, C_TREES, seed)
        preds_c.append(np.clip(model.predict(test[feat]), 0, None))
    pred_c = np.mean(preds_c, axis=0)

    # The nowcast target is "what was the level at this hour?". On labelled rows
    # that is just the previous hour's next-hour target.
    has_now = train[train.now_y.notna()]
    has_delta = train[train.delta_y.notna()]
    preds_nd = []
    for seed in nd_seeds:
        print(f"  nowcast + change, seed {seed}")
        nowcast = fit_lgb(has_now[feat], has_now.now_y.values, NOWCAST_TREES, seed)
        delta = fit_lgb(has_delta[feat], has_delta.delta_y.values, DELTA_TREES, seed)
        preds_nd.append(np.clip(nowcast.predict(test[feat]) + delta.predict(test[feat]), 0, None))
    pred_nd = np.mean(preds_nd, axis=0)
    return pred_c, pred_nd


def run_validation(train, feat):
    """
    Time-aware check: train on earlier labelled rows, score a later winter.
    One seed each, so this is a little noisier than the multi-seed numbers
    in METHODOLOGY.md, but it uses the same model and the same folds.
    """
    print("time-aware validation (1 seed per head)")
    blend_scores = []
    for name, start, end in HOLDOUTS:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        tr = train[train.ts < start]
        va = train[(train.ts >= start) & (train.ts < end)]
        print(f"  {name}: train {len(tr):,}  validate {len(va):,}")
        pred_c, pred_nd = predict_heads(tr, va, feat, c_seeds=[42], nd_seeds=[42])
        pred = BLEND_C * pred_c + (1.0 - BLEND_C) * pred_nd
        y = va[Y].values
        score_c = rmse(y, pred_c)
        score_nd = rmse(y, pred_nd)
        score = rmse(y, pred)
        blend_scores.append(score)
        print(f"    direct {score_c:.3f}   nowcast+change {score_nd:.3f}   blend {score:.3f}")
    print(f"  mean blend RMSE {np.mean(blend_scores):.3f}")


def fill_missing_hours(test, pred, pred_c):
    """
    A few hours have almost no pollutant readings at any station.
    Carry the last good prediction along each station, then fall back to
    the direct head if a station is still empty.
    """
    tmp = test[["id", "station", "ts"]].copy()
    tmp["pred"] = pred
    tmp["pred_c"] = pred_c
    tmp = tmp.sort_values(["station", "ts"])
    tmp["pred"] = tmp.groupby("station")["pred"].ffill().bfill()
    tmp["pred"] = tmp["pred"].fillna(tmp["pred_c"])
    return tmp


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the final model and write a submission CSV.")
    parser.add_argument("--data-dir", type=Path, default=find_data_dir(here))
    parser.add_argument("--out", type=Path, default=here / "submission_reproduced.csv")
    parser.add_argument("--validate", action="store_true", help="Score the two winter holdouts before the full train.")
    args = parser.parse_args()

    print(f"data dir: {args.data_dir}")
    df = load_competition_frames(args.data_dir)
    df, feat = add_features(df)

    # now_y is the labelled current level; delta_y is the one-hour change.
    by_station = df.groupby("station", group_keys=False)
    df["now_y"] = by_station[Y].shift(1)
    df["delta_y"] = df[Y] - df["now_y"]

    train = df[df.is_test == 0].copy()
    test = df[df.is_test == 1].copy()
    print(f"features {len(feat)}  train {len(train):,}  test {len(test):,}")

    if args.validate:
        run_validation(train, feat)

    print("training on all labelled rows")
    pred_c, pred_nd = predict_heads(train, test, feat, C_SEEDS, ND_SEEDS)
    pred = BLEND_C * pred_c + (1.0 - BLEND_C) * pred_nd
    filled = fill_missing_hours(test, pred, pred_c)
    pred_map = dict(zip(filled["id"], filled["pred"]))
    fallback_c = dict(zip(filled["id"], filled["pred_c"]))

    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    sample["PM2_5_next_hour"] = sample["id"].map(pred_map)
    sample["PM2_5_next_hour"] = sample["PM2_5_next_hour"].fillna(sample["id"].map(fallback_c))
    sample["PM2_5_next_hour"] = sample["PM2_5_next_hour"].fillna(float(np.nanmean(pred)))
    if sample["PM2_5_next_hour"].isna().any():
        raise SystemExit("some rows are still missing a prediction")
    sample.to_csv(args.out, index=False)
    print(f"wrote {args.out}  rows={len(sample)}")


if __name__ == "__main__":
    main()
