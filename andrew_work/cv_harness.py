"""
Time-aware cross-validation harness for the Beijing air-quality task.

This file is ONLY the CV scaffolding. It ships with a dummy placeholder model
(predicts the training-set mean) purely so the harness runs end to end.
Replace `dummy_fit_predict` with a real baseline/model later WITHOUT touching
the harness functions.

Read-only on train.csv. The test file is read ONLY inside make_submission().
"""

import numpy as np
import pandas as pd

CSV = "inter-uni-datathon-stream-2-beijing-multi-site-air-quality/train.csv"
# On disk this is test(1).csv (the competition test set the brief called test_1_.csv).
TEST_CSV = "inter-uni-datathon-stream-2-beijing-multi-site-air-quality/test(1).csv"
SAMPLE_CSV = "inter-uni-datathon-stream-2-beijing-multi-site-air-quality/sample_submission.csv"
SUBMISSION_PATH = "submission_v1.csv"
TARGET = "PM2_5_next_hour"
TIME_COL = "observation_timestamp"

# Module-level collector for inner-train RMSEs (filled by lightgbm_v1 each round,
# reset by main() before every run so repeated runs don't stack stale scores).
_TRAIN_RMSES = []


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def rmse(actual, predicted):
    """RMSE = sqrt(mean((actual - predicted)^2))."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ---------------------------------------------------------------------------
# Data loading (read-only)
# ---------------------------------------------------------------------------
def load_data(csv_path=CSV):
    """Load train.csv, parse the timestamp, sort chronologically ascending."""
    df = pd.read_csv(csv_path, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL, ascending=True).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Block assignment (time-aware, no shuffling)
# ---------------------------------------------------------------------------
def assign_blocks(df, n_blocks=5):
    """
    Split rows into `n_blocks` contiguous chronological blocks, numbered
    0..n-1 oldest->newest.

    Blocks are cut on the int64 TIMESTAMP VALUE (not row position), so every
    row sharing the same hour (all 12 stations) always lands in the same block.

    Returns a NEW dataframe with an added integer 'block' column. Does not
    mutate the input.
    """
    out = df.copy()
    ts_int = out[TIME_COL].astype("int64")  # nanoseconds since epoch
    # labels=False -> integer bin codes ascending (0 = oldest timestamps).
    # duplicates='drop' guards against collapsed bin edges.
    out["block"] = pd.qcut(ts_int, q=n_blocks, labels=False, duplicates="drop")
    out["block"] = out["block"].astype(int)
    return out


def describe_blocks(df_blocked):
    """Print each block's start timestamp, end timestamp, and row count."""
    print("\nBlock summary (oldest -> newest):")
    print(f"{'block':>5} | {'start':<19} | {'end':<19} | {'rows':>8}")
    print("-" * 64)
    for b in sorted(df_blocked["block"].unique()):
        chunk = df_blocked[df_blocked["block"] == b]
        start = chunk[TIME_COL].min()
        end = chunk[TIME_COL].max()
        print(f"{b:>5} | {str(start):<19} | {str(end):<19} | {len(chunk):>8,}")


# ---------------------------------------------------------------------------
# Cross-validation (expanding window)
# ---------------------------------------------------------------------------
def time_series_cv(df, fit_predict_fn, n_blocks=5, verbose=True):
    """
    Expanding-window, time-aware CV.

    For round i (i = 1 .. n_blocks-1):
        train = all rows in blocks 0..(i-1)
        val   = rows in block i

    `fit_predict_fn(train_df, val_df) -> array of predictions for val_df`.

    LEAKAGE RULE: fit_predict_fn must compute everything (imputation, encoding,
    model fitting) using ONLY train_df. The harness enforces this: the val_df
    handed to the callback has its TARGET column blanked (NaN), so peeking is
    impossible. The true target is kept only inside the harness for scoring.

    Returns (per_round_rmses, mean_rmse).
    """
    df_blocked = assign_blocks(df, n_blocks=n_blocks)
    if verbose:
        describe_blocks(df_blocked)

    blocks = [int(b) for b in sorted(df_blocked["block"].unique())]
    per_round = []

    if verbose:
        print("\nCross-validation rounds (expanding window):")

    for i in range(1, len(blocks)):
        train_blocks = blocks[:i]
        val_block = blocks[i]

        train_df = df_blocked[df_blocked["block"].isin(train_blocks)].copy()
        val_df = df_blocked[df_blocked["block"] == val_block].copy()

        # Hand the callback a copy of val_df with the target BLANKED (NaN) so it
        # is physically impossible to peek at validation truth. The real target
        # is retained in val_df for scoring, which happens here in the harness.
        val_df_features = val_df.copy()
        val_df_features[TARGET] = np.nan

        # Callback fits on train_df only and returns preds for val_df features.
        preds = np.asarray(fit_predict_fn(train_df, val_df_features), dtype=float)
        if len(preds) != len(val_df):
            raise ValueError(
                f"fit_predict_fn returned {len(preds)} preds for "
                f"{len(val_df)} val rows in round {i}."
            )

        # Scoring happens HERE, using the held-out truth the callback never saw.
        score = rmse(val_df[TARGET].values, preds)
        per_round.append(score)

        if verbose:
            print(
                f"  Round {i}: train blocks {train_blocks} "
                f"({len(train_df):,} rows)  ->  val block [{val_block}] "
                f"({len(val_df):,} rows)   RMSE = {score:.4f}"
            )

    mean_rmse = float(np.mean(per_round)) if per_round else float("nan")
    return per_round, mean_rmse


# ---------------------------------------------------------------------------
# DUMMY PLACEHOLDER MODEL  (temporary - replace with a real model later)
# ---------------------------------------------------------------------------
def dummy_fit_predict(train_df, val_df):
    """
    TEMPORARY placeholder. Ignores all features and predicts the MEAN of
    train_df[TARGET] for every validation row. Exists only to prove the
    harness runs end to end.
    """
    train_mean = train_df[TARGET].mean()
    return np.full(len(val_df), train_mean, dtype=float)


# ---------------------------------------------------------------------------
# BASELINE MODEL: per-station training-set mean (no features)
# ---------------------------------------------------------------------------
def baseline_station_mean(train_df, val_df):
    """
    Predict each row's station's own training-set mean of the target.

    Stats come ONLY from train_df. Stations in val_df that were never seen in
    train_df fall back to the overall train mean. Never reads val_df[TARGET]
    (the harness blanks it anyway).
    """
    station_means = train_df.groupby("station")[TARGET].mean()
    overall_mean = train_df[TARGET].mean()
    preds = val_df["station"].map(station_means).fillna(overall_mean)
    return preds.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING (row-local, leakage-free)
# ---------------------------------------------------------------------------
# Winds from the northerly/north-westerly group tend to clear Beijing's air.
_CLEAN_WIND = {"N", "NNW", "NW", "NNE", "NE"}
_HEATING_MONTHS = {11, 12, 1, 2, 3}
# Hard-coded Lunar New Year DAYS (not derivable from the calendar columns).
_LUNAR_NEW_YEAR_DAYS = [
    "2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28",
]


def _lunar_new_year_mask(ts):
    """
    Boolean mask: True if timestamp `ts` (a datetime Series) falls in a Lunar
    New Year firework window. Window = [eve 18:00 .. LNY-day 23:59], where
    eve = LNY_day - 1 day. Covers both the NYE-midnight spike and New Year's
    Day evening fireworks. Row-local: depends only on each row's own timestamp.
    """
    mask = pd.Series(False, index=ts.index)
    for day in _LUNAR_NEW_YEAR_DAYS:
        lny = pd.Timestamp(day)
        start = (lny - pd.Timedelta(days=1)) + pd.Timedelta(hours=18)  # eve 18:00
        end = lny + pd.Timedelta(hours=23, minutes=59)                 # LNY 23:59
        mask |= (ts >= start) & (ts <= end)
    return mask


def make_features(df):
    """
    Return a COPY of df with engineered features added. Uses ONLY each row's
    own values (no stats computed across rows), so it is safe to apply to any
    train/val slice with zero leakage. Missing values are left as-is.
    """
    out = df.copy()

    out["is_heating_season"] = out["month"].isin(_HEATING_MONTHS).astype(int)

    # Months since dataset start (gives the model the 2013->2017 trend).
    ti = (out["year"] - 2013) * 12 + out["month"]
    out["time_index"] = (ti - ti.min()).astype(int)

    out["wind_is_clean"] = out["wd"].isin(_CLEAN_WIND).astype(int)  # NaN -> 0

    # Cyclical encodings so 23:00 sits next to 00:00, Dec next to Jan.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out["is_lunar_new_year"] = _lunar_new_year_mask(out[TIME_COL]).astype(int)

    # station as native LightGBM categorical.
    out["station"] = out["station"].astype("category")

    return out


# Columns fed to the model. Never includes id / observation_timestamp / raw wd /
# target / block / year / day.
FEATURES = [
    "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "hour", "month",
    "is_heating_season", "time_index", "wind_is_clean",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_lunar_new_year",
    "station",
]
CATEGORICAL = ["station"]


# ---------------------------------------------------------------------------
# MODEL: lightgbm_v1
# ---------------------------------------------------------------------------
def lightgbm_v1(train_df, val_df):
    """
    LightGBM regressor with row-local feature engineering and LEAK-SAFE early
    stopping.

    Early stopping uses a chronological inner split of train_df ONLY: the
    earliest ~85% is inner-train, the latest ~15% is inner-watch. The round's
    real val_df is never used for early stopping (and arrives target-blanked
    from the harness anyway).

    Side effect: appends this round's inner-train RMSE to the module-level
    _TRAIN_RMSES list so main() can print the train-vs-val overfitting gap.
    """
    import lightgbm as lgb

    train_feat = make_features(train_df)
    val_feat = make_features(val_df)

    # Chronological inner split (NOT random). train_df is already time-sorted by
    # the harness, but sort again defensively before splitting by row position.
    train_sorted = train_feat.sort_values(TIME_COL).reset_index(drop=True)
    cut = int(len(train_sorted) * 0.85)
    inner_train = train_sorted.iloc[:cut]
    inner_watch = train_sorted.iloc[cut:]

    X_tr, y_tr = inner_train[FEATURES], inner_train[TARGET]
    X_wt, y_wt = inner_watch[FEATURES], inner_watch[TARGET]

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_wt, y_wt)],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    best_it = model.best_iteration_
    # Record inner-train RMSE (overfitting check) using the best iteration.
    train_pred = model.predict(X_tr, num_iteration=best_it)
    _TRAIN_RMSES.append(rmse(y_tr.values, train_pred))

    return np.asarray(model.predict(val_feat[FEATURES], num_iteration=best_it), dtype=float)


# ---------------------------------------------------------------------------
# Shared LightGBM fit core (identical settings to lightgbm_v1; parameterised by
# the feature-engineering fn and feature list so v2 changes ONLY the features).
# ---------------------------------------------------------------------------
def _fit_lgbm_core(train_df, val_df, feature_fn, feats, record_train=True):
    """
    Fit a lightgbm_v1-style model and predict val_df. Returns (model, preds).

    Same params, same leak-safe chronological 85/15 inner split for early
    stopping as lightgbm_v1. If record_train, appends the inner-train RMSE to
    _TRAIN_RMSES (set False for one-off diagnostics so the CV table isn't
    polluted).
    """
    import lightgbm as lgb

    train_feat = feature_fn(train_df)
    val_feat = feature_fn(val_df)

    train_sorted = train_feat.sort_values(TIME_COL).reset_index(drop=True)
    cut = int(len(train_sorted) * 0.85)
    inner_train = train_sorted.iloc[:cut]
    inner_watch = train_sorted.iloc[cut:]

    X_tr, y_tr = inner_train[feats], inner_train[TARGET]
    X_wt, y_wt = inner_watch[feats], inner_watch[TARGET]

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_wt, y_wt)],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    best_it = model.best_iteration_
    if record_train:
        train_pred = model.predict(X_tr, num_iteration=best_it)
        _TRAIN_RMSES.append(rmse(y_tr.values, train_pred))

    preds = np.asarray(model.predict(val_feat[feats], num_iteration=best_it), dtype=float)
    return model, preds


# ---------------------------------------------------------------------------
# STEP 1 DIAGNOSTIC: feature importances (gain). One-off, NOT logged.
# ---------------------------------------------------------------------------
def print_feature_importances():
    """
    Train ONE lightgbm_v1-style model on all blocks EXCEPT the latest, then
    print every feature's GAIN importance (how much it improved the model),
    sorted descending, with each feature's share of total gain.

    Diagnostic only: not run through full CV, not logged to research_log.md.
    """
    df = load_data()
    df_blocked = assign_blocks(df, n_blocks=5)
    blocks = [int(b) for b in sorted(df_blocked["block"].unique())]
    latest = blocks[-1]

    train_df = df_blocked[df_blocked["block"] != latest].copy()
    val_df = df_blocked[df_blocked["block"] == latest].copy()  # only for API; preds unused

    # record_train=False so this diagnostic doesn't touch the CV _TRAIN_RMSES.
    model, _ = _fit_lgbm_core(train_df, val_df, make_features, FEATURES,
                              record_train=False)

    # Gain importances straight from the booster (robust regardless of the
    # sklearn wrapper's default importance_type).
    gains = model.booster_.feature_importance(importance_type="gain")
    names = model.booster_.feature_name()
    total = gains.sum()

    order = np.argsort(gains)[::-1]
    print("\n" + "=" * 64)
    print("STEP 1 DIAGNOSTIC: lightgbm_v1 feature importances (gain)")
    print(f"(trained on blocks {blocks[:-1]}; {len(train_df):,} rows; not logged)")
    print("=" * 64)
    print(f"  {'feature':<20} | {'gain':>16} | {'% of total':>10}")
    print("  " + "-" * 52)
    for idx in order:
        pct = (gains[idx] / total * 100) if total > 0 else 0.0
        print(f"  {names[idx]:<20} | {gains[idx]:>16,.1f} | {pct:>9.2f}%")


# ---------------------------------------------------------------------------
# SUBMISSION: retrain champion (lightgbm_v1 recipe) on ALL train data
# ---------------------------------------------------------------------------
def make_submission(out_path=SUBMISSION_PATH):
    """
    Retrain the lightgbm_v1 recipe on the FULL train.csv and write
    submission_v1.csv for the competition test set.

    Early stopping uses a chronological 85/15 split of train only. The test
    set is predicted, never used to fit or to pick the tree count.

    Test rows stay in file order so ids match sample_submission.csv.
    """
    import lightgbm as lgb

    print("\n" + "=" * 64)
    print("SUBMISSION: retrain lightgbm_v1 on ALL train.csv")
    print("=" * 64)

    train_df = load_data()
    train_feat = make_features(train_df)
    print(f"  Train rows: {len(train_feat):,}  "
          f"({train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()})")

    # Test is loaded in FILE ORDER (no sort) so ids stay aligned.
    test_raw = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])
    print(f"  Test rows:  {len(test_raw):,}  "
          f"({test_raw[TIME_COL].min()} -> {test_raw[TIME_COL].max()})")
    print(f"  Test has {TARGET}? {TARGET in test_raw.columns}  "
          "(expected False — that is what we predict)")

    missing_raw = [c for c in
                   ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
                    "RAIN", "WSPM", "hour", "month", "year", "wd", "station",
                    TIME_COL, "id"]
                   if c not in test_raw.columns]
    if missing_raw:
        raise ValueError(f"test is missing columns needed for features: {missing_raw}")
    print("  Sanity: test has every raw column make_features / FEATURES need.")

    test_feat = make_features(test_raw)

    missing_feat = [c for c in FEATURES if c not in test_feat.columns]
    if missing_feat:
        raise ValueError(f"make_features(test) is missing FEATURES: {missing_feat}")
    print(f"  Sanity: after make_features, all {len(FEATURES)} model features present.")

    # make_features subtracts each slice's own time_index min. On full train
    # that min is the dataset start (correct). On test it would reset to 0 and
    # look like 2013. Re-anchor test to the same origin the model was trained on.
    train_origin = ((train_df["year"] - 2013) * 12 + train_df["month"]).min()
    test_ti_raw = (test_feat["year"] - 2013) * 12 + test_feat["month"]
    test_feat["time_index"] = (test_ti_raw - train_origin).astype(int)
    print(f"  time_index train range: "
          f"{train_feat['time_index'].min()} -> {train_feat['time_index'].max()}")
    print(f"  time_index test range:  "
          f"{test_feat['time_index'].min()} -> {test_feat['time_index'].max()}  "
          f"(anchored to train origin {train_origin})")

    # Same categorical levels as train so LightGBM maps station consistently.
    test_feat["station"] = pd.Categorical(
        test_feat["station"], categories=train_feat["station"].cat.categories
    )

    # Leak-safe early stopping on FULL train, chronological 85/15.
    train_sorted = train_feat.sort_values(TIME_COL).reset_index(drop=True)
    cut = int(len(train_sorted) * 0.85)
    inner_train = train_sorted.iloc[:cut]
    inner_watch = train_sorted.iloc[cut:]
    print(f"  Inner-train: {len(inner_train):,} rows  |  "
          f"inner-watch: {len(inner_watch):,} rows  (chronological, not random)")

    X_tr, y_tr = inner_train[FEATURES], inner_train[TARGET]
    X_wt, y_wt = inner_watch[FEATURES], inner_watch[TARGET]

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_wt, y_wt)],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    best_it = model.best_iteration_
    print(f"  Best iteration (picked on inner-watch only): {best_it}")

    preds = np.asarray(model.predict(test_feat[FEATURES], num_iteration=best_it),
                       dtype=float)
    # LightGBM regression can dip below 0; PM2.5 cannot. Clip to the physical floor.
    n_neg = int((preds < 0).sum())
    if n_neg:
        print(f"  Clipped {n_neg} negative predictions to 0 "
              f"(raw min was {preds.min():.2f})")
        preds = np.clip(preds, 0, None)

    submission = pd.DataFrame({
        "id": test_raw["id"].values,
        TARGET: preds,
    })
    submission.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")

    _verify_submission(out_path)
    return out_path


def _verify_submission(out_path):
    """Print pass/fail for every required submission check."""
    sub = pd.read_csv(out_path)
    sample = pd.read_csv(SAMPLE_CSV)

    print("\n" + "=" * 64)
    print("SUBMISSION VERIFICATION")
    print("=" * 64)

    checks = []

    n_ok = len(sub) == len(sample)
    checks.append(n_ok)
    print(f"  [{'PASS' if n_ok else 'FAIL'}] row count: "
          f"submission={len(sub):,}  sample={len(sample):,}  "
          f"(expected 51,063)")

    cols = list(sub.columns)
    cols_ok = cols == ["id", TARGET]
    checks.append(cols_ok)
    print(f"  [{'PASS' if cols_ok else 'FAIL'}] columns: {cols}  "
          f"(expected ['id', '{TARGET}'])")

    sub_ids = set(sub["id"])
    sample_ids = set(sample["id"])
    ids_ok = sub_ids == sample_ids
    checks.append(ids_ok)
    extra = sub_ids - sample_ids
    missing = sample_ids - sub_ids
    print(f"  [{'PASS' if ids_ok else 'FAIL'}] id set matches sample_submission "
          f"(extra={len(extra)}, missing={len(missing)})")

    nan_ok = not sub[TARGET].isna().any()
    checks.append(nan_ok)
    print(f"  [{'PASS' if nan_ok else 'FAIL'}] no NaN in {TARGET}  "
          f"(n_missing={int(sub[TARGET].isna().sum())})")

    pmin = float(sub[TARGET].min())
    pmean = float(sub[TARGET].mean())
    pmax = float(sub[TARGET].max())
    range_ok = (pmin >= 0) and (pmax <= 999)
    checks.append(range_ok)
    print(f"  [{'PASS' if range_ok else 'FAIL'}] prediction range  "
          f"min={pmin:.2f}  mean={pmean:.2f}  max={pmax:.2f}")

    if all(checks):
        print("\n  All verification checks passed.")
    else:
        print("\n  SOME CHECKS FAILED — do not submit this file as-is.")


# ---------------------------------------------------------------------------
# RESEARCH LOG (markdown, append-only)
# ---------------------------------------------------------------------------
LOG_PATH = "research_log.md"


def _next_iteration_label(log_path, continuation):
    """
    Work out the next iteration heading from existing ones.

    Existing headings look like '## Iteration 2' or '## Iteration 2A'.
      - New model (continuation=False): bump the max main number, no suffix
        (e.g. last was 2 -> '3').
      - Continuation (continuation=True): keep the latest main number and add
        the next letter suffix (e.g. last was 2 or 2A -> '2A' or '2B').
    """
    import os
    import re

    mains = []          # all main integer numbers seen
    suffixes = {}       # main number -> list of letter suffixes seen
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                m = re.match(r"^##\s*Iteration\s+(\d+)([A-Z]*)\s*$", line.strip())
                if m:
                    num = int(m.group(1))
                    mains.append(num)
                    if m.group(2):
                        suffixes.setdefault(num, []).append(m.group(2))

    if not mains:
        return "1"  # empty log -> first entry is always a new model

    latest_main = max(mains)
    if not continuation:
        return str(latest_main + 1)

    # Continuation of the latest main number: next unused letter (A, B, C, ...).
    used = suffixes.get(latest_main, [])
    next_letter = chr(ord("A") + len(used))
    return f"{latest_main}{next_letter}"


def log_experiment(model, description, per_round_rmses, mean_rmse,
                   log_path=LOG_PATH, continuation=False):
    """
    Append ONE iteration block to the markdown research log (append-only;
    existing entries are never overwritten).

    Auto-fills the objective fields (Iteration label, Date/time, Model,
    CV avg RMSE, Per-block RMSEs). `description` is passed in. Keep? / Notes
    are left blank for manual editing.

    Set continuation=True when this is a tweak/continuation of the most recent
    model (labels it '<N>A', '<N>B', ...) rather than a brand-new model.
    """
    import os
    from datetime import datetime

    label = _next_iteration_label(log_path, continuation)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    per_block = " / ".join(f"{r:.1f}" for r in per_round_rmses)

    block = (
        f"## Iteration {label}\n"
        f"- Date/time: {now}\n"
        f"- Model: {model}\n"
        f"- Description: {description}\n"
        f"- CV avg RMSE: {mean_rmse:.4f}\n"
        f"- Per-block RMSEs: {per_block}\n"
        f"- Keep?: \n"
        f"- Notes: \n"
    )

    # Ensure a blank line before the new block if the file already has content.
    prefix = ""
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        prefix = "\n"

    with open(log_path, "a") as f:
        if not prefix and (not os.path.exists(log_path) or os.path.getsize(log_path) == 0):
            f.write("# Research Log\n\n")
        f.write(prefix + block)

    print(f"\nLogged Iteration {label} to {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    make_submission()


if __name__ == "__main__":
    main()
