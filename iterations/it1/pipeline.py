"""
MVP pipeline for the PM2.5-next-hour datathon.
Implements: cleaning, feature engineering, block assignment, and the
inner-loop / outer-loop CV harness described in datathon_plan.docx.
"""
import numpy as np
import pandas as pd

RAW_NUMERIC = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

# Time-block boundaries: 5 chronological blocks over the training period
# (2013-03-01 -> 2016-08-31), each end-exclusive.
BLOCK_BOUNDS = [
    ('2013-03-01', '2013-12-01'),
    ('2013-12-01', '2014-08-01'),
    ('2014-08-01', '2015-04-01'),
    ('2015-04-01', '2015-12-01'),
    ('2015-12-01', '2016-09-01'),
]

# wd ordered clean -> dirty based on the EDA (N/NW cleanest, E/SE dirtiest)
WD_CLEAN_TO_DIRTY = ['NW', 'NNW', 'N', 'WNW', 'NNE', 'W', 'NE', 'WSW',
                     'SW', 'SSW', 'S', 'SSE', 'ENE', 'SE', 'E', 'ESE']
WD_ORDER = {d: i for i, d in enumerate(WD_CLEAN_TO_DIRTY)}

# Known calendar events (see EXHAUSTIVE external factors doc). APEC/Parade/
# red-alert-2015 only ever occur in TRAIN; CNY occurs in both train and test.
CNY_DATES = ['2013-02-10', '2014-01-31', '2015-02-19', '2016-02-08', '2017-01-28']
APEC_BLUE = ('2014-11-01', '2014-11-13')
PARADE_BLUE = ('2015-08-20', '2015-09-04')
RED_ALERTS = [('2015-12-08', '2015-12-10'), ('2015-12-19', '2015-12-22'),
              ('2016-12-16', '2016-12-21'), ('2016-12-30', '2017-01-03')]


def assign_block(dt_series):
    """Map each timestamp to a block id 1..5, or 0 if outside range."""
    block = pd.Series(0, index=dt_series.index)
    for i, (s, e) in enumerate(BLOCK_BOUNDS, 1):
        mask = (dt_series >= s) & (dt_series < e)
        block[mask] = i
    return block


def _in_window(dt_series, start, end):
    return (dt_series >= start) & (dt_series < end)


def engineer_features(df, is_train=True):
    """
    Add engineered columns in place and return the dataframe.
    Safe to call on train and test identically (no leakage: every
    feature here is derivable from a single row's own timestamp/station,
    or from raw columns already present in that row).
    """
    df = df.copy()
    dt = pd.to_datetime(df['observation_timestamp'])
    df['dt'] = dt

    # --- Cyclical time encodings (hour, month both wrap around) ---
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # --- Heating season flag: Nov 15 - Mar 15 ---
    md = dt.dt.month * 100 + dt.dt.day
    df['heating_season'] = (((md >= 1115) | (md <= 315))).astype(int)

    # --- Secular trend: fractional year, lets a tree model represent the
    #     2013->2016 clean-air-campaign decline (see EXHAUSTIVE doc, section A1) ---
    df['year_frac'] = dt.dt.year + (dt.dt.dayofyear - 1) / 365.0

    # --- Wind direction: ordinal clean->dirty (from EDA), plus a raw
    #     categorical for the model's native categorical handling ---
    df['wd_clean_dirty_rank'] = df['wd'].map(WD_ORDER)
    df['wd_cat'] = df['wd'].astype('category')
    df['station_cat'] = df['station'].astype('category')

    # --- Weekday / weekend (traffic + ozone-weekend-effect proxy) ---
    df['dow'] = dt.dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # --- Dust vs haze signature: PM10 relative to PM10 baseline ---
    # (PM2.5 "next hour" can't be used as a same-row feature - that's the
    #  target. We use PM10 alone plus its interaction with month.)
    df['pm10_x_spring'] = df['PM10'] * df['month'].isin([3, 4, 5]).astype(int)

    # --- Sensor-cap flags (see EXHAUSTIVE doc, section E2) ---
    df['pm10_capped'] = (df['PM10'] >= 999).astype(int)
    df['co_capped'] = (df['CO'] >= 10000).astype(int)

    # --- Calendar event flags ---
    df['is_cny_window'] = 0
    for d in CNY_DATES:
        center = pd.Timestamp(d)
        df['is_cny_window'] |= _in_window(dt, center - pd.Timedelta(days=2),
                                           center + pd.Timedelta(days=3)).astype(int)
    df['is_apec_blue'] = _in_window(dt, *APEC_BLUE).astype(int)
    df['is_parade_blue'] = _in_window(dt, *PARADE_BLUE).astype(int)
    df['is_red_alert'] = 0
    for s, e in RED_ALERTS:
        df['is_red_alert'] |= _in_window(dt, s, e).astype(int)

    return df


def impute_missing(df, numeric_cols=RAW_NUMERIC, ffill_limit=3):
    """
    Hybrid imputation per the plan:
      1. Forward-fill within each station's time-ordered series, capped
         at `ffill_limit` consecutive hours (short sensor blips).
      2. Remaining gaps (long outages, e.g. CO in 2013) filled with the
         station+month+hour-of-day group mean (captures diurnal + seasonal
         pattern rather than a flat global average).
      3. Any still-missing (e.g. a station+month+hour combo never observed)
         filled with the station's overall mean, then the global mean.
    """
    df = df.sort_values(['station', 'dt']).copy()

    for col in numeric_cols:
        df[col] = df.groupby('station')[col].transform(
            lambda s: s.ffill(limit=ffill_limit)
        )

    for col in numeric_cols:
        still_missing = df[col].isna()
        if still_missing.any():
            grp_mean = df.groupby(['station', 'month', 'hour'])[col].transform('mean')
            df.loc[still_missing, col] = grp_mean[still_missing]

    for col in numeric_cols:
        still_missing = df[col].isna()
        if still_missing.any():
            station_mean = df.groupby('station')[col].transform('mean')
            df.loc[still_missing, col] = station_mean[still_missing]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    return df


FEATURE_COLS_NUMERIC = RAW_NUMERIC + [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'heating_season', 'year_frac', 'wd_clean_dirty_rank',
    'is_weekend', 'pm10_x_spring', 'pm10_capped', 'co_capped',
    'is_cny_window', 'is_apec_blue', 'is_parade_blue', 'is_red_alert',
]
FEATURE_COLS_CATEGORICAL = ['station_cat', 'wd_cat']
ALL_FEATURES = FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL
TARGET = 'PM2_5_next_hour'


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def baseline_predict(train_df, val_df):
    """Baseline model: per-station mean of the target, computed on train only."""
    station_means = train_df.groupby('station')[TARGET].mean()
    global_mean = train_df[TARGET].mean()
    return val_df['station'].map(station_means).fillna(global_mean).values


def inner_loop(df, model_fit_predict_fn, n_blocks=5, verbose=True):
    """
    Rolling-origin CV exactly as specified: for round r (2..n_blocks),
    train on blocks [1..r-1], validate on block [r]. Returns the list of
    per-block RMSEs and their average.

    `model_fit_predict_fn(train_df, val_df) -> np.array of predictions`
    lets any model (baseline function, or a real trained model) plug in.
    """
    block_rmses = []
    for r in range(2, n_blocks + 1):
        train_df = df[df['block'] < r]
        val_df = df[df['block'] == r]
        preds = model_fit_predict_fn(train_df, val_df)
        score = rmse(val_df[TARGET].values, preds)
        block_rmses.append(score)
        if verbose:
            print(f"  Round {r-1}: train blocks 1..{r-1} -> validate block {r} | RMSE = {score:.2f}")
    avg = float(np.mean(block_rmses))
    if verbose:
        print(f"  -> Inner-loop average RMSE = {avg:.2f}  (blocks: {[round(s,2) for s in block_rmses]})")
    return block_rmses, avg


def make_lgbm_fit_predict(params=None):
    """Returns a model_fit_predict_fn for inner_loop that trains a fresh
    LightGBM model each call (no leakage across rounds)."""
    import lightgbm as lgb
    default_params = dict(
        objective='regression', metric='rmse', n_estimators=500,
        learning_rate=0.05, num_leaves=63, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
    if params:
        default_params.update(params)

    def fit_predict(train_df, val_df):
        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            train_df[ALL_FEATURES], train_df[TARGET],
            categorical_feature=FEATURE_COLS_CATEGORICAL,
        )
        return model.predict(val_df[ALL_FEATURES])
    return fit_predict
