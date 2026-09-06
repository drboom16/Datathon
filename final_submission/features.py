"""Load the official CSVs and build the feature table used by the final model."""
from pathlib import Path

import numpy as np
import pandas as pd

# Other pollutants released at the observation hour. Current PM2.5 is not released.
POLL = ["PM10", "SO2", "NO2", "CO", "O3"]
Y = "PM2_5_next_hour"


def find_data_dir(start):
    """Official CSVs go in ./data next to this file, or pass --data-dir."""
    return Path(start) / "data"


def test_csv_path(data_dir):
    data_dir = Path(data_dir)
    for name in ("test(1).csv", "test.csv"):
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No test CSV found in {data_dir}. Put train.csv, test(1).csv, "
        "and sample_submission.csv there, or pass --data-dir."
    )


def load_competition_frames(data_dir):
    """Stack train and test so lags can be computed along each station's timeline."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Put the official CSVs in {data_dir} or pass --data-dir."
        )
    train = pd.read_csv(train_path, parse_dates=["observation_timestamp"])
    test = pd.read_csv(test_csv_path(data_dir), parse_dates=["observation_timestamp"])
    train = train.rename(columns={"observation_timestamp": "ts"})
    test = test.rename(columns={"observation_timestamp": "ts"})
    train["is_test"] = 0
    test["is_test"] = 1
    test[Y] = np.nan
    return pd.concat([train, test], ignore_index=True).sort_values(["station", "ts"]).reset_index(drop=True)


def add_features(df):
    """
    Build the released-column feature set.

    Everything here uses the observation hour or earlier hours at the same
    station (or the same hour at the other stations). Nothing looks ahead
    at the target.
    """
    g = df.groupby("station", group_keys=False)

    # PM10 and CO get a longer memory — they move with the target the most.
    for col in POLL:
        lags = [1, 2, 3, 6, 12, 24] if col in ("PM10", "CO") else [1, 2, 3]
        for lag in lags:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        df[f"{col}_tend3"] = df[col] - g[col].shift(3)

    for col in ["PM10", "CO"]:
        df[f"{col}_rmean6"] = g[col].transform(lambda s: s.rolling(6, min_periods=1).mean())
        df[f"{col}_rmean24"] = g[col].transform(lambda s: s.rolling(24, min_periods=1).mean())
        df[f"{col}_rstd6"] = g[col].transform(lambda s: s.rolling(6, min_periods=1).std())
        df[f"{col}_rmax6"] = g[col].transform(lambda s: s.rolling(6, min_periods=1).max())
        df[f"{col}_ewma6"] = g[col].transform(lambda s: s.ewm(span=6, min_periods=1).mean())

    # Short-term movement and a couple of simple ratios.
    df["PM10_d1"] = df["PM10"] - df["PM10_lag1"]
    df["CO_d1"] = df["CO"] - df["CO_lag1"]
    df["PM10_d3"] = df["PM10"] - df["PM10_lag3"]
    df["PM10_accel"] = df["PM10_d1"] - (df["PM10_lag1"] - df["PM10_lag2"])
    df["co_over_pm10"] = df["CO"] / (df["PM10"] + 1)
    df["PM10_pw"] = df["PM10"] / (df["WSPM"].fillna(0) + 0.5)
    df["WSPM_rmean6"] = g["WSPM"].transform(lambda s: s.rolling(6, min_periods=1).mean())

    # Weather context: how dry the air is, plus recent pressure / dew-point drift.
    df["tempdew_depr"] = df["TEMP"] - df["DEWP"]
    df["DEWP_d3"] = df["DEWP"] - g["DEWP"].shift(3)
    df["PRES_d3"] = df["PRES"] - g["PRES"].shift(3)

    # Clock features. The last one flags the colder months, when levels run higher.
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df["dow"] = df.ts.dt.dayofweek
    df["doy"] = df.ts.dt.dayofyear
    df["heating"] = df.month.isin([11, 12, 1, 2, 3]).astype(int)

    # Same-hour picture across the 12-station network.
    spatial = []
    for col in POLL:
        wide = df.pivot_table(index="ts", columns="station", values=col)
        df = df.merge(wide.mean(axis=1).rename(f"city_{col}_mean"), on="ts")
        df = df.merge(wide.max(axis=1).rename(f"city_{col}_max"), on="ts")
        df = df.merge(wide.std(axis=1).rename(f"city_{col}_std"), on="ts")
        spatial += [f"city_{col}_mean", f"city_{col}_max", f"city_{col}_std"]

    df = df.sort_values(["station", "ts"]).reset_index(drop=True)
    g = df.groupby("station", group_keys=False)
    n_stations = df.station.nunique()
    df["city_PM10_lo"] = (df.city_PM10_mean * n_stations - df.PM10) / (n_stations - 1)
    df["gap_PM10_city"] = df.PM10 - df.city_PM10_mean
    df["gap_CO_city"] = df.CO - df.city_CO_mean
    df["city_PM10_d1"] = df.city_PM10_mean - g["city_PM10_mean"].shift(1)
    df["city_CO_d1"] = df.city_CO_mean - g["city_CO_mean"].shift(1)
    spatial += ["city_PM10_lo", "gap_PM10_city", "gap_CO_city", "city_PM10_d1", "city_CO_d1"]

    df["station"] = df["station"].astype("category")
    df["wd"] = df["wd"].astype("category")

    # Keep this list in a fixed order so retraining stays stable.
    feat = (
        POLL
        + ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "hour", "month", "dow", "doy",
           "heating", "hour_sin", "hour_cos", "month_sin", "month_cos", "tempdew_depr",
           "PM10_d1", "CO_d1", "WSPM_rmean6"]
        + [f"{c}_lag{L}" for c in POLL for L in ([1, 2, 3, 6, 12, 24] if c in ("PM10", "CO") else [1, 2, 3])]
        + [f"{c}_tend3" for c in POLL]
        + ["PM10_rmean6", "PM10_rmean24", "PM10_rstd6", "CO_rmean6", "CO_rmean24", "CO_rstd6"]
        + ["PM10_rmax6", "PM10_ewma6", "PM10_d3", "PM10_accel", "co_over_pm10", "PM10_pw",
           "CO_rmax6", "CO_ewma6", "DEWP_d3", "PRES_d3"]
        + spatial
        + ["station", "wd"]
    )
    return df, feat
