"""
Demo: train the it1 MVP pipeline on the full training set, then predict
PM2.5-next-hour for a single real test row and explain WHY via SHAP.

Run from repo root with the project venv:
    . .venv/bin/activate
    python Datathon/iterations/it1/demo_single_row.py
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import pipeline as P  # reuse the exact it1 feature/imputation/model code

REPO = "/Users/christianmartinez/Documents/grind/data_science/datathon"
DATA = os.path.join(REPO, "inter-uni-datathon-stream-2-beijing-multi-site-air-quality")
TRAIN = os.path.join(DATA, "train.csv")
TEST = os.path.join(DATA, "test(1).csv")

# Which test row to explain (0 = first row = the Aotizhongxin example).
ROW_IDX = 0

print("Loading data...")
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

print("Feature engineering + imputation (identical for train & test)...")
train = P.engineer_features(train, is_train=True)
train = P.impute_missing(train)
test = P.engineer_features(test, is_train=False)
test = P.impute_missing(test)

# Categorical dtypes must share categories between train & test.
for c in P.FEATURE_COLS_CATEGORICAL:
    cats = pd.api.types.union_categoricals(
        [train[c].astype("category"), test[c].astype("category")]
    ).categories
    train[c] = pd.Categorical(train[c], categories=cats)
    test[c] = pd.Categorical(test[c], categories=cats)

import lightgbm as lgb

params = dict(
    objective="regression", metric="rmse", n_estimators=500,
    learning_rate=0.05, num_leaves=63, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
)
print("Training LightGBM on 100% of train.csv (500 trees)...")
model = lgb.LGBMRegressor(**params)
model.fit(train[P.ALL_FEATURES], train[P.TARGET],
          categorical_feature=P.FEATURE_COLS_CATEGORICAL)

# ---- The single example row ----
row = test.iloc[[ROW_IDX]]
pred = float(model.predict(row[P.ALL_FEATURES])[0])

print("\n" + "=" * 64)
print("EXAMPLE ROW (raw):")
raw_show = ["id", "observation_timestamp", "station", "PM10", "SO2", "NO2",
            "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"]
for k in raw_show:
    print(f"  {k:22s}: {row.iloc[0][k]}")

print(f"\nMODEL PREDICTION  PM2_5_next_hour = {pred:.2f} ug/m3")
print(f"(training target mean = {train[P.TARGET].mean():.1f}, "
      f"per-station-mean baseline = "
      f"{train.groupby('station')[P.TARGET].mean().loc[row.iloc[0]['station']]:.1f})")

# ---- SHAP: which features pushed the prediction up / down ----
print("\nComputing SHAP contributions for this row...")
import shap
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(row[P.ALL_FEATURES])
base = float(explainer.expected_value)
contribs = pd.Series(sv[0], index=P.ALL_FEATURES).sort_values(key=np.abs,
                                                              ascending=False)

print(f"\nSHAP base value (model's average output) = {base:.2f}")
print("Top feature contributions (+ pushes prediction UP, - pushes DOWN):")
vals = {f: row.iloc[0][f] for f in P.ALL_FEATURES}
for feat, c in contribs.head(12).items():
    print(f"  {feat:22s} value={str(vals[feat])[:10]:>10s}  ->  {c:+7.2f}")
print(f"\n  base {base:.2f}  +  sum(contribs) {contribs.sum():+.2f}  =  {base + contribs.sum():.2f}")
print("=" * 64)
