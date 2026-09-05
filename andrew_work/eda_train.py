import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

CSV = "inter-uni-datathon-stream-2-beijing-multi-site-air-quality/train.csv"
df = pd.read_csv(CSV)

TARGET = "PM2_5_next_hour"


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# 1. SHAPE & TYPES
# ---------------------------------------------------------------------------
header("1. SHAPE & TYPES")
print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")
print("\nDtype of every column:")
print(df.dtypes.to_string())


# ---------------------------------------------------------------------------
# 2. MISSING VALUES
# ---------------------------------------------------------------------------
header("2. MISSING VALUES")
miss_count = df.isna().sum()
miss_pct = df.isna().mean() * 100
miss = pd.DataFrame({"missing_count": miss_count, "missing_pct": miss_pct})
miss = miss[miss["missing_count"] > 0].sort_values("missing_count", ascending=False)

if miss.empty:
    print("No missing values in any column.")
else:
    print("Columns with any missing values (sorted, most first):")
    print(miss.to_string())

    worst_col = miss.index[0]
    print(f"\n--- Breakdown of missing rows for column with MOST missing: '{worst_col}' ---")
    missing_rows = df[df[worst_col].isna()]

    print("\n(a) Missing by station:")
    by_station = missing_rows["station"].value_counts()
    print(by_station.to_string())

    print("\n(b) Missing by year-month (top 5):")
    ym = missing_rows["year"].astype(str) + "-" + missing_rows["month"].astype(str).str.zfill(2)
    by_ym = ym.value_counts().sort_values(ascending=False).head(5)
    print(by_ym.to_string())

    # Simple heuristic on random vs clumped
    n_stations = df["station"].nunique()
    station_spread = by_station / by_station.sum() * 100
    ym_full = ym.value_counts()
    total_ym_periods = (df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)).nunique()
    top5_share = by_ym.sum() / miss.loc[worst_col, "missing_count"] * 100
    print(
        f"\nSpread: missingness touches {by_station.shape[0]}/{n_stations} stations; "
        f"station share range {station_spread.min():.1f}%-{station_spread.max():.1f}%."
    )
    print(
        f"Top-5 year-months account for {top5_share:.1f}% of all missing in this column "
        f"(out of {ym_full.shape[0]}/{total_ym_periods} year-months containing any missing)."
    )
    print(
        "Interpretation: if station shares are roughly even and the top-5 year-months hold "
        "only a small % -> looks random/spread out; if concentrated in few stations or "
        "year-months -> looks clumped."
    )


# ---------------------------------------------------------------------------
# 3. UNUSUAL VALUES
# ---------------------------------------------------------------------------
header("3. UNUSUAL VALUES")
t = df[TARGET]
print(f"Target '{TARGET}':")
print(f"  count == 999 : {(t == 999).sum():,}")
print(f"  count >= 900 : {(t >= 900).sum():,}")
print(f"  count >= 500 : {(t >= 500).sum():,}")

nines = df[df[TARGET] == 999]
if len(nines):
    ym_nines = (nines["year"].astype(str) + "-" + nines["month"].astype(str).str.zfill(2)).value_counts()
    print("\n  999s by year-month (top 5):")
    print("  " + ym_nines.head(5).to_string().replace("\n", "\n  "))
    top_share = ym_nines.iloc[0] / ym_nines.sum() * 100
    print(f"  -> Top year-month holds {top_share:.1f}% of all 999s "
          f"({'clustered' if top_share > 50 else 'not strongly clustered'}).")
else:
    print("  No 999 values in target.")

print("\nMaxima of pollutant columns (flagging round sensor caps like 999 / 10000):")
for col in ["PM10", "SO2", "NO2", "CO", "O3"]:
    mx = df[col].max()
    flag = ""
    if mx in (999, 9999, 10000, 99999):
        flag = "  <-- looks like a round sensor cap"
    print(f"  {col:5s} max = {mx:,.2f}{flag}")


# ---------------------------------------------------------------------------
# 4. TARGET DISTRIBUTION
# ---------------------------------------------------------------------------
header("4. TARGET DISTRIBUTION (PM2_5_next_hour)")
print(f"count  : {t.count():,}")
print(f"mean   : {t.mean():.4f}")
print(f"median : {t.median():.4f}")
print(f"std    : {t.std():.4f}")
print(f"min    : {t.min():.4f}")
print(f"max    : {t.max():.4f}")
for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"{int(q*100):3d}th pct: {t.quantile(q):.4f}")
print(f"skewness: {t.skew():.4f}")


# ---------------------------------------------------------------------------
# 5. CORRELATION WITH TARGET
# ---------------------------------------------------------------------------
header("5. CORRELATION WITH TARGET (Pearson, sorted by |r|)")
feats = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "hour"]
corr = df[feats + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
corr_tbl = pd.DataFrame({"pearson_r": corr, "abs_r": corr.abs()}).sort_values("abs_r", ascending=False)
print(corr_tbl.to_string())
print("\nNote: hour (and month) are cyclical - low linear r does NOT mean no signal.")


# ---------------------------------------------------------------------------
# 6. PATTERNS
# ---------------------------------------------------------------------------
header("6. PATTERNS: mean PM2_5_next_hour")
print("\n(a) By hour of day:")
print(df.groupby("hour")[TARGET].mean().to_string())

print("\n(b) By month:")
print(df.groupby("month")[TARGET].mean().to_string())

print("\n(c) By station:")
print(df.groupby("station")[TARGET].mean().sort_values(ascending=False).to_string())

print("\nDone.")
