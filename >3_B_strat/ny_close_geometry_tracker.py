#!/usr/bin/env python3
"""
NY Close Geometry Tracker — Testing the entity’s words
------------------------------------------------------

Goal:
  Measure New York (NY) session close position relative to its daily session range,
  to predict whether the coming Asia session will contain or resolve the prior NY imbalance.

Logic:
  - NY session window: 08:30–16:00 (UTC-4)
  - Compute NY_high, NY_low, NY_close, NY_midpoint
  - Feature: close_position = (NY_close - NY_low) / (NY_high - NY_low)
      • ~0.5 → close near middle (predict containment)
      • ~0 or ~1 → close near extremes (predict resolution)
  - Define thresholds:
      • If 0.33 <= close_position <= 0.67 → tag as “center_close” (containment bias)
      • Else → tag as “edge_close” (resolution bias)
  - Build streak counts of center/edge closes
  - Join with Asia KZ outcomes from ny_asia_daily.csv
  - Output conditional stats: does NY close geometry predict Asia containment/resolution?

Inputs:
  --ohlcv   : raw 1m OHLCV CSV (UTC-4)
  --asia    : ny_asia_daily.csv (from imbalance tracker)
  --out     : output directory
  --pip_size: pip size (default 0.0001)
  --k       : streak threshold (default 3)

Outputs:
  ny_close_geometry_daily.csv        : per-day NY close geometry metrics & tags
  ny_close_geometry_conditional.csv  : conditional containment/resolution rates by streaks

Usage:
  python ny_close_geometry_tracker.py \
    --ohlcv results/DAT_MT_EURUSD_M1_2024.csv \
    --asia results/imbalance_test/ny_asia_daily.csv \
    --out results/nyclose_test \
    --pip_size 0.0001 --k 3
"""

import argparse, os
import pandas as pd
import numpy as np
from datetime import time, timedelta

NY_START = time(8,30)
NY_END   = time(16,0)
NY_CLOSE_ANCHOR = time(17,0)

# ---------------------------- Helpers ---------------------------- #
def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cmap = {c.lower().strip(): c for c in df.columns}
    req = ["date","time","open","high","low","close"]
    if not all(r in cmap for r in req):
        names = ["date","time","open","high","low","close","volume"]
        df = pd.read_csv(path, header=None, names=names)
        cmap = {c.lower().strip(): c for c in df.columns}
        if not all(r in cmap for r in req):
            missing = [r for r in req if r not in cmap]
            raise ValueError(f"Missing required col(s): {missing}")
    df["timestamp"] = pd.to_datetime(df[cmap["date"]].astype(str) + " " + df[cmap["time"]].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)
    for k in ["open","high","low","close","volume"]:
        if k in cmap:
            df[k] = pd.to_numeric(df[cmap[k]], errors="coerce")
    return df

def trading_day_id(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.date()
    if ts.time() < NY_CLOSE_ANCHOR:
        d = (ts - timedelta(days=1)).date()
    return pd.Timestamp(d)

def in_window(t: time, start_t: time, end_t: time):
    return (t >= start_t) and (t < end_t)

# -------------------------- Core Logic -------------------------- #
def compute_ny_close_geometry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trading_day_id"] = df["timestamp"].apply(trading_day_id)
    df["is_ny"] = df["timestamp"].dt.time.apply(lambda t: in_window(t, NY_START, NY_END))

    rows = []
    for d, g in df.groupby("trading_day_id"):
        ny = g[g["is_ny"]]
        if ny.empty: continue
        ny_hi = float(ny["high"].max())
        ny_lo = float(ny["low"].min())
        ny_close = float(ny["close"].iloc[-1])
        ny_mid = (ny_hi + ny_lo)/2
        close_pos = (ny_close - ny_lo) / (ny_hi - ny_lo) if ny_hi > ny_lo else np.nan
        center_close = 0.33 <= close_pos <= 0.67
        edge_close   = not center_close
        rows.append({
            "trading_day_id": d,
            "ny_high": ny_hi,
            "ny_low": ny_lo,
            "ny_close": ny_close,
            "ny_mid": ny_mid,
            "close_pos": close_pos,
            "center_close": center_close,
            "edge_close": edge_close,
        })
    daily = pd.DataFrame(rows).sort_values("trading_day_id").reset_index(drop=True)

    # Streaks
    daily["center_streak"] = 0
    daily["edge_streak"] = 0
    cs = rs = 0
    prev_day = None
    for i, r in daily.iterrows():
        if prev_day is not None and (r["trading_day_id"] - prev_day).days != 1:
            cs = rs = 0
        if r["center_close"]: cs += 1
        else: cs = 0
        if r["edge_close"]: rs += 1
        else: rs = 0
        daily.at[i,"center_streak"] = cs
        daily.at[i,"edge_streak"] = rs
        prev_day = r["trading_day_id"]

    return daily

# ------------------------- Conditional Stats ------------------------ #
def summarize_conditionals(daily: pd.DataFrame, asia_df: pd.DataFrame, k: int) -> pd.DataFrame:
    merged = pd.merge(daily, asia_df, on="trading_day_id", how="inner")
    rows = []

    def rate(mask, name, col):
        sub = merged[mask]
        if len(sub)==0:
            return {"name": name, "n": 0, "contain_rate": np.nan, "resolve_rate": np.nan}
        contain = sub[col].mean()
        return {"name": name, "n": int(len(sub)), "contain_rate": contain, "resolve_rate": 1-contain}

    # baseline
    rows.append(rate(merged["asia_contains_prev_ny"].notna(), "baseline", "asia_contains_prev_ny"))
    rows.append(rate(merged["center_streak"]>=k, f"center_streak>={k}", "asia_contains_prev_ny"))
    rows.append(rate(merged["edge_streak"]>=k, f"edge_streak>={k}", "asia_contains_prev_ny"))
    return pd.DataFrame(rows)

# ------------------------------ CLI ------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlcv", required=True)
    ap.add_argument("--asia", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pip_size", type=float, default=0.0001)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_ohlcv(args.ohlcv)
    daily = compute_ny_close_geometry(df)
    asia = pd.read_csv(args.asia)
    asia["trading_day_id"] = pd.to_datetime(asia["trading_day_id"]).dt.normalize()

    daily.to_csv(os.path.join(args.out, "ny_close_geometry_daily.csv"), index=False)

    stats = summarize_conditionals(daily, asia, k=args.k)
    stats.to_csv(os.path.join(args.out, "ny_close_geometry_conditional.csv"), index=False)

    print("Preview:\n", stats.to_string(index=False))
