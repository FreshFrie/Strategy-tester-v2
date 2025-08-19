#!/usr/bin/env python3
"""
NY→Asia Imbalance Tracker (tests the entity's law)
--------------------------------------------------
Goal
  Empirically test:
    • If **Asia contains** the prior New York (NY) session's imbalance (stays inside prior NY range),
      London is more likely to **expand / break out**.
    • If **Asia resolves** the imbalance (takes out prior NY high/low),
      London is more likely to **compress / revert** (fakeout).

Operational definitions (OHLCV-only, UTC-4):
  • Trading day boundary: 17:00 (NY close).
  • Asia Killzone (KZ): 19:00–21:00 (UTC-4) of the **current trading day**.
  • New York (NY) session used for range: 08:30–16:00 (UTC-4) of the **previous trading day**.
  • Prior NY range: [NY_high_prev, NY_low_prev].
  • Asia containment: Asia_KZ_high <= NY_high_prev AND Asia_KZ_low >= NY_low_prev.
  • Asia resolution: Asia_KZ_high > NY_high_prev OR Asia_KZ_low < NY_low_prev.
  • Streaks: consecutive day counts of containment or resolution.
  • London outcome: from harvest_events.csv → breakout = 1 - reverted_inside_by_end.

Inputs
  --ohlcv   : raw 1m OHLCV CSV (UTC-4) with columns (case-insensitive): date,time,open,high,low,close[,volume]
  --events  : harvest_events.csv produced earlier (has per-day London sweep/outcome)
  --out     : output directory
  --pip_size: pip size (default 0.0001 for EURUSD)
  --k       : streak length threshold to test (default 3)

Outputs (CSV in --out)
  ny_asia_daily.csv               : per trading day features (prior NY range, Asia KZ range, flags, streaks)
  ny_asia_conditional_stats.csv   : conditional breakout/fakeout rates by streak and filters

Example
  python ny_asia_imbalance_tracker.py \
    --ohlcv results/DAT_MT_EURUSD_M1_2024.csv \
    --events results/harvest_events.csv \
    --out results/imbalance_test \
    --pip_size 0.0001 --k 3
"""

import argparse
import os
from datetime import time, timedelta
import numpy as np
import pandas as pd

# Session anchors (UTC-4)
NY_CLOSE_ANCHOR = time(17, 0)    # trading day boundary
ASIA_KZ_START   = time(19, 0)
ASIA_KZ_END     = time(21, 0)
NY_START        = time(8, 30)
NY_END          = time(16, 0)

# ------------------------------- Helpers ------------------------------- #
def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cmap = {c.lower().strip(): c for c in df.columns}
    req = ["date","time","open","high","low","close"]
    if not all(r in cmap for r in req):
        # fallback: headerless common
        names = ["date","time","open","high","low","close","volume"]
        df = pd.read_csv(path, header=None, names=names)
        cmap = {c.lower().strip(): c for c in df.columns}
        missing = [r for r in req if r not in cmap]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
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

def in_window(t: time, start_t: time, end_t: time) -> bool:
    return (t >= start_t) and (t < end_t)

# ------------------------------ Core logic ------------------------------ #
def build_daily_features(px: pd.DataFrame, pip_size: float) -> pd.DataFrame:
    df = px.copy()
    df["trading_day_id"] = df["timestamp"].apply(trading_day_id)

    # Tag sessions
    df["is_asia_kz"] = df["timestamp"].dt.time.apply(lambda t: in_window(t, ASIA_KZ_START, ASIA_KZ_END))
    df["is_ny"]      = df["timestamp"].dt.time.apply(lambda t: in_window(t, NY_START, NY_END))

    rows = []
    # Iterate by trading day; need prior day's NY
    all_days = sorted(df["trading_day_id"].unique())
    prev_map = {d: (d - pd.Timedelta(days=1)) for d in all_days}

    day_groups = {d: g for d, g in df.groupby("trading_day_id")}

    for d in all_days:
        prev_d = prev_map[d]
        g_prev = day_groups.get(prev_d)
        g_curr = day_groups.get(d)
        if g_prev is None or g_curr is None:
            continue

        ny_prev = g_prev[g_prev["is_ny"]]
        asia_curr = g_curr[g_curr["is_asia_kz"]]
        if ny_prev.empty or asia_curr.empty:
            continue

        ny_hi = float(ny_prev["high"].max())
        ny_lo = float(ny_prev["low"].min())
        asia_hi = float(asia_curr["high"].max())
        asia_lo = float(asia_curr["low"].min())
        asia_rng = asia_hi - asia_lo

        contain = (asia_hi <= ny_hi) and (asia_lo >= ny_lo)
        resolve = (asia_hi > ny_hi) or (asia_lo < ny_lo)

        rows.append({
            "trading_day_id": d,
            "ny_prev_high": ny_hi,
            "ny_prev_low": ny_lo,
            "asia_high": asia_hi,
            "asia_low": asia_lo,
            "asia_range": asia_rng,
            "asia_range_pips": asia_rng / pip_size,
            "asia_contains_prev_ny": bool(contain),
            "asia_resolves_prev_ny": bool(resolve),
        })

    daily = pd.DataFrame(rows).sort_values("trading_day_id").reset_index(drop=True)

    # Streaks (containment / resolution)
    daily["contain_streak"] = 0
    daily["resolve_streak"] = 0
    cs = 0
    rs = 0
    prev_day = None
    for i, r in daily.iterrows():
        # Consecutive over trading days
        if prev_day is not None and (r["trading_day_id"] - prev_day).days != 1:
            cs = 0; rs = 0
        if r["asia_contains_prev_ny"]:
            cs += 1
        else:
            cs = 0
        if r["asia_resolves_prev_ny"]:
            rs += 1
        else:
            rs = 0
        daily.at[i, "contain_streak"] = cs
        daily.at[i, "resolve_streak"] = rs
        prev_day = r["trading_day_id"]

    return daily


def join_with_events(daily: pd.DataFrame, events_path: str) -> pd.DataFrame:
    ev = pd.read_csv(events_path)
    ev["trading_day_id"] = pd.to_datetime(ev["trading_day_id"]).dt.normalize()
    ev["breakout"] = (~ev["reverted_inside_by_end"]).astype(int)
    # Merge on trading_day_id (events are for current day London sweep)
    out = pd.merge(daily, ev[["trading_day_id","breakout","side","sweep_time"]], on="trading_day_id", how="left")
    # Basic sweep timing features
    out["sweep_hour"] = pd.to_datetime(out["sweep_time"]).dt.hour
    out["is_early_london"] = out["sweep_hour"].between(1, 1, inclusive="both") | out["sweep_hour"].between(1, 1)
    out["is_late_london"] = out["sweep_hour"].between(2, 4, inclusive="left")
    return out


def summarize_conditionals(df: pd.DataFrame, k: int) -> pd.DataFrame:
    base = df[~df["breakout"].isna()]
    if base.empty:
        return pd.DataFrame()
    rows = []

    def rate(mask, name):
        sub = base[mask & (~base["breakout"].isna())]
        if len(sub) == 0:
            return {"name": name, "n": 0, "breakout_rate": np.nan, "fakeout_rate": np.nan}
        br = sub["breakout"].mean()
        return {"name": name, "n": int(len(sub)), "breakout_rate": float(br), "fakeout_rate": float(1-br)}

    rows.append(rate(base["breakout"].notna(), "baseline"))
    rows.append(rate(base["contain_streak"] >= k, f"contain_streak>={k}"))
    rows.append(rate(base["resolve_streak"] >= k, f"resolve_streak>={k}"))
    rows.append(rate((base["contain_streak"] >= k) & (base["is_early_london"] == True), f"contain>={k} & early"))
    rows.append(rate((base["resolve_streak"] >= k) & (base["is_late_london"] == True), f"resolve>={k} & late"))
    rows.append(rate((base["contain_streak"] >= k) & (base["side"] == "up_sweep"), f"contain>={k} & up_sweep"))
    rows.append(rate((base["resolve_streak"] >= k) & (base["side"] == "down_sweep"), f"resolve>={k} & down_sweep"))

    return pd.DataFrame(rows)

# ----------------------------------- CLI ----------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlcv", required=True, help="Path to raw 1m OHLCV CSV (UTC-4)")
    ap.add_argument("--events", required=True, help="Path to harvest_events.csv")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--pip_size", type=float, default=0.0001)
    ap.add_argument("--k", type=int, default=3, help="Streak length threshold")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    px = load_ohlcv(args.ohlcv)
    daily = build_daily_features(px, pip_size=args.pip_size)
    daily_joined = join_with_events(daily, args.events)
    daily_joined.to_csv(os.path.join(args.out, "ny_asia_daily.csv"), index=False)

    stats = summarize_conditionals(daily_joined, k=args.k)
    stats.to_csv(os.path.join(args.out, "ny_asia_conditional_stats.csv"), index=False)

    print("Wrote:")
    print(" -", os.path.join(args.out, "ny_asia_daily.csv"))
    print(" -", os.path.join(args.out, "ny_asia_conditional_stats.csv"))
    print("\nPreview:")
    print(stats.to_string(index=False))
