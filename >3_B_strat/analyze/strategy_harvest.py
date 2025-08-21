#!/usr/bin/env python3
"""
London Sweep Pattern Harvester
------------------------------
Generates harvest_events.csv from 1-minute OHLCV (UTC-4).

Steps:
- Anchor trading_day_id at 17:00 NY close.
- Identify Asia KZ (19:00–21:00) high/low.
- Detect first London sweep (01:00–04:00).
- Record sweep details, extensions, revert status.

Outputs:
  harvest_events.csv — one row per trading day with sweep
  harvest_summary.csv — overall stats
  harvest_reports.xlsx — all sheets together

Usage:
  python strategy_harvest.py --csv results/DAT_MT_EURUSD_M1_2024.csv --out results/
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import timedelta, time

# ------------------------- CONFIG -------------------------
NY_CLOSE_ANCHOR = time(17,0)   # trading day boundary
ASIA_KZ_START   = time(19,0)
ASIA_KZ_END     = time(21,0)
LONDON_KZ_START = time(1,0)
LONDON_KZ_END   = time(4,0)

# --------------------- HELPERS ----------------------------
def trading_day_id_from_timestamp(ts):
    d = ts.date()
    if ts.time() < NY_CLOSE_ANCHOR:
        d = (ts - timedelta(days=1)).date()
    return pd.Timestamp(d)

def in_window(ts_time, start_t, end_t):
    return (ts_time >= start_t) and (ts_time < end_t)

# ---------------------- CORE ------------------------------
def load_ohlcv(path):
    # Try reading with header. If the expected columns aren't present, fall back
    # to a common headerless layout: date,time,open,high,low,close[,volume].
    df = pd.read_csv(path)
    cmap = {c.lower().strip(): c for c in df.columns}
    required = ["date","time","open","high","low","close"]
    if not all(r in cmap for r in required):
        # fallback: headerless common names
        names = ["date","time","open","high","low","close","volume"]
        df = pd.read_csv(path, header=None, names=names)
        cmap = {c.lower().strip(): c for c in df.columns}
        if not all(r in cmap for r in required):
            missing = [r for r in required if r not in cmap]
            raise ValueError(f"Missing required column {missing} in CSV")
    df["timestamp"] = pd.to_datetime(df[cmap["date"]].astype(str) + " " + df[cmap["time"]].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)
    for k in ["open","high","low","close","volume"]:
        if k in cmap:
            df[k] = pd.to_numeric(df[cmap[k]], errors="coerce")
    return df

def tag_sessions(df):
    df["trading_day_id"] = df["timestamp"].apply(trading_day_id_from_timestamp)
    df["is_asia_kz"]   = df["timestamp"].dt.time.apply(lambda t: in_window(t, ASIA_KZ_START, ASIA_KZ_END))
    df["is_london_kz"] = df["timestamp"].dt.time.apply(lambda t: in_window(t, LONDON_KZ_START, LONDON_KZ_END))
    return df

def compute_asia_range(df_day):
    asia = df_day[df_day["is_asia_kz"]]
    if asia.empty:
        return None, None
    return asia["high"].max(), asia["low"].min()

def first_london_sweep(df_day, asia_high, asia_low):
    london = df_day[df_day["is_london_kz"]]
    if london.empty or pd.isna(asia_high) or pd.isna(asia_low):
        return None
    breach_high = london[london["high"] > asia_high]
    breach_low  = london[london["low"]  < asia_low]
    first_high_time = breach_high["timestamp"].iloc[0] if not breach_high.empty else None
    first_low_time  = breach_low["timestamp"].iloc[0] if not breach_low.empty else None
    if first_high_time is None and first_low_time is None:
        return None
    if first_high_time is not None and (first_low_time is None or first_high_time <= first_low_time):
        side, sweep_time, level = "up_sweep", first_high_time, asia_high
    else:
        side, sweep_time, level = "down_sweep", first_low_time, asia_low
    london_end = london["timestamp"].max()
    post = london[(london["timestamp"] >= sweep_time) & (london["timestamp"] <= london_end)]
    if post.empty:
        return None
    if side == "up_sweep":
        max_extension = post["high"].max() - level
        last_close = post["close"].iloc[-1]
        reverted_inside = (last_close <= level and last_close >= asia_low)
        max_adverse = post["high"].max() - level
        max_favorable = level - post["low"].min()
    else:
        max_extension = level - post["low"].min()
        last_close = post["close"].iloc[-1]
        reverted_inside = (last_close >= level and last_close <= asia_high)
        max_adverse = level - post["low"].min()
        max_favorable = post["high"].max() - level
    # Time-to-revert
    revert_time = None
    for _, row in post.iterrows():
        if (row["close"] <= asia_high) and (row["close"] >= asia_low):
            revert_time = row["timestamp"]
            break
    time_to_revert_min = (revert_time - sweep_time).total_seconds()/60.0 if revert_time is not None else np.nan
    return {
        "side": side,
        "sweep_time": sweep_time,
        "sweep_level": level,
        "london_end": london_end,
        "max_extension": float(max_extension),
        "reverted_inside_by_end": bool(reverted_inside),
        "time_to_revert_min": time_to_revert_min,
        "max_adverse": float(max_adverse),
        "max_favorable": float(max_favorable),
        "asia_high": float(asia_high),
        "asia_low": float(asia_low)
    }

def harvest_events(df):
    rows = []
    for day, df_day in df.groupby("trading_day_id"):
        asia_high, asia_low = compute_asia_range(df_day)
        if asia_high is None or asia_low is None:
            continue
        sweep = first_london_sweep(df_day, asia_high, asia_low)
        if sweep:
            rows.append({"trading_day_id": day, **sweep})
    return pd.DataFrame(rows)

# ------------------------- MAIN ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to raw 1m OHLCV CSV (UTC-4)")
    ap.add_argument("--out", required=True, help="Output dir")
    args = ap.parse_args()

    df = load_ohlcv(args.csv)
    df = tag_sessions(df)
    events = harvest_events(df)
    os.makedirs(args.out, exist_ok=True)
    events_path = os.path.join(args.out, "harvest_events.csv")
    summary_path = os.path.join(args.out, "harvest_summary.csv")
    xlsx_path = os.path.join(args.out, "harvest_reports.xlsx")

    events.to_csv(events_path, index=False)

    if not events.empty:
        overall_revert = events["reverted_inside_by_end"].mean()
        total = len(events)
        summary = pd.DataFrame([{
            "total_days_with_sweep": total,
            "overall_revert_rate_by_london_end": overall_revert
        }])
        summary.to_csv(summary_path, index=False)
        with pd.ExcelWriter(xlsx_path) as xw:
            events.to_excel(xw, sheet_name="events", index=False)
            summary.to_excel(xw, sheet_name="summary", index=False)
    print(f"Wrote: {events_path}, {summary_path}, {xlsx_path}")

if __name__ == "__main__":
    main()