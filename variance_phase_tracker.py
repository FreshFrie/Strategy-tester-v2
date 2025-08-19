#!/usr/bin/env python3
"""
Variance Phase Tracker - Asia Range Regime Detection (UTC-4, 1m OHLCV)
------------------------------------------------------------------------
Purpose
    Detect volatility regime shifts that rotate the edge between quartiles (Q1..Q4)
    by monitoring the weekly variance of Asia Killzone ranges (19:00-21:00 UTC-4).

This script computes daily Asia ranges, aggregates weekly statistics, computes
a rolling z-score on weekly variance and emits CSVs with regime labels and
transitions.
"""

import argparse
import os
from dataclasses import dataclass
from datetime import time, timedelta
import numpy as np
import pandas as pd

# ---------------------------- Config Constants ---------------------------- #
NY_CLOSE_ANCHOR = time(17, 0)    # Trading day boundary at 17:00 (NY close)
ASIA_KZ_START   = time(19, 0)    # 19:00-21:00 UTC-4
ASIA_KZ_END     = time(21, 0)

# ------------------------------ IO / Loading ------------------------------ #
def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cmap = {c.lower().strip(): c for c in df.columns}
    required = ["date", "time", "open", "high", "low", "close"]
    # Fallback to headerless layout if needed
    if not all(r in cmap for r in required):
        names = ["date","time","open","high","low","close","volume"]
        df = pd.read_csv(path, header=None, names=names)
        cmap = {c.lower().strip(): c for c in df.columns}
        if not all(r in cmap for r in required):
            missing = [r for r in required if r not in cmap]
            raise ValueError(f"Missing required column(s): {missing}")
    df["timestamp"] = pd.to_datetime(df[cmap["date"]].astype(str) + " " + df[cmap["time"]].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Coerce numerics
    for k in ["open","high","low","close","volume"]:
        if k in cmap:
            df[k] = pd.to_numeric(df[cmap[k]], errors="coerce")
    return df

# --------------------------- Time/Session Helpers ------------------------- #
def trading_day_id_from_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.date()
    if ts.time() < NY_CLOSE_ANCHOR:
        d = (ts - timedelta(days=1)).date()
    return pd.Timestamp(d)


def in_window(ts_time: time, start_t: time, end_t: time) -> bool:
    return (ts_time >= start_t) and (ts_time < end_t)

# --------------------------------- Core ----------------------------------- #

def compute_asia_daily(df: pd.DataFrame, pip_size: float) -> pd.DataFrame:
    """Return per-day Asia KZ high/low/range (in price & pips)."""
    out_rows = []
    df = df.copy()
    df["trading_day_id"] = df["timestamp"].apply(trading_day_id_from_timestamp)
    df["is_asia_kz"] = df["timestamp"].dt.time.apply(lambda t: in_window(t, ASIA_KZ_START, ASIA_KZ_END))

    for day, g in df.groupby("trading_day_id"):
        asia = g[g["is_asia_kz"]]
        if asia.empty:
            continue
        a_high = float(asia["high"].max())
        a_low  = float(asia["low"].min())
        a_rng  = a_high - a_low
        rng_pips = a_rng / pip_size
        out_rows.append({
            "trading_day_id": day,
            "asia_high": a_high,
            "asia_low": a_low,
            "asia_range": a_rng,
            "asia_range_pips": rng_pips,
        })
    daily = pd.DataFrame(out_rows).sort_values("trading_day_id").reset_index(drop=True)
    return daily


@dataclass
class RegimeThresholds:
    z_up: float = 0.75
    z_down: float = -0.75


def classify_regime(z: float, thr: RegimeThresholds) -> str:
    if np.isnan(z):
        return "Neutral"
    if z >= thr.z_up:
        return "Expanding"
    if z <= thr.z_down:
        return "Compressing"
    return "Neutral"


def compute_weekly_metrics(daily: pd.DataFrame, week_end: str, roll: int,
                            thr: RegimeThresholds) -> pd.DataFrame:
    # Build week periods (e.g., W-SUN). Pandas expects W-<DAY>
    week_rule = f"W-{week_end.upper()}"
    daily = daily.copy()
    daily["week"] = daily["trading_day_id"].dt.to_period(week_rule).dt.to_timestamp()

    grp = daily.groupby("week")
    weekly = grp["asia_range_pips"].agg([
        ("days", "count"),
        ("mean_pips", "mean"),
        ("std_pips", "std"),
        ("var_pips", "var"),
        ("p25_pips", lambda s: np.nanpercentile(s, 25)),
        ("median_pips", "median"),
        ("p75_pips", lambda s: np.nanpercentile(s, 75)),
    ]).reset_index()

    # Handle small-sample weeks (std/var may be NaN)
    weekly["std_pips"].fillna(0.0, inplace=True)
    weekly["var_pips"].fillna(0.0, inplace=True)
    weekly["cv"] = np.where(weekly["mean_pips"].abs() > 1e-9, weekly["std_pips"] / weekly["mean_pips"], np.nan)

    # Rolling z-score on weekly variance
    weekly["roll_var_mean"] = weekly["var_pips"].rolling(roll, min_periods=max(2, roll//3)).mean()
    weekly["roll_var_std"]  = weekly["var_pips"].rolling(roll, min_periods=max(2, roll//3)).std(ddof=0)
    weekly["z_var"] = (weekly["var_pips"] - weekly["roll_var_mean"]) / weekly["roll_var_std"].replace(0, np.nan)

    # Regime classification
    weekly["regime"] = weekly["z_var"].apply(lambda z: classify_regime(z, thr))

    # Slope / direction (simple diff)
    weekly["var_diff"] = weekly["var_pips"].diff()
    weekly["z_diff"] = weekly["z_var"].diff()
    weekly["trend"] = np.select([
        weekly["z_diff"] > 0.0,
        weekly["z_diff"] < 0.0
    ], ["rising", "falling"], default="flat")

    # Regime transitions
    weekly["regime_prev"] = weekly["regime"].shift(1)
    weekly["transition"] = np.where(weekly["regime"] != weekly["regime_prev"], True, False)

    # Heuristic module weights (0..1) — tune as you learn
    def weights(row):
        reg = row["regime"]
        if reg == "Expanding":
            return pd.Series({"w_q4": 1.00, "w_q1": 0.40, "w_q2": 0.10, "w_fade_bias": 0.10})
        if reg == "Compressing":
            return pd.Series({"w_q4": 0.30, "w_q1": 0.80, "w_q2": 0.20, "w_fade_bias": 0.60})
        # Neutral
        return pd.Series({"w_q4": 0.60, "w_q1": 0.40, "w_q2": 0.20, "w_fade_bias": 0.30})

    weekly = pd.concat([weekly, weekly.apply(weights, axis=1)], axis=1)
    return weekly

# --------------------------------- CLI ------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to raw 1m OHLCV CSV (UTC-4)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--pip_size", type=float, default=0.0001, help="Pip size (EURUSD=0.0001)")
    ap.add_argument("--roll", type=int, default=12, help="Rolling weeks for z-score window")
    ap.add_argument("--z_up", type=float, default=0.75, help="Z threshold for Expanding regime")
    ap.add_argument("--z_down", type=float, default=-0.75, help="Z threshold for Compressing regime")
    ap.add_argument("--week_end", type=str, default="SUN", help="Week ending day: SUN|MON|... (pandas W-<DAY>)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    thr = RegimeThresholds(z_up=args.z_up, z_down=args.z_down)

    df = load_ohlcv(args.csv)
    daily = compute_asia_daily(df, pip_size=args.pip_size)
    weekly = compute_weekly_metrics(daily, week_end=args.week_end, roll=args.roll, thr=thr)

    # Save outputs
    daily.to_csv(os.path.join(args.out, "asia_daily.csv"), index=False)

    weekly.to_csv(os.path.join(args.out, "asia_weekly_metrics.csv"), index=False)

    transitions = weekly[weekly["transition"]].copy()
    transitions.to_csv(os.path.join(args.out, "asia_regime_transitions.csv"), index=False)

    # Console summary (last 8 weeks)
    tail = weekly.tail(8)[[
        "week","days","mean_pips","std_pips","var_pips","z_var","regime","trend","w_q4","w_q1","w_q2","w_fade_bias"
    ]]
    print("\nLast 8 weeks preview:\n", tail.to_string(index=False))
    last = weekly.tail(1).iloc[0]
    print("\nCurrent regime:", last["regime"], "| z_var=", round(float(last["z_var"]), 2), "| trend:", last["trend"])
    print("Suggested weights → Q4:", round(float(last["w_q4"]),2), " Q1:", round(float(last["w_q1"]),2), " Q2:", round(float(last["w_q2"]),2), " FadeBias:", round(float(last["w_fade_bias"]),2))


if __name__ == "__main__":
    main()
