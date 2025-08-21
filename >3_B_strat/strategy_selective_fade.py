#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os

def load_ohlcv(path):
    # Try reading as a headered CSV first
    df = pd.read_csv(path)
    # Normalize column names case-insensitively
    cmap = {c.lower().strip(): c for c in df.columns}
    req = ["date","time","open","high","low","close"]
    # If required columns aren't present, try headerless format with conventional names
    if not all(r in cmap for r in req):
        # common headerless layout used in this workspace
        names = ["date","time","open","high","low","close","volume"]
        df = pd.read_csv(path, header=None, names=names)
        cmap = {c.lower().strip(): c for c in df.columns}
        if not all(r in cmap for r in req):
            missing = [r for r in req if r not in cmap]
            raise ValueError(f"Missing required column(s) {missing} in OHLCV CSV.")
    df["timestamp"] = pd.to_datetime(df[cmap["date"]].astype(str) + " " + df[cmap["time"]].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)
    for k in ["open","high","low","close","volume"]:
        if k in cmap:
            df[k] = pd.to_numeric(df[cmap[k]], errors="coerce")
    return df

def atr(series_high, series_low, series_close, n=5):
    # True Range components
    prev_close = series_close.shift(1)
    tr1 = series_high - series_low
    tr2 = (series_high - prev_close).abs()
    tr3 = (series_low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def backtest_selective_fade(events_path, ohlcv_path, out_dir, rr_target=1.0):
    os.makedirs(out_dir, exist_ok=True)
    ev = pd.read_csv(events_path)
    ev["sweep_time"] = pd.to_datetime(ev["sweep_time"])
    ev["london_end"] = pd.to_datetime(ev["london_end"])
    ev["trading_day_id"] = pd.to_datetime(ev["trading_day_id"]).dt.date

    # Feature enrichment for filters
    ev["dow"] = ev["sweep_time"].dt.day_name()
    ev["hour"] = ev["sweep_time"].dt.hour

    # Load price data and precompute ATR(5)
    px = load_ohlcv(ohlcv_path).copy()
    px["atr5"] = atr(px["high"], px["low"], px["close"], n=5)

    # Index by timestamp for fast slicing
    px = px.set_index("timestamp")

    trades = []

    for idx, r in ev.iterrows():
        # Filters for Strategy #1: only UP sweeps, late London, Thu/Fri
        if r["side"] != "up_sweep":
            continue
        if r["hour"] < 2:  # late = hour >= 2 (02:00–04:00)
            continue
        if r["dow"] not in ["Thursday", "Friday"]:
            continue

        sweep_time = r["sweep_time"]
        london_end = r["london_end"]
        sweep_level = float(r["sweep_level"])
        asia_high = float(r["asia_high"])
        asia_low = float(r["asia_low"])
        asia_mid = (asia_high + asia_low) / 2.0

        # Segment from sweep -> London end
        seg = px.loc[(px.index >= sweep_time) & (px.index <= london_end)].copy()
        if seg.empty:
            continue

        # Entry condition: first close back inside Asia range AND <= sweep_level (since up_sweep, we want re-entry short)
        inside = (seg["close"] <= asia_high) & (seg["close"] >= asia_low) & (seg["close"] <= sweep_level)
        if not inside.any():
            # Skip if never re-entered before London end
            continue
        entry_time = inside.idxmax()  # first True index
        entry_row = seg.loc[entry_time]
        entry_price = float(entry_row["close"])

        # Extension since sweep until entry
        ext_since_sweep = (seg.loc[:entry_time]["high"].max() - sweep_level)
        atr_buf = float(seg.loc[entry_time, "atr5"]) * 0.5 if not np.isnan(seg.loc[entry_time, "atr5"]) else 0.0

        # For short: SL above entry. Conservative: just beyond the sweep level + extension + ATR buffer
        stop_price = float(sweep_level + ext_since_sweep + atr_buf)
        # compute target take-profit from desired reward:risk (rr_target)
        # short: reward per unit = entry_price - take_profit; risk per unit = stop_price - entry_price
        risk_per_unit = max(1e-9, stop_price - entry_price)
        take_profit = float(entry_price - rr_target * risk_per_unit)

        # Walk forward from entry_time to london_end to check TP/SL
        fwd = seg.loc[entry_time:]
        hit_tp_time = None
        hit_sl_time = None

        # For short: TP if low <= take_profit, SL if high >= stop_price
        for ts, row in fwd.iterrows():
            if hit_tp_time is None and row["low"] <= take_profit:
                hit_tp_time = ts
            if hit_sl_time is None and row["high"] >= stop_price:
                hit_sl_time = ts
            if hit_tp_time or hit_sl_time:
                break

        exit_reason = "time"
        exit_time = london_end
        exit_price = float(fwd.iloc[-1]["close"])

        if hit_tp_time and hit_sl_time:
            # If both hit in same bar, assume worst-case: SL first if high >= SL before low <= TP isn't resolvable in 1m bars
            # You can refine with intrabar modeling if needed.
            first_hit_time = min(hit_tp_time, hit_sl_time)
            if first_hit_time == hit_sl_time:
                exit_reason = "SL"
                exit_time = hit_sl_time
                # approximate fill at stop
                exit_price = stop_price
            else:
                exit_reason = "TP"
                exit_time = hit_tp_time
                exit_price = take_profit
        elif hit_tp_time:
            exit_reason = "TP"
            exit_time = hit_tp_time
            exit_price = take_profit
        elif hit_sl_time:
            exit_reason = "SL"
            exit_time = hit_sl_time
            exit_price = stop_price

        # Compute R multiple (short)
        risk_per_unit = max(1e-9, stop_price - entry_price)
        reward_per_unit = entry_price - take_profit
        r_multiple = (reward_per_unit / risk_per_unit) if exit_reason == "TP" else \
                     (-1.0 if exit_reason == "SL" else ((entry_price - exit_price) / risk_per_unit))

        trades.append({
            "trading_day_id": r["trading_day_id"],
            "dow": r["dow"],
            "entry_time": entry_time,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "take_profit": take_profit,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "r_multiple": r_multiple,
            "sweep_level": sweep_level,
            "asia_high": asia_high,
            "asia_low": asia_low,
            "ext_since_sweep": float(ext_since_sweep),
            "atr5_at_entry": float(atr_buf*2) if atr_buf is not None else np.nan,
            "rr_target": rr_target
        })

    trades_df = pd.DataFrame(trades)
    trades_path = os.path.join(out_dir, "strategy_selective_fade_results.csv")
    trades_df.to_csv(trades_path, index=False)

    # Summary
    if not trades_df.empty:
        total = len(trades_df)
        win_rate = (trades_df["r_multiple"] > 0).mean()
        avg_r = trades_df["r_multiple"].mean()
        exp = avg_r  # expectancy in R
        tp_rate = (trades_df["exit_reason"] == "TP").mean()
        sl_rate = (trades_df["exit_reason"] == "SL").mean()

        summary = pd.DataFrame([{
            "trades": total,
            "win_rate": win_rate,
            "avg_R": avg_r,
            "expectancy_R": exp,
            "tp_rate": tp_rate,
            "sl_rate": sl_rate
        }])
    else:
        summary = pd.DataFrame([{
            "trades": 0,
            "win_rate": np.nan,
            "avg_R": np.nan,
            "expectancy_R": np.nan,
            "tp_rate": np.nan,
            "sl_rate": np.nan
        }])

    summary_path = os.path.join(out_dir, "strategy_selective_fade_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Done. Wrote:\n  - {trades_path}\n  - {summary_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="Path to harvest_events.csv")
    ap.add_argument("--csv", required=True, help="Path to raw 1m OHLCV CSV (UTC-4)")
    ap.add_argument("--out", default="./fade_out", help="Output dir")
    ap.add_argument("--rr", type=float, default=1.0, help="Desired reward:risk target (e.g., 2.0 for 2:1)")
    args = ap.parse_args()
    backtest_selective_fade(args.events, args.csv, args.out, rr_target=args.rr)
