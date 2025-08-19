#!/usr/bin/env python3
"""Run ny_asia_imbalance_tracker.py over all CSVs in the `csv/` directory and print the conditional stats CSV for each year.

Assumption: the events file is provided as `results/harvest_events.csv` and is applicable to each run.
If you want per-year events, we can generate harvest events per CSV first and pass those paths instead.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta, time

# Small copy of the harvester logic from main.py so we can generate per-year events
NY_CLOSE_ANCHOR = time(17, 0)
ASIA_KZ_START = time(19, 0)
ASIA_KZ_END = time(21, 0)


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["date", "time", "open", "high", "low", "close"]
    if not all(r in cols for r in required):
        # try headerless
        names = ["date", "time", "open", "high", "low", "close", "volume"]
        df = pd.read_csv(path, header=None, names=names)
        cols = {c.lower().strip(): c for c in df.columns}
        if not all(r in cols for r in required):
            missing = [r for r in required if r not in cols]
            raise ValueError(f"Missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)
    for k in ["open", "high", "low", "close", "volume"]:
        if k in cols:
            df[k] = pd.to_numeric(df[cols[k]], errors="coerce")
    return df


def trading_day_id_from_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.date()
    if ts.time() < NY_CLOSE_ANCHOR:
        d = (ts - timedelta(days=1)).date()
    return pd.Timestamp(d)


def in_window(ts_time: time, start_t: time, end_t: time) -> bool:
    return (ts_time >= start_t) and (ts_time < end_t)


def tag_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trading_day_id"] = df["timestamp"].apply(trading_day_id_from_timestamp)
    df["is_asia_kz"] = df["timestamp"].dt.time.apply(lambda t: in_window(t, ASIA_KZ_START, ASIA_KZ_END))
    return df


def compute_asia_range(df_day: pd.DataFrame):
    asia = df_day[df_day["is_asia_kz"]]
    if asia.empty:
        return None, None, None, None
    asia_high = asia["high"].max()
    asia_low = asia["low"].min()
    hi_row = asia.loc[asia["high"].idxmax()]
    lo_row = asia.loc[asia["low"].idxmin()]
    return asia_high, asia_low, hi_row["timestamp"], lo_row["timestamp"]


def first_london_sweep(df_day: pd.DataFrame, asia_high, asia_low):
    # Use the same London window as main.py (01:00-04:00 UTC-4)
    london = df_day[df_day["timestamp"].dt.time.apply(lambda t: (t >= time(1, 0)) and (t < time(4, 0)))]
    if london.empty or pd.isna(asia_high) or pd.isna(asia_low):
        return None
    breach_high = london[london["high"] > asia_high]
    breach_low = london[london["low"] < asia_low]
    first_high_time = breach_high["timestamp"].iloc[0] if not breach_high.empty else None
    first_low_time = breach_low["timestamp"].iloc[0] if not breach_low.empty else None
    if first_high_time is None and first_low_time is None:
        return None
    if first_high_time is not None and (first_low_time is None or first_high_time <= first_low_time):
        side = "up_sweep"
        sweep_time = first_high_time
        level = asia_high
    else:
        side = "down_sweep"
        sweep_time = first_low_time
        level = asia_low
    london_end = london["timestamp"].max()
    post = london[(london["timestamp"] >= sweep_time) & (london["timestamp"] <= london_end)]
    if post.empty:
        return None
    if side == "up_sweep":
        max_extension = (post["high"].max() - level)
        last_close = post["close"].iloc[-1]
        reverted_inside = (last_close <= level) and (last_close >= df_day["asia_low"].iloc[0])
    else:
        max_extension = (level - post["low"].min())
        last_close = post["close"].iloc[-1]
        reverted_inside = (last_close >= level) and (last_close <= df_day["asia_high"].iloc[0])
    revert_time = None
    for _, r in post.iterrows():
        if (r["close"] <= df_day["asia_high"].iloc[0]) and (r["close"] >= df_day["asia_low"].iloc[0]):
            revert_time = r["timestamp"]
            break
    time_to_revert_min = (revert_time - sweep_time).total_seconds() / 60.0 if revert_time is not None else np.nan
    if side == "up_sweep":
        max_adverse = (post["high"].max() - level)
        max_favorable = (level - post["low"].min())
    else:
        max_adverse = (level - post["low"].min())
        max_favorable = (post["high"].max() - level)
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
        "asia_high": float(df_day["asia_high"].iloc[0]),
        "asia_low": float(df_day["asia_low"].iloc[0]),
    }


def harvest_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, df_day in df.groupby("trading_day_id"):
        asia_high, asia_low, _, _ = compute_asia_range(df_day)
        df_day = df_day.copy()
        df_day["asia_high"] = asia_high
        df_day["asia_low"] = asia_low
        if asia_high is None or asia_low is None:
            continue
        sweep = first_london_sweep(df_day, asia_high, asia_low)
        if sweep is None:
            continue
        rows.append({"trading_day_id": day, **sweep})
    return pd.DataFrame(rows)

DEFAULT_CSV_DIR = "csv"
DEFAULT_PATTERN = r"DAT_MT_.*_(\d{4})\.csv"
DEFAULT_EVENTS = "results/harvest_events.csv"


def run_one(csv_path: Path, out_root: Path, events_path: Path, k: int, pip_size: float):
    m = re.search(DEFAULT_PATTERN, csv_path.name)
    year = m.group(1) if m else csv_path.stem
    out_dir = out_root / year
    out_dir.mkdir(parents=True, exist_ok=True)
    # --- generate per-year events using the harvesting logic ---
    try:
        df = load_ohlcv(csv_path)
        df = tag_sessions(df)
        events_df = harvest_events(df)
        events_file = out_dir / "harvest_events.csv"
        events_df.to_csv(events_file, index=False)
    except Exception as e:
        events_file = out_dir / "harvest_events.csv"
        # write empty file to indicate failure
        events_file.write_text("")

    # now run the imbalance tracker using the per-year events file
    cmd = [
        "python",
        "ny_asia_imbalance_tracker.py",
        "--ohlcv",
        str(csv_path),
        "--events",
        str(events_file),
        "--out",
        str(out_dir),
        "--pip_size",
        str(pip_size),
        "--k",
        str(k),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    # save logs
    (out_dir / "run_stdout.log").write_text(p.stdout)
    (out_dir / "run_stderr.log").write_text(p.stderr)

    stats_file = out_dir / "ny_asia_conditional_stats.csv"
    stats = None
    if stats_file.exists():
        stats = stats_file.read_text()
    return {"year": year, "out": str(out_dir), "returncode": p.returncode, "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    ap.add_argument("--out-root", default="results/imbalance_all")
    ap.add_argument("--events", default=DEFAULT_EVENTS)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pip_size", type=float, default=0.0001)
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_root = Path(args.out_root)
    events_path = Path(args.events)

    files = sorted([p for p in csv_dir.glob("*.csv") if re.search(DEFAULT_PATTERN, p.name)])
    if not files:
        print("No CSV files found in", csv_dir)
        return

    results = []
    for f in files:
        print(f"Running imbalance tracker for {f.name} -> {out_root}/{f.stem[-4:]} ...")
        r = run_one(f, out_root, events_path, args.k, args.pip_size)
        results.append(r)

    # Print only the conditional stats for each year
    print("\nConditional stats per year:\n")
    for r in results:
        print("====", r['year'], "====")
        if r['stats']:
            print(r['stats'])
        else:
            print("(no ny_asia_conditional_stats.csv found, returncode=", r['returncode'], ")\n")

if __name__ == '__main__':
    main()
