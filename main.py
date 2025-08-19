import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

# ------------------ CONFIG ------------------
CSV_PATH = "csv/DAT_MT_EURUSD_M1_2024.csv"   # <-- run harvester on 2023 dataset for this task
HAS_HEADER = False                         # CSV is headerless in this workspace
COLUMN_NAMES = ["date", "time", "open", "high", "low", "close", "volume"]  # used only if HAS_HEADER=False

# Session/KZ (UTC-4) definitions
NY_CLOSE_ANCHOR = time(17, 0)    # trading day boundary at 17:00 (NY close)
ASIA_KZ_START   = time(19, 0)
ASIA_KZ_END     = time(21, 0)
LONDON_KZ_START = time(1, 0)
LONDON_KZ_END   = time(4, 0)
NY_KZ_START     = time(8, 30)
NY_KZ_END       = time(11, 0)

# Outcome rule: classify "revert_inside" if, by LONDON_KZ_END, close is back within Asia range
# Otherwise classify as "continuation"
# -------------------------------------------

def load_ohlcv(path, has_header=True, names=None):
    if has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=names)
    # Try to handle common column name variants
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["date","time","open","high","low","close"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing required column '{r}' (case-insensitive) in CSV.")
    # Build timestamp (naive, already UTC-4)
    df["timestamp"] = pd.to_datetime(df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Normalize numeric columns
    for k in ["open","high","low","close","volume"]:
        if k in cols:
            df[k] = pd.to_numeric(df[cols[k]], errors="coerce")
    return df

def trading_day_id_from_timestamp(ts):
    # Trading day boundary at 17:00. If time < 17:00, belongs to previous "trading day date"
    d = ts.date()
    if ts.time() < NY_CLOSE_ANCHOR:
        d = (ts - timedelta(days=1)).date()
    return pd.Timestamp(d)

def in_window(ts_time, start_t, end_t):
    # all times naive UTC-4; handle windows that do not cross midnight only
    return (ts_time >= start_t) and (ts_time < end_t)

def tag_sessions(df):
    # trading day id
    df["trading_day_id"] = df["timestamp"].apply(trading_day_id_from_timestamp)
    # tags for KZ
    df["is_asia_kz"]   = df["timestamp"].dt.time.apply(lambda t: in_window(t, ASIA_KZ_START, ASIA_KZ_END))
    df["is_london_kz"] = df["timestamp"].dt.time.apply(lambda t: in_window(t, LONDON_KZ_START, LONDON_KZ_END))
    df["is_ny_kz"]     = df["timestamp"].dt.time.apply(lambda t: in_window(t, NY_KZ_START, NY_KZ_END))
    return df

def compute_asia_range(df_day):
    asia = df_day[df_day["is_asia_kz"]]
    if asia.empty:
        return None, None, None, None
    asia_high = asia["high"].max()
    asia_low  = asia["low"].min()
    # timestamps of extremes
    hi_row = asia.loc[asia["high"].idxmax()]
    lo_row = asia.loc[asia["low"].idxmin()]
    return asia_high, asia_low, hi_row["timestamp"], lo_row["timestamp"]

def first_london_sweep(df_day, asia_high, asia_low):
    london = df_day[df_day["is_london_kz"]]
    if london.empty or pd.isna(asia_high) or pd.isna(asia_low):
        return None
    # Detect first time high is exceeded or low is undercut
    breach_high = london[london["high"] > asia_high]
    breach_low  = london[london["low"]  < asia_low]
    first_high_time = breach_high["timestamp"].iloc[0] if not breach_high.empty else None
    first_low_time  = breach_low["timestamp"].iloc[0] if not breach_low.empty else None

    if first_high_time is None and first_low_time is None:
        return None

    # Determine which came first
    if first_high_time is not None and (first_low_time is None or first_high_time <= first_low_time):
        side = "up_sweep"   # sweep Asia high
        sweep_time = first_high_time
        level = asia_high
    else:
        side = "down_sweep" # sweep Asia low
        sweep_time = first_low_time
        level = asia_low

    # Collect post-sweep segment until London KZ end
    london_end = london["timestamp"].max()
    post = london[(london["timestamp"] >= sweep_time) & (london["timestamp"] <= london_end)]
    if post.empty:
        return None

    # Extension beyond level and whether price closes back inside range by london end
    if side == "up_sweep":
        max_extension = (post["high"].max() - level)
        last_close = post["close"].iloc[-1]
        reverted_inside = (last_close <= level) and (last_close >= df_day["asia_low"].iloc[0])
    else:
        max_extension = (level - post["low"].min())
        last_close = post["close"].iloc[-1]
        reverted_inside = (last_close >= level) and (last_close <= df_day["asia_high"].iloc[0])

    # Time-to-revert (first time close returns inside range)
    def inside_range(c):
        return (c <= df_day["asia_high"].iloc[0]) and (c >= df_day["asia_low"].iloc[0])

    revert_time = None
    for _, r in post.iterrows():
        if inside_range(r["close"]):
            revert_time = r["timestamp"]
            break
    time_to_revert_min = (revert_time - sweep_time).total_seconds()/60.0 if revert_time is not None else np.nan

    # Max adverse move if fading the sweep immediately at level (rough proxy)
    if side == "up_sweep":
        max_adverse = (post["high"].max() - level)
        max_favorable = (level - post["low"].min())  # favorable for a short back inside
    else:
        max_adverse = (level - post["low"].min())
        max_favorable = (post["high"].max() - level)  # favorable for a long back inside

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

def harvest_events(df):
    rows = []
    for day, df_day in df.groupby("trading_day_id"):
        asia_high, asia_low, asia_hi_t, asia_lo_t = compute_asia_range(df_day)
        df_day = df_day.copy()
        df_day["asia_high"] = asia_high
        df_day["asia_low"] = asia_low

        if asia_high is None or asia_low is None:
            continue

        sweep = first_london_sweep(df_day, asia_high, asia_low)
        if sweep is None:
            continue

        rows.append({
            "trading_day_id": day,
            **sweep
        })
    events = pd.DataFrame(rows)
    return events

def summarize(events):
    if events.empty:
        return pd.DataFrame()
    # Basic outcome stats
    total = len(events)
    revert_rate = events["reverted_inside_by_end"].mean()
    up_down = events["side"].value_counts(normalize=True).rename("proportion")
    # Distance summaries (use raw price units; user can convert to pips externally)
    dist_summary = events[["max_extension", "max_adverse", "max_favorable", "time_to_revert_min"]].describe().T
    dist_summary["metric"] = dist_summary.index
    dist_summary = dist_summary.reset_index(drop=True)
    # By side
    by_side = events.groupby("side")["reverted_inside_by_end"].mean().rename("revert_rate").reset_index()
    summary = {
        "total_days_with_sweep": total,
        "overall_revert_rate_by_london_end": float(revert_rate),
    }
    # Build a compact table
    out = pd.DataFrame([summary])
    return out, up_down.reset_index().rename(columns={"index":"side"}), dist_summary, by_side

# --------- RUN (safe even without a file; will just skip if not found) ----------
try:
    df = load_ohlcv(CSV_PATH, HAS_HEADER, COLUMN_NAMES if not HAS_HEADER else None)
    df = tag_sessions(df)
    events = harvest_events(df)
    events_path = "results/harvest_events.csv"
    summary_path = "results/harvest_summary.csv"

    if not events.empty:
        # Build expanded summaries
        overall, side_prop, dist_stats, by_side = summarize(events)

        # Save artifacts
        os.makedirs(os.path.dirname(events_path), exist_ok=True)
        events.to_csv(events_path, index=False)
        with pd.ExcelWriter("results/harvest_reports.xlsx") as xw:
            events.to_excel(xw, sheet_name="events", index=False)
            overall.to_excel(xw, sheet_name="overall", index=False)
            side_prop.to_excel(xw, sheet_name="side_proportion", index=False)
            dist_stats.to_excel(xw, sheet_name="distance_stats", index=False)
            by_side.to_excel(xw, sheet_name="by_side_revert_rate", index=False) 
            overall.to_csv(summary_path, index=False)

        # optional display helper may not exist in this environment
        try:
            from caas_jupyter_tools import display_dataframe_to_user
            display_dataframe_to_user("London Sweep Events (by trading day)", events)
        except Exception:
            pass

        print("Artifacts written:")
        print(f"- Events CSV: {events_path}")
        print(f"- Summary CSV: {summary_path}")
        print("- Excel report: results/harvest_reports.xlsx")
    else:
        print("No events harvested (possibly no sweeps detected or file missing).")
        print("Check CSV_PATH and column mapping.")

except FileNotFoundError:
    print("CSV not found. Upload your CSV and update CSV_PATH.")
except Exception as e:
    print("Error:", e)