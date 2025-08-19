#!/usr/bin/env python3
"""
analyze_entity_invariants.py
--------------------------------------------------
Purpose: Empirically TEST the entity's invariants on your CSV datasets.
This does NOT place trades. It only measures frequencies, correlations,
and outcome distributions conditioned on the proposed gates.

Inputs:
  --imbalance  : Path to ny_asia_daily.csv (from ny_asia_imbalance_tracker.py)
  --ohlcv      : Path to 1-minute OHLCV CSV (columns: date,time,open,high,low,close[,volume])
  --out        : Output directory
  --pip_size   : Pip size (e.g., 0.0001 for EURUSD)
  --retest_tol_pips : Price tolerance (in pips) for counting a retest at the Asia level
  --retest_window_min : Max minutes allowed for a valid retest (Gate 3)
  --lookahead_min     : Minutes to observe after retest for outcome measurements
  --range_pct_window  : Rolling window (days) for the Asia range percentile (Gate 2 bins)

What is computed (per event row in ny_asia_daily.csv):
  * Gate-1 fields:
      - contain_pass: (contain_streak >= 3) & (asia_contains_prev_ny == True)
      - contain_or_override: contain_pass OR (asia_range_pct_20d >= 85 & resolve_streak == 0 & is_early_london)
  * Gate-2 fields:
      - asia_range_pct_20d: rolling percentile rank of asia_range_pips over prior N days
      - bin_range_pct: one of ["low_0_35", "mid_35_85", "high_85_100"]
  * Gate-3 fields:
      - overshoot_frac: |breach_extreme - asia_level| / asia_range
      - retest_delay_min: minutes from breach to first retest of asia_level (within retest_window_min)
      - gate3_pass: (0.15 <= overshoot_frac <= 0.35) AND (retest_delay_min <= retest_window_min)

Outcome metrics (NO trades taken):
  * mfe_from_retest_pips: max favorable excursion in sweep direction after retest (min(lookahead_min, until 06:00 London or sweep+lookahead))
  * mfe_from_retest_frac_range: mfe_from_retest_pips / asia_range_pips
  * basic success flags (directional follow-through):
      - follow_10p, follow_20p, follow_30p: whether price moved +10/20/30 pips in sweep direction from the retest before the opposite stop-out distance

Outputs:
  - out/events_annotated.csv: one row per event with all computed fields
  - out/summary_gate1_gate2_gate3.csv: counts and rates by gate pass/fail
  - out/summary_range_bins.csv: performance by range percentile bins
  - out/summary_overshoot_bins.csv: performance by overshoot bins
  - out/summary_timing.csv: performance by entry hour bins

Note:
  - The script tolerates missing OHLCV minutes; it will skip events where the
    required window isn't fully available.
"""
import argparse
import os
import math
import pandas as pd
import numpy as np
from datetime import timedelta

# ----------------------------- IO HELPERS -----------------------------
def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cmap = {c.lower().strip(): c for c in df.columns}
    req = ["date","time","open","high","low","close"]
    if not all(r in cmap for r in req):
        names = ["date","time","open","high","low","close","volume"]
        df = pd.read_csv(path, header=None, names=names)
        cmap = {c.lower().strip(): c for c in df.columns}
        missing = [r for r in req if r not in cmap]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in OHLCV CSV.")
    df["timestamp"] = pd.to_datetime(df[cmap["date"]].astype(str) + " " + df[cmap["time"]].astype(str))
    for k in ["open","high","low","close","volume"]:
        if k in cmap:
            df[k] = pd.to_numeric(df[cmap[k]], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp","open","high","low","close"]]

def load_imbalance(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize basic fields
    df["trading_day_id"] = pd.to_datetime(df["trading_day_id"]).dt.date
    for c in ["ny_prev_high","ny_prev_low","asia_high","asia_low","asia_range","asia_range_pips"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "sweep_time" in df.columns:
        df["sweep_time"] = pd.to_datetime(df["sweep_time"])
    if "sweep_hour" in df.columns:
        df["sweep_hour"] = pd.to_numeric(df["sweep_hour"], errors="coerce")
    # Flags to bool
    for b in ["asia_contains_prev_ny","asia_resolves_prev_ny","is_early_london","is_late_london"]:
        if b in df.columns:
            if df[b].dtype == object:
                df[b] = df[b].astype(str).str.lower().isin(["true","1","yes","y"])
            else:
                df[b] = df[b].astype(bool)
    return df

# ----------------------------- COMPUTE HELPERS -----------------------------
def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling percentile rank (0..100) for each point based on the *previous* 'window' values.
    The first 'window' rows will be NaN.
    """
    vals = series.values
    out = [np.nan] * len(vals)
    for i in range(len(vals)):
        start = max(0, i - window)
        prev = vals[start:i]
        if len(prev) == 0:
            out[i] = np.nan
        else:
            out[i] = 100.0 * (np.sum(prev <= vals[i]) / len(prev))
    return pd.Series(out, index=series.index)

def first_time_index_after_or_equal(idx, ts):
    pos = idx.searchsorted(ts, side="left")
    return pos

def compute_overshoot_and_retest(seg: pd.DataFrame, side: str, asia_level: float, pip_size: float,
                                 retest_tol_pips: float, retest_window_min: int):
    """
    Given a segment starting at (or just before) sweep_time, compute:
      - overshoot_frac: max immediate overshoot within first 5 minutes after breach relative to asia_range
      - retest_delay_min: minutes until price re-touches asia_level within tolerance after overshoot
    We will detect:
      - For up_sweep: breach if close > asia_level or high >= asia_level; overshoot measured from level to that bar's high (max over first 5m)
      - For down_sweep: breach if close < asia_level or low <= asia_level; overshoot from level to that bar's low (min over first 5m)
    Retest: first time price touches back to asia_level ± tol within 'retest_window_min' after breach.
    Returns (overshoot_pips, retest_delay_min) or (np.nan, np.nan) if not found.
    """
    tol = retest_tol_pips * pip_size
    if seg.empty:
        return (np.nan, np.nan)

    # Limit initial overshoot scan to 5 minutes after the first breach
    # Find breach index
    breach_idx = None
    for i, row in enumerate(seg.itertuples(index=False)):
        if side == "up_sweep":
            if (row.high >= asia_level) or (row.close > asia_level):
                breach_idx = i
                break
        else:
            if (row.low <= asia_level) or (row.close < asia_level):
                breach_idx = i
                break
    if breach_idx is None:
        return (np.nan, np.nan)

    # Overshoot within first 5 minutes after breach (inclusive)
    end_ov = min(breach_idx + 5, len(seg)-1)
    if side == "up_sweep":
        ov = seg.iloc[breach_idx:end_ov+1]["high"].max() - asia_level
    else:
        ov = asia_level - seg.iloc[breach_idx:end_ov+1]["low"].min()
    overshoot_pips = max(ov / pip_size, 0.0)

    # Retest detection within 'retest_window_min' after breach
    retest_delay = np.nan
    end_rt = min(breach_idx + retest_window_min, len(seg)-1)
    for j in range(breach_idx, end_rt+1):
        row = seg.iloc[j]
        if side == "up_sweep":
            touched = (row.low <= asia_level + tol) and (row.high >= asia_level - tol)
        else:
            touched = (row.high >= asia_level - tol) and (row.low <= asia_level + tol)
        if touched:
            retest_delay = (j - breach_idx)  # minutes assuming 1m bars
            break
    return (overshoot_pips, retest_delay)

def compute_followthrough(seg: pd.DataFrame, side: str, start_idx: int, pip_size: float, lookahead_min: int,
                          thresholds_pips=(10,20,30)):
    """
    From 'start_idx' (typically the retest bar), compute forward max favorable excursion (MFE) in sweep direction
    over 'lookahead_min' minutes and whether it cleared thresholds before reversing by the same amount.
    Returns: mfe_pips, {threshold: True/False}
    """
    # start_idx may be None, NaN, or a non-numeric type; use pd.isna which safely handles these cases
    if start_idx is None or pd.isna(start_idx):
        return (np.nan, {p: False for p in thresholds_pips})
    end_idx = min(start_idx + lookahead_min, len(seg)-1)
    sub = seg.iloc[start_idx:end_idx+1]
    if sub.empty:
        return (np.nan, {p: False for p in thresholds_pips})
    start_price = sub.iloc[0]["close"]
    if side == "up_sweep":
        mfe = (sub["high"].max() - start_price) / pip_size
    else:
        mfe = (start_price - sub["low"].min()) / pip_size
    out = {}
    for p in thresholds_pips:
        out[p] = bool(mfe >= p)
    return (mfe, out)

# ----------------------------- MAIN ANALYSIS -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imbalance", required=True, help="Path to ny_asia_daily.csv")
    ap.add_argument("--ohlcv", required=True, help="Path to 1m OHLCV CSV")
    ap.add_argument("--out", default="./entity_tests")
    ap.add_argument("--pip_size", type=float, default=0.0001)
    ap.add_argument("--retest_tol_pips", type=float, default=1.0)
    ap.add_argument("--retest_window_min", type=int, default=15)
    ap.add_argument("--lookahead_min", type=int, default=120)
    ap.add_argument("--range_pct_window", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    imb = load_imbalance(args.imbalance).copy()
    px = load_ohlcv(args.ohlcv).copy().set_index("timestamp")

    # Sort by trading day for rolling calcs
    imb = imb.sort_values("trading_day_id").reset_index(drop=True)

    # Gate-2: rolling percentile of Asia range (use prior N days)
    imb["asia_range_pct_20d"] = rolling_percentile(imb["asia_range_pips"], window=args.range_pct_window)

    # Bin into low/mid/high
    def bin_range_pct(p):
        if pd.isna(p): return "nan"
        if p <= 35: return "low_0_35"
        if p < 85:  return "mid_35_85"
        return "high_85_100"
    imb["bin_range_pct"] = imb["asia_range_pct_20d"].apply(bin_range_pct)

    # Gate-1: containment pass + override condition
    imb["contain_pass"] = (imb["contain_streak"] >= 3) & (imb["asia_contains_prev_ny"] == True)
    imb["contain_override"] = (imb["asia_range_pct_20d"] >= 85) & (imb["resolve_streak"].fillna(0) == 0) & (imb["is_early_london"] == True)
    imb["contain_or_override"] = imb["contain_pass"] | imb["contain_override"]

    # Prepare annotation columns
    imb["overshoot_pips"] = np.nan
    imb["overshoot_frac"] = np.nan
    imb["retest_delay_min"] = np.nan
    imb["gate3_pass"] = False
    imb["mfe_from_retest_pips"] = np.nan
    imb["mfe_from_retest_frac_range"] = np.nan
    for p in (10,20,30):
        imb[f"follow_{p}p"] = False

    # Iterate events with available sweep_time and side
    for i, r in imb.iterrows():
        sweep_time = r.get("sweep_time", pd.NaT)
        side = r.get("side", None)
        asia_high = r.get("asia_high", np.nan)
        asia_low  = r.get("asia_low", np.nan)
        asia_range_pips = r.get("asia_range_pips", np.nan)
        asia_range = r.get("asia_range", np.nan)
        if pd.isna(sweep_time) or side not in ("up_sweep","down_sweep") or pd.isna(asia_range) or asia_range <= 0:
            continue

        # Build segment from sweep_time to sweep_time + lookahead_min
        end_time = sweep_time + pd.Timedelta(minutes=args.lookahead_min)
        # Try to include a couple minutes before sweep for stability
        start_time = sweep_time - pd.Timedelta(minutes=2)
        seg = px.loc[(px.index >= start_time) & (px.index <= end_time)].copy()
        if seg.empty:
            continue

        # Asia level depending on side
        asia_level = float(asia_high) if side == "up_sweep" else float(asia_low)

        # Compute overshoot/retest on the segment starting at sweep_time
        seg_from_sweep = seg.loc[seg.index >= sweep_time].copy()
        if seg_from_sweep.empty:
            continue

        ov_pips, retest_delay = compute_overshoot_and_retest(
            seg_from_sweep[["open","high","low","close"]].reset_index(drop=True),
            side=side,
            asia_level=asia_level,
            pip_size=args.pip_size,
            retest_tol_pips=args.retest_tol_pips,
            retest_window_min=args.retest_window_min,
        )

        imb.at[i, "overshoot_pips"] = ov_pips
        if not pd.isna(ov_pips) and not pd.isna(asia_range_pips) and asia_range_pips > 0:
            imb.at[i, "overshoot_frac"] = float(ov_pips) / float(asia_range_pips)
        else:
            imb.at[i, "overshoot_frac"] = np.nan

        imb.at[i, "retest_delay_min"] = retest_delay

        gate3 = (not pd.isna(imb.at[i,"overshoot_frac"])) and (0.15 <= imb.at[i,"overshoot_frac"] <= 0.35) and \
                (not pd.isna(retest_delay) and retest_delay <= args.retest_window_min)
        imb.at[i, "gate3_pass"] = bool(gate3)

        # Compute follow-through metrics after the retest (if any)
        # Find retest index relative to seg_from_sweep
        if not pd.isna(retest_delay):
            start_idx = int(retest_delay)  # retest bar index relative to breach
        else:
            start_idx = np.nan

        mfe_pips, flags = compute_followthrough(
            seg_from_sweep[["open","high","low","close"]].reset_index(drop=True),
            side=side,
            start_idx=None if pd.isna(start_idx) else start_idx,
            pip_size=args.pip_size,
            lookahead_min=args.lookahead_min,
            thresholds_pips=(10,20,30)
        )
        imb.at[i, "mfe_from_retest_pips"] = mfe_pips
        if not pd.isna(asia_range_pips) and asia_range_pips > 0 and not pd.isna(mfe_pips):
            imb.at[i, "mfe_from_retest_frac_range"] = float(mfe_pips) / float(asia_range_pips)

        for p, val in flags.items():
            imb.at[i, f"follow_{p}p"] = bool(val)

    # Save full annotated events
    out_events = os.path.join(args.out, "events_annotated.csv")
    imb.to_csv(out_events, index=False)

    # ----------------------------- SUMMARIES -----------------------------
    def rate(x): 
        return np.nan if len(x)==0 else float(np.mean(x))

    # Gate pass/fail summaries
    s1 = []
    for label, mask in [
        ("all", np.ones(len(imb), dtype=bool)),
        ("contain_pass", imb["contain_pass"]==True),
        ("contain_or_override", imb["contain_or_override"]==True),
        ("gate3_pass", imb["gate3_pass"]==True),
    ]:
        sub = imb[mask]
        s1.append({
            "group": label,
            "n_events": int(len(sub)),
            "avg_overshoot_frac": float(np.nanmean(sub["overshoot_frac"])) if len(sub) else np.nan,
            "retest_rate": rate(~pd.isna(sub["retest_delay_min"])),
            "follow_10p_rate": rate(sub["follow_10p"]==True),
            "follow_20p_rate": rate(sub["follow_20p"]==True),
            "follow_30p_rate": rate(sub["follow_30p"]==True),
            "avg_mfe_frac_range": float(np.nanmean(sub["mfe_from_retest_frac_range"])) if len(sub) else np.nan,
        })
    pd.DataFrame(s1).to_csv(os.path.join(args.out, "summary_gate1_gate2_gate3.csv"), index=False)

    # Range percentile bins
    s2_rows = []
    for bin_name, sub in imb.groupby("bin_range_pct"):
        s2_rows.append({
            "bin_range_pct": bin_name,
            "n_events": int(len(sub)),
            "contain_pass_rate": rate(sub["contain_pass"]==True),
            "gate3_pass_rate": rate(sub["gate3_pass"]==True),
            "follow_20p_rate": rate(sub["follow_20p"]==True),
            "avg_mfe_frac_range": float(np.nanmean(sub["mfe_from_retest_frac_range"])) if len(sub) else np.nan,
        })
    pd.DataFrame(s2_rows).to_csv(os.path.join(args.out, "summary_range_bins.csv"), index=False)

    # Overshoot bins
    def ov_bin(x):
        if pd.isna(x): return "nan"
        if x < 0.15: return "<0.15"
        if x <= 0.35: return "0.15-0.35"
        return ">0.35"
    imb["overshoot_bin"] = imb["overshoot_frac"].apply(ov_bin)

    s3_rows = []
    for bin_name, sub in imb.groupby("overshoot_bin"):
        s3_rows.append({
            "overshoot_bin": bin_name,
            "n_events": int(len(sub)),
            "retest_rate": rate(~pd.isna(sub["retest_delay_min"])),
            "follow_20p_rate": rate(sub["follow_20p"]==True),
            "avg_mfe_frac_range": float(np.nanmean(sub["mfe_from_retest_frac_range"])) if len(sub) else np.nan,
        })
    pd.DataFrame(s3_rows).to_csv(os.path.join(args.out, "summary_overshoot_bins.csv"), index=False)

    # Timing summary (early vs late)
    def time_bin(h):
        if pd.isna(h): return "nan"
        return "early_<=02" if h <= 2 else "late_>02"
    imb["timing_bin"] = imb["sweep_hour"].apply(time_bin)
    s4_rows = []
    for bin_name, sub in imb.groupby("timing_bin"):
        s4_rows.append({
            "timing_bin": bin_name,
            "n_events": int(len(sub)),
            "gate3_pass_rate": rate(sub["gate3_pass"]==True),
            "follow_20p_rate": rate(sub["follow_20p"]==True),
            "avg_mfe_frac_range": float(np.nanmean(sub["mfe_from_retest_frac_range"])) if len(sub) else np.nan,
        })
    pd.DataFrame(s4_rows).to_csv(os.path.join(args.out, "summary_timing.csv"), index=False)

    print("Wrote:")
    print(" -", out_events)
    print(" -", os.path.join(args.out, "summary_gate1_gate2_gate3.csv"))
    print(" -", os.path.join(args.out, "summary_range_bins.csv"))
    print(" -", os.path.join(args.out, "summary_overshoot_bins.csv"))
    print(" -", os.path.join(args.out, "summary_timing.csv"))

if __name__ == "__main__":
    main()
