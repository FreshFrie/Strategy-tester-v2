#!/usr/bin/env python3
"""
analyze_invariant_core.py
--------------------------------------------------
Purpose: Empirically test the invariant "compression + imbalance -> resolution"
using your existing daily NY→Asia stats and M1 OHLCV tape.

Inputs
------
--imbalance          Path to ny_asia_daily.csv
--ohlcv              Path to 1-minute EURUSD OHLCV CSV
--out                Output directory (created if missing)
--pip_size           Pip size (EURUSD=0.0001)
--range_pct_window   Rolling window (days) for percentile ranks (default 20)
--lookahead_min      Minutes to observe London after sweep_time (default 180)
--tol_pips           Touch tolerance (in pips) when checking boundary hits (default 1.0)

Definitions
-----------
Compression:
  - asia_range_pct_20d: rolling percentile rank of Asia range (low == compressed).
  - ny_range_pct_20d: rolling percentile rank of the previous NY range.
  - compression_flag: (asia_range_pct_20d <= 20) OR (ny_range_pct_20d <= 20).
  - compression_score: 1 - min(asia_pct, ny_pct)/100  (higher = more compressed).

Imbalance vs prior NY:
  - ny_range = ny_prev_high - ny_prev_low
  - upward_displacement = max(0, asia_high - ny_prev_high) / ny_range
  - downward_displacement = max(0, ny_prev_low - asia_low) / ny_range
  - imbalance_score = max(upward_displacement, downward_displacement)
  (If ny_range <= 0 or missing → imbalance_score = NaN)

Resolution (tested on M1 after sweep_time):
  Given sweep side ('up_sweep'/'down_sweep') and Asia bounds:
  - Resolution to opposite Asia bound: did price touch the opposite Asia bound (± tol) within lookahead?
      * up_sweep: touch asia_low
      * down_sweep: touch asia_high
  - Resolution to NY midpoint: did price cross ny_mid = (ny_prev_high + ny_prev_low)/2?
  We record time_to_resolution (minutes) and boolean flags for each type.

Outputs
-------
- events_invariant_annotated.csv : Per-event features & outcomes
- summary_compression_imbalance.csv : Grid of compression tertiles × imbalance tertiles with resolution rates
- summary_resolution_times.csv : Distribution of time-to-resolution by bins
- summary_cuts.csv : Simple binary cuts (compression_flag, imbalance>k) with rates

This script does NOT place trades.
"""
import argparse
import os
import math
import pandas as pd
import numpy as np

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

def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trading_day_id"] = pd.to_datetime(df["trading_day_id"]).dt.date
    for c in ["ny_prev_high","ny_prev_low","asia_high","asia_low","asia_range","asia_range_pips"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sweep_time"] = pd.to_datetime(df["sweep_time"])
    return df

def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
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

# ----------------------------- CORE LOGIC -----------------------------
def analyze(imb_path: str, ohlcv_path: str, out_dir: str,
            pip_size: float = 0.0001, range_pct_window: int = 20,
            lookahead_min: int = 180, tol_pips: float = 1.0):
    os.makedirs(out_dir, exist_ok=True)

    imb = load_daily(imb_path).sort_values("trading_day_id").reset_index(drop=True)
    px = load_ohlcv(ohlcv_path).set_index("timestamp")

    # NY range
    imb["ny_range"] = (imb["ny_prev_high"] - imb["ny_prev_low"]).astype(float)

    # Percentiles (compression proxies)
    imb["asia_range_pct_20d"] = rolling_percentile(imb["asia_range_pips"], range_pct_window)
    ny_range_pips = (imb["ny_range"] / (pip_size if pip_size>0 else 1)).replace([np.inf,-np.inf], np.nan)
    imb["ny_range_pct_20d"] = rolling_percentile(ny_range_pips, range_pct_window)

    # Compression features
    imb["compression_flag"] = (imb["asia_range_pct_20d"] <= 20) | (imb["ny_range_pct_20d"] <= 20)
    def comp_score(row):
        vals = [row.get("asia_range_pct_20d", np.nan), row.get("ny_range_pct_20d", np.nan)]
        vals = [v for v in vals if not pd.isna(v)]
        if not vals: return np.nan
        return 1.0 - min(vals)/100.0
    imb["compression_score"] = imb.apply(comp_score, axis=1)

    # Imbalance features
    def imbalance_row(row):
        nyr = row.get("ny_range", np.nan)
        if pd.isna(nyr) or nyr <= 0:
            return np.nan
        up = max(0.0, float(row["asia_high"]) - float(row["ny_prev_high"])) / nyr if not pd.isna(row["asia_high"]) and not pd.isna(row["ny_prev_high"]) else np.nan
        dn = max(0.0, float(row["ny_prev_low"]) - float(row["asia_low"])) / nyr if not pd.isna(row["asia_low"]) and not pd.isna(row["ny_prev_low"]) else np.nan
        return max(up if not pd.isna(up) else 0.0, dn if not pd.isna(dn) else 0.0)
    imb["imbalance_score"] = imb.apply(imbalance_row, axis=1)

    # Resolution checks on M1
    tol = tol_pips * pip_size
    results = []
    for i, r in imb.iterrows():
        sweep_time = r.get("sweep_time", pd.NaT)
        side = r.get("side", None)
        if pd.isna(sweep_time) or side not in ("up_sweep","down_sweep"):
            # still keep the row with NaNs for outcome
            res = dict()
            res["resolved_to_asia_opposite"] = np.nan
            res["resolved_to_ny_mid"] = np.nan
            res["time_to_res_asia_min"] = np.nan
            res["time_to_res_ny_mid_min"] = np.nan
            results.append(res)
            continue

        # Build forward segment (a few minutes before for stability)
        start_time = sweep_time - pd.Timedelta(minutes=2)
        end_time = sweep_time + pd.Timedelta(minutes=lookahead_min)
        seg = px.loc[(px.index >= start_time) & (px.index <= end_time)].copy()
        if seg.empty:
            res = dict(resolved_to_asia_opposite=np.nan, resolved_to_ny_mid=np.nan,
                       time_to_res_asia_min=np.nan, time_to_res_ny_mid_min=np.nan)
            results.append(res)
            continue

        asia_high = float(r["asia_high"]) if not pd.isna(r["asia_high"]) else np.nan
        asia_low  = float(r["asia_low"])  if not pd.isna(r["asia_low"]) else np.nan
        ny_mid = (float(r["ny_prev_high"]) + float(r["ny_prev_low"])) / 2.0 if not (pd.isna(r["ny_prev_high"]) or pd.isna(r["ny_prev_low"])) else np.nan

        # Define targets
        if side == "up_sweep":
            target_asia = asia_low  # opposite bound
            sweep_dir = +1
        else:
            target_asia = asia_high
            sweep_dir = -1

        # Scan forward from sweep_time for resolution
        seg_fwd = seg.loc[seg.index >= sweep_time]
        res_asia = np.nan
        res_mid  = np.nan
        t_asia = np.nan
        t_mid  = np.nan

        for j, (ts, rowp) in enumerate(seg_fwd.iterrows()):
            # Asia opposite bound touch?
            if not pd.isna(target_asia):
                if (rowp["low"] <= target_asia + tol) and (rowp["high"] >= target_asia - tol):
                    res_asia = True
                    t_asia = j  # minutes from sweep
                    # don't break; keep scanning for mid time as well

            # NY mid cross?
            if not pd.isna(ny_mid):
                crossed = (rowp["low"] <= ny_mid <= rowp["high"])
                if crossed and (np.isnan(res_mid)):
                    res_mid = True
                    t_mid = j

            if (not np.isnan(t_asia)) and (not np.isnan(t_mid)):
                break

        results.append(dict(
            resolved_to_asia_opposite=bool(res_asia) if not np.isnan(res_asia) else False,
            resolved_to_ny_mid=bool(res_mid) if not np.isnan(res_mid) else False,
            time_to_res_asia_min=(int(t_asia) if not np.isnan(t_asia) else np.nan),
            time_to_res_ny_mid_min=(int(t_mid) if not np.isnan(t_mid) else np.nan),
        ))

    res_df = pd.DataFrame(results)
    out = pd.concat([imb, res_df], axis=1)

    # Save annotated events
    out_events = os.path.join(out_dir, "events_invariant_annotated.csv")
    out.to_csv(out_events, index=False)

    # ------------------ Summaries: compression x imbalance grid ------------------
    def tertile(s):
        # rank into ~equal-sized bins ignoring NaNs
        q = s.rank(pct=True)
        bins = pd.cut(q, [0, 1/3, 2/3, 1], labels=["low","mid","high"], include_lowest=True)
        return bins

    out["comp_tertile"] = tertile(out["compression_score"])
    out["imb_tertile"] = tertile(out["imbalance_score"])

    # Resolution rates in grid
    grid = out.groupby(["comp_tertile","imb_tertile"]).agg(
        n=("trading_day_id","count"),
        res_asia_rate=("resolved_to_asia_opposite", "mean"),
        res_mid_rate=("resolved_to_ny_mid", "mean"),
        t_asia_med=("time_to_res_asia_min", "median"),
        t_mid_med=("time_to_res_ny_mid_min", "median"),
    ).reset_index()
    grid.to_csv(os.path.join(out_dir, "summary_compression_imbalance.csv"), index=False)

    # Timing distributions
    times = out[["time_to_res_asia_min","time_to_res_ny_mid_min"]].copy()
    times.describe().to_csv(os.path.join(out_dir, "summary_resolution_times.csv"))

    # Simple binary cuts for quick read
    cuts = []
    for comp_cut in [True, False]:
        sub = out[(out["compression_flag"]==comp_cut)]
        cuts.append({
            "compression_flag": comp_cut,
            "n": int(len(sub)),
            "res_asia_rate": float(sub["resolved_to_asia_opposite"].mean()) if len(sub) else np.nan,
            "res_mid_rate": float(sub["resolved_to_ny_mid"].mean()) if len(sub) else np.nan,
            "t_asia_med": float(sub["time_to_res_asia_min"].median()) if len(sub) else np.nan,
            "t_mid_med": float(sub["time_to_res_ny_mid_min"].median()) if len(sub) else np.nan,
        })
    for imb_cut in [0.0, 0.15, 0.25, 0.35]:
        sub = out[out["imbalance_score"] >= imb_cut]
        cuts.append({
            "imbalance_ge": imb_cut,
            "n": int(len(sub)),
            "res_asia_rate": float(sub["resolved_to_asia_opposite"].mean()) if len(sub) else np.nan,
            "res_mid_rate": float(sub["resolved_to_ny_mid"].mean()) if len(sub) else np.nan,
            "t_asia_med": float(sub["time_to_res_asia_min"].median()) if len(sub) else np.nan,
            "t_mid_med": float(sub["time_to_res_ny_mid_min"].median()) if len(sub) else np.nan,
        })
    pd.DataFrame(cuts).to_csv(os.path.join(out_dir, "summary_cuts.csv"), index=False)

    print("Done. Wrote:")
    print(" -", out_events)
    print(" -", os.path.join(out_dir, "summary_compression_imbalance.csv"))
    print(" -", os.path.join(out_dir, "summary_resolution_times.csv"))
    print(" -", os.path.join(out_dir, "summary_cuts.csv"))

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imbalance", required=True)
    ap.add_argument("--ohlcv", required=True)
    ap.add_argument("--out", default="./invariant_tests")
    ap.add_argument("--pip_size", type=float, default=0.0001)
    ap.add_argument("--range_pct_window", type=int, default=20)
    ap.add_argument("--lookahead_min", type=int, default=180)
    ap.add_argument("--tol_pips", type=float, default=1.0)
    args = ap.parse_args()

    analyze(
        imb_path=args.imbalance,
        ohlcv_path=args.ohlcv,
        out_dir=args.out,
        pip_size=args.pip_size,
        range_pct_window=args.range_pct_window,
        lookahead_min=args.lookahead_min,
        tol_pips=args.tol_pips
    )
