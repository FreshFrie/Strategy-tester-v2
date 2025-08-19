#!/usr/bin/env python3
"""
Breakout Strategy — Fixed RR with Quartile-Based TP + NY→Asia Containment Gate (FX-aware)
----------------------------------------------------------------------------------------
• Keeps RR fixed (default 1.5R).
• Sets TP distance (in pips) by Asia-range quartile (Q1..Q4).
  - tp_mode=auto   → compute median(max_extension) per quartile from events (filtered by day/time)
  - tp_mode=manual → use CLI-provided quartile TP distances in pips
• SL distance = TP_distance / RR (so RR stays constant).
• **Regime gate:** Pre-entry regime gate using NY→Asia imbalance features
  (contain_streak / resolve_streak / asia_contains_prev_ny / asia_resolves_prev_ny),
  merged from `ny_asia_daily.csv` produced by `ny_asia_imbalance_tracker.py`.
  - Only trade if `contain_streak >= --min_contain_streak` (default 3).
  - Optional `--avoid_resolution True` skips days with `resolve_streak >= k`.
  - Optional check that the current day is a containment day.

• **NEW: North Star (Invariant) Gate** — compression + imbalance:
  - We compute rolling 20d percentiles for Asia and NY ranges.
  - compression_flag = (asia_pct <= --compression_pct) OR (ny_pct <= --compression_pct)
  - imbalance_score = max( (asia_high - ny_prev_high)/ny_range,
                           (ny_prev_low - asia_low)/ny_range )
  - Require imbalance_score >= --imbalance_min.
  - Toggle with --invariant_gate (default True).

• Optional timing filter: restrict to early sweeps (01:00/01:30) and weekday filter.
• Logs MFE in R and per-quartile summary + debug sample after filters.

Outputs:
  out_dir/strategy_breakout_results.csv
  out_dir/strategy_breakout_summary.csv
  out_dir/strategy_breakout_by_quartile.csv
  out_dir/strategy_breakout_tp_table.csv (TP pips actually used per quartile)
  out_dir/events_after_filters_sample.csv (when --debug True)
"""

import pandas as pd
import numpy as np
import os
import argparse
import json

# ------------------------------- IO HELPERS -------------------------------
def load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV with flexible headers. Returns DataFrame with 'timestamp' indexable column."""
    df = pd.read_csv(path)
    cmap = {c.lower().strip(): c for c in df.columns}
    req = ["date","time","open","high","low","close"]
    if not all(r in cmap for r in req):
        # Fallback to headerless common layout
        names = ["date","time","open","high","low","close","volume"]
        df = pd.read_csv(path, header=None, names=names)
        cmap = {c.lower().strip(): c for c in df.columns}
        missing = [r for r in req if r not in cmap]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in OHLCV CSV.")
    df["timestamp"] = pd.to_datetime(df[cmap["date"]].astype(str) + " " + df[cmap["time"]].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)
    for k in ["open","high","low","close","volume"]:
        if k in cmap:
            df[k] = pd.to_numeric(df[cmap[k]], errors="coerce")
    return df

# --------------------------- FEATURE ENGINEERING ---------------------------
def assign_quartiles(ev: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float]:
    """Attach asia_range and asia_quartile (Q1..Q4) using global distribution."""
    ev = ev.copy()
    ev["asia_range"] = ev["asia_high"] - ev["asia_low"]
    q1 = ev["asia_range"].quantile(0.25)
    q2 = ev["asia_range"].quantile(0.50)
    q3 = ev["asia_range"].quantile(0.75)
    def qlabel(x: float) -> str:
        if x <= q1:   return "Q1"
        if x <= q2:   return "Q2"
        if x <= q3:   return "Q3"
        return "Q4"
    ev["asia_quartile"] = ev["asia_range"].apply(qlabel)
    return ev, q1, q2, q3

def parse_days(arg: str) -> tuple[str, ...]:
    if arg == "TueWed":
        return ("Tuesday","Wednesday")
    if arg == "MonThu":
        return ("Monday","Thursday")
    if arg == "All":
        return ("Monday","Tuesday","Wednesday","Thursday","Friday")
    # comma list
    return tuple(p.strip().capitalize() for p in arg.split(","))

def compute_tp_table_auto(ev: pd.DataFrame, days: tuple[str,...], restrict_early: bool, pip_size: float) -> tuple[dict, pd.DataFrame]:
    """Compute median(max_extension) per quartile (in pips), filtered by requested DOW/time."""
    sub = ev.copy()
    if restrict_early:
        sub = sub[sub["hour"] < 2]
    if days:
        sub = sub[sub["dow"].isin(days)]
    # Ensure quartiles present on the filtered subset too
    sub, _, _, _ = assign_quartiles(sub)
    # Convert extension to pips
    sub["ext_pips"] = (sub["max_extension"] / pip_size).astype(float)
    tp_table = sub.groupby("asia_quartile")["ext_pips"].median().to_dict()
    # Backfill if any quartile absent
    fallback = float(sub["ext_pips"].median() if len(sub) else 10.0)
    for q in ("Q1","Q2","Q3","Q4"):
        tp_table.setdefault(q, fallback)
    return tp_table, sub

# ------------------------------ ROLLING PERCENTILE -------------------------
def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank (0..100) for each point based on the *previous* 'window' values.
    First 'window' rows → NaN.
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

# ------------------------------ INVARIANT GATE -----------------------------
def passes_invariant_gate(row: pd.Series,
                          pip_size: float = 0.0001,
                          imbalance_min: float = 0.15,
                          compression_pct: float = 20.0) -> bool:
    """
    North Star Gate:
    - compression_flag: asia_range_pct_20d <= compression_pct OR ny_range_pct_20d <= compression_pct
    - imbalance_score >= imbalance_min
    imbalance_score = max( (asia_high - ny_prev_high)/ny_range, (ny_prev_low - asia_low)/ny_range )
    """
    try:
        asia_pct = float(row.get("asia_range_pct_20d", np.nan))
        ny_pct   = float(row.get("ny_range_pct_20d", np.nan))
        compression_flag = (not np.isnan(asia_pct) and asia_pct <= compression_pct) or \
                           (not np.isnan(ny_pct) and ny_pct <= compression_pct)

        ny_prev_high = float(row["ny_prev_high"])
        ny_prev_low  = float(row["ny_prev_low"])
        asia_high    = float(row["asia_high"])
        asia_low     = float(row["asia_low"])

        ny_range = ny_prev_high - ny_prev_low
        if ny_range <= 0 or np.isnan(ny_range):
            return False

        up_disp = max(0.0, asia_high - ny_prev_high) / ny_range
        dn_disp = max(0.0, ny_prev_low - asia_low) / ny_range
        imbalance_score = max(up_disp, dn_disp)

        # Attach for downstream logging (not strictly required to return True/False)
        row["compression_flag"] = compression_flag
        row["imbalance_score"] = imbalance_score

        if not compression_flag:
            return False
        if imbalance_score < float(imbalance_min):
            return False
        return True
    except Exception:
        return False

# --------------------------------- BACKTEST --------------------------------
def backtest(events_path: str, ohlcv_path: str, out_dir: str,
             rr: float = 1.5, pip_size: float = 0.0001, tp_mode: str = "auto",
             tp_q1: float | None = None, tp_q2: float | None = None, tp_q3: float | None = None, tp_q4: float | None = None,
             days: tuple[str,...] = ("Tuesday","Wednesday"),
             restrict_early: bool = True,
             skip_q4: bool = False,
             retest_tol: float = 0.0002,
             # Regime gate args
             min_contain_streak: int = 3,
             avoid_resolution: bool = False,
             imbalance_path: str | None = None,
             require_containment_today: bool = False,
             # NEW: Invariant gate args
             invariant_gate: bool = True,
             imbalance_min: float = 0.15,
             compression_pct: float = 20.0,
             range_pct_window: int = 20,
             debug: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Load events and enrich
    ev = pd.read_csv(events_path)
    ev["sweep_time"]   = pd.to_datetime(ev["sweep_time"])
    ev["london_end"]   = pd.to_datetime(ev["london_end"])
    ev["trading_day_id"] = pd.to_datetime(ev["trading_day_id"]).dt.date
    ev["dow"] = pd.to_datetime(ev["trading_day_id"]).dt.day_name()
    ev["hour"] = ev["sweep_time"].dt.hour
    # Note: quartiles require asia_high/asia_low which are provided by the imbalance CSV.
    # We'll assign quartiles after merging imbalance onto the event rows below.

    # --- Merge imbalance features (containment / resolution) ---
    if not imbalance_path or not os.path.exists(imbalance_path):
        raise FileNotFoundError("--imbalance path is required and must point to ny_asia_daily.csv")
    imb = pd.read_csv(imbalance_path)
    imb["trading_day_id"] = pd.to_datetime(imb["trading_day_id"]).dt.date
    needed_cols = {"contain_streak","resolve_streak","asia_contains_prev_ny","asia_resolves_prev_ny",
                   "ny_prev_high","ny_prev_low","asia_high","asia_low","asia_range_pips"}
    miss = [c for c in needed_cols if c not in imb.columns]
    if miss:
        raise ValueError(f"imbalance CSV missing columns: {miss}. Did you pass ny_asia_daily.csv?")

    # INVARIANT: compute rolling percentiles on imbalance daily rows BEFORE merge
    imb = imb.sort_values("trading_day_id").reset_index(drop=True)
    # Asia percentile already via asia_range_pips
    imb["asia_range_pct_20d"] = rolling_percentile(imb["asia_range_pips"].astype(float), window=range_pct_window)
    # NY range percentile (convert to pips using pip_size to stay comparable)
    ny_range = (imb["ny_prev_high"].astype(float) - imb["ny_prev_low"].astype(float)).clip(lower=0)
    ny_range_pips = (ny_range / (pip_size if pip_size > 0 else 1.0)).replace([np.inf, -np.inf], np.nan)
    imb["ny_range_pct_20d"] = rolling_percentile(ny_range_pips, window=range_pct_window)

    ev = pd.merge(
        ev,
        imb[["trading_day_id","contain_streak","resolve_streak","asia_contains_prev_ny","asia_resolves_prev_ny",
             "ny_prev_high","ny_prev_low","asia_high","asia_low","asia_range_pips",
             "asia_range_pct_20d","ny_range_pct_20d"]],
        on="trading_day_id",
        how="left"
    )

    # Normalize possible duplicate columns created by the merge (pandas appends _x/_y).
    # Preference order: imbalance (_y) -> events (_x) -> existing base column.
    for base in ("asia_high","asia_low","asia_range_pips","ny_prev_high","ny_prev_low"):
        col_y = f"{base}_y"
        col_x = f"{base}_x"
        if col_y in ev.columns:
            ev[base] = ev[col_y]
        elif col_x in ev.columns:
            ev[base] = ev[col_x]
        # else leave as-is if base already present

    # Drop helper duplicate columns that end with _x or _y to keep a clean frame
    drop_cols = [c for c in ev.columns if c.endswith("_x") or c.endswith("_y")]
    if drop_cols:
        ev = ev.drop(columns=drop_cols)

    # Compute invariant fields vectorized: compression_flag and imbalance_score
    # compression_flag: asia_range_pct_20d <= compression_pct OR ny_range_pct_20d <= compression_pct
    ev["asia_range_pct_20d"] = ev.get("asia_range_pct_20d")
    ev["ny_range_pct_20d"] = ev.get("ny_range_pct_20d")
    # Compute compression_flag (may contain NaNs)
    ev["compression_flag"] = ((ev["asia_range_pct_20d"] <= compression_pct) | (ev["ny_range_pct_20d"] <= compression_pct)).fillna(False)

    # imbalance_score: max( (asia_high - ny_prev_high)/ny_range, (ny_prev_low - asia_low)/ny_range )
    # safely compute elementwise
    for c in ["ny_prev_high","ny_prev_low","asia_high","asia_low"]:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ny_high = ev.get("ny_prev_high")
    ny_low  = ev.get("ny_prev_low")
    asia_h  = ev.get("asia_high")
    asia_l  = ev.get("asia_low")
    ny_range = (ny_high - ny_low)
    up_disp = (asia_h - ny_high) / ny_range
    dn_disp = (ny_low - asia_l) / ny_range
    # where ny_range <= 0 or NaN, set imbalance_score = NaN
    imbalance_score = np.where((ny_range > 0) & (~pd.isna(ny_range)), np.maximum(np.nan_to_num(up_disp, nan=0.0), np.nan_to_num(dn_disp, nan=0.0)), np.nan)
    ev["imbalance_score"] = imbalance_score

    # Now that asia_high/asia_low are present on event rows, compute asia_range quartiles
    ev, q1, q2, q3 = assign_quartiles(ev)

    # Build TP table (pips)
    if tp_mode.lower() == "manual":
        missing = [k for k,v in dict(Q1=tp_q1,Q2=tp_q2,Q3=tp_q3,Q4=tp_q4).items() if v is None]
        if missing:
            raise SystemExit(f"--tp_mode manual requires all quartiles: missing {missing}")
        tp_table = {"Q1": float(tp_q1), "Q2": float(tp_q2), "Q3": float(tp_q3), "Q4": float(tp_q4)}
    else:
        tp_table, _ = compute_tp_table_auto(ev, days, restrict_early, pip_size)

    # Persist the TP table actually used
    pd.DataFrame([{"quartile":q, "tp_pips":tp_table[q]} for q in ("Q1","Q2","Q3","Q4")]) \
        .to_csv(os.path.join(out_dir, "strategy_breakout_tp_table.csv"), index=False)

    # Load prices
    px = load_ohlcv(ohlcv_path).copy().set_index("timestamp")

    # Optionally prepare a debug sample dump BEFORE filters to compare
    if debug:
        ev.to_csv(os.path.join(out_dir, "_debug_merged_events_raw.csv"), index=False)

    trades = []

    # Diagnostic funnel to count how many events pass each gate
    funnel = dict(total=len(ev), after_invariant=0, after_time=0, after_regime=0, taken=0)

    # --------------------------- MAIN LOOP --------------------------- #
    for _, r in ev.iterrows():
        # INVARIANT GATE (optional)
        pass_invariant = True if not invariant_gate else passes_invariant_gate(r,
                                                                             pip_size=pip_size,
                                                                             imbalance_min=imbalance_min,
                                                                             compression_pct=compression_pct)
        if not pass_invariant:
            continue
        funnel["after_invariant"] += 1

        # Time / DOW filters
        if restrict_early and r["hour"] >= 2:
            continue
        if r["dow"] not in days:
            continue
        if skip_q4 and r["asia_quartile"] == "Q4":
            continue
        funnel["after_time"] += 1

        # --- Legacy Regime gate: containment streak + optional resolution avoidance ---
        # NOTE: only enforce containment-streak lower bound when the value is present
        if not pd.isna(r.get("contain_streak")) and int(r["contain_streak"]) < int(min_contain_streak):
            continue
        if avoid_resolution and not pd.isna(r.get("resolve_streak")) and int(r["resolve_streak"]) >= int(min_contain_streak):
            continue
        # require_containment_today is now opt-in; don't force containment unless requested
        if require_containment_today:
            if r.get("asia_contains_prev_ny") is not None and not bool(r["asia_contains_prev_ny"]):
                continue
        funnel["after_regime"] += 1

        sweep_time = r["sweep_time"]
        london_end = r["london_end"]
        side = r["side"]
        asia_high = float(r["asia_high"])
        asia_low  = float(r["asia_low"])
        asia_quartile = r["asia_quartile"]

        seg = px.loc[(px.index >= sweep_time) & (px.index <= london_end)].copy()
        if seg.empty:
            continue

        # Entry: prefer retest of the level within tolerance; else first close beyond level
        if side == "up_sweep":
            retest = seg[(seg["low"] <= asia_high + retest_tol) & (seg["close"] > asia_high)]
            if not retest.empty:
                entry_time = retest.index[0]
                entry_price = float(retest.iloc[0]["close"])
            else:
                above = seg[seg["close"] > asia_high]
                if above.empty:
                    continue
                entry_time = above.index[0]
                entry_price = float(above.iloc[0]["close"])

            # TP distance from quartile (pips → price units); SL from fixed-R rule
            tp_pips = float(tp_table.get(asia_quartile, tp_table["Q2"]))
            tp_dist = tp_pips * pip_size
            sl_dist = tp_dist / rr

            take_profit = entry_price + tp_dist
            stop_price  = entry_price - sl_dist

            fwd = seg.loc[entry_time:]
            hit_tp_time = hit_sl_time = None
            for ts, row in fwd.iterrows():
                if hit_tp_time is None and row["high"] >= take_profit: hit_tp_time = ts
                if hit_sl_time is None and row["low"]  <= stop_price:  hit_sl_time = ts
                if hit_tp_time or hit_sl_time: break

            exit_reason, exit_time, exit_price = "time", london_end, float(fwd.iloc[-1]["close"])
            if hit_tp_time and hit_sl_time:
                if min(hit_tp_time, hit_sl_time) == hit_sl_time:
                    exit_reason, exit_time, exit_price = "SL", hit_sl_time, stop_price
                else:
                    exit_reason, exit_time, exit_price = "TP", hit_tp_time, take_profit
            elif hit_tp_time:
                exit_reason, exit_time, exit_price = "TP", hit_tp_time, take_profit
            elif hit_sl_time:
                exit_reason, exit_time, exit_price = "SL", hit_sl_time, stop_price

            risk_per_unit = sl_dist
            r_multiple = (tp_dist / sl_dist) if exit_reason == "TP" else \
                         (-1.0 if exit_reason == "SL" else (exit_price - entry_price) / sl_dist)
            mfe_R = max((fwd["high"].max() - entry_price), 0.0) / sl_dist

        else:  # down_sweep
            retest = seg[(seg["high"] >= asia_low - retest_tol) & (seg["close"] < asia_low)]
            if not retest.empty:
                entry_time = retest.index[0]
                entry_price = float(retest.iloc[0]["close"])
            else:
                below = seg[seg["close"] < asia_low]
                if below.empty:
                    continue
                entry_time = below.index[0]
                entry_price = float(below.iloc[0]["close"])

            tp_pips = float(tp_table.get(asia_quartile, tp_table["Q2"]))
            tp_dist = tp_pips * pip_size
            sl_dist = tp_dist / rr

            take_profit = entry_price - tp_dist
            stop_price  = entry_price + sl_dist

            fwd = seg.loc[entry_time:]
            hit_tp_time = hit_sl_time = None
            for ts, row in fwd.iterrows():
                if hit_tp_time is None and row["low"]  <= take_profit: hit_tp_time = ts
                if hit_sl_time is None and row["high"] >= stop_price:  hit_sl_time = ts
                if hit_tp_time or hit_sl_time: break

            exit_reason, exit_time, exit_price = "time", london_end, float(fwd.iloc[-1]["close"])
            if hit_tp_time and hit_sl_time:
                if min(hit_tp_time, hit_sl_time) == hit_sl_time:
                    exit_reason, exit_time, exit_price = "SL", hit_sl_time, stop_price
                else:
                    exit_reason, exit_time, exit_price = "TP", hit_tp_time, take_profit
            elif hit_tp_time:
                exit_reason, exit_time, exit_price = "TP", hit_tp_time, take_profit
            elif hit_sl_time:
                exit_reason, exit_time, exit_price = "SL", hit_sl_time, stop_price

            risk_per_unit = sl_dist
            r_multiple = (tp_dist / sl_dist) if exit_reason == "TP" else \
                         (-1.0 if exit_reason == "SL" else (entry_price - exit_price) / sl_dist)
            mfe_R = max((entry_price - fwd["low"].min()), 0.0) / sl_dist

    trades.append({
            "trading_day_id": r["trading_day_id"],
            "dow": r["dow"],
            "side": side,
            "asia_quartile": asia_quartile,
            "tp_mode": tp_mode,
            "tp_pips": tp_pips,
            "rr_fixed": rr,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "take_profit": take_profit,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "r_multiple": r_multiple,
            "mfe_R": mfe_R,
            # regime fields for audit
            "contain_streak": r.get("contain_streak"),
            "resolve_streak": r.get("resolve_streak"),
            "asia_contains_prev_ny": r.get("asia_contains_prev_ny"),
            "asia_resolves_prev_ny": r.get("asia_resolves_prev_ny"),
            # invariant audit
            "asia_range_pct_20d": r.get("asia_range_pct_20d"),
            "ny_range_pct_20d": r.get("ny_range_pct_20d"),
            "compression_flag": r.get("compression_flag"),
            "imbalance_score": r.get("imbalance_score"),
        })
    funnel["taken"] += 1

    trades_df = pd.DataFrame(trades)
    trades_path = os.path.join(out_dir, "strategy_breakout_results.csv")
    trades_df.to_csv(trades_path, index=False)

    if not trades_df.empty:
        summary = pd.DataFrame([{
            "trades": len(trades_df),
            "win_rate": (trades_df["r_multiple"] > 0).mean(),
            "avg_R": trades_df["r_multiple"].mean(),
            "expectancy_R": trades_df["r_multiple"].mean(),
            "tp_rate": (trades_df["exit_reason"] == "TP").mean(),
            "sl_rate": (trades_df["exit_reason"] == "SL").mean()
        }])
        by_q = trades_df.groupby("asia_quartile")["r_multiple"].agg(["count","mean"]).reset_index()
        by_q.rename(columns={"count":"trades_q","mean":"avg_R_q"}, inplace=True)
    else:
        summary = pd.DataFrame([{ "trades": 0, "win_rate": np.nan, "avg_R": np.nan,
                                  "expectancy_R": np.nan, "tp_rate": np.nan, "sl_rate": np.nan }])
        by_q = pd.DataFrame(columns=["asia_quartile","trades_q","avg_R_q"])

    summary.to_csv(os.path.join(out_dir, "strategy_breakout_summary.csv"), index=False)
    by_q.to_csv(os.path.join(out_dir, "strategy_breakout_by_quartile.csv"), index=False)

    # Optional debug dump AFTER filters (events that passed gate)
    if debug:
        dbg_path = os.path.join(out_dir, "events_after_filters_sample.csv")
        cols = ["trading_day_id","dow","hour","side","asia_quartile",
                "contain_streak","resolve_streak","asia_contains_prev_ny","asia_resolves_prev_ny",
                "asia_range_pct_20d","ny_range_pct_20d","compression_flag","imbalance_score"]
        # Ensure the debug columns exist on the event frame
        for c in cols:
            if c not in ev.columns:
                ev[c] = np.nan
        # If there are trades, filter events to traded days; otherwise write full merged events
        if not trades_df.empty:
            traded_days = trades_df["trading_day_id"].drop_duplicates()
            ev_pass = ev[ev["trading_day_id"].isin(traded_days)]
            ev_pass[cols].to_csv(dbg_path, index=False)
        else:
            ev[cols].to_csv(dbg_path, index=False)
        print(f"[debug] wrote merged events sample → {dbg_path}")

    # Write funnel diagnostics
    try:
        with open(os.path.join(out_dir, "_debug_filter_funnel.json"), "w") as fh:
            fh.write(json.dumps(funnel, indent=2, default=str))
    except Exception:
        pass

    print(f"Done. Wrote:\n  - {trades_path}\n  - {os.path.join(out_dir, 'strategy_breakout_summary.csv')}\n  - {os.path.join(out_dir, 'strategy_breakout_by_quartile.csv')}\n  - {os.path.join(out_dir, 'strategy_breakout_tp_table.csv')}")

# ----------------------------------- CLI -----------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="./breakout_tp_ctx")
    ap.add_argument("--rr", type=float, default=1.5, help="Fixed RR (TP/SL ratio)")
    ap.add_argument("--pip_size", type=float, default=0.0001, help="Pip size (EURUSD=0.0001)")
    ap.add_argument("--tp_mode", choices=["auto","manual"], default="auto")
    ap.add_argument("--tp_q1", type=float, default=None, help="Manual TP in pips for Q1")
    ap.add_argument("--tp_q2", type=float, default=None, help="Manual TP in pips for Q2")
    ap.add_argument("--tp_q3", type=float, default=None, help="Manual TP in pips for Q3")
    ap.add_argument("--tp_q4", type=float, default=None, help="Manual TP in pips for Q4")
    ap.add_argument("--days", default="TueWed")
    ap.add_argument("--restrict_early", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)
    ap.add_argument("--skip_q4", type=lambda s: s.lower() in ["true","1","yes","y"], default=False)
    ap.add_argument("--retest_tol", type=float, default=0.0002)
    # Regime gate args
    ap.add_argument("--imbalance", type=str, required=True,
                    help="Path to ny_asia_daily.csv from NY→Asia tracker")
    ap.add_argument("--min_contain_streak", type=int, default=3,
                    help="Minimum Asia containment streak required to trade (default 3)")
    ap.add_argument("--avoid_resolution", type=lambda s: s.lower() in ["true","1","yes","y"], default=False,
                    help="If True, skip days with resolve_streak >= min_contain_streak")
    # NEW: Invariant gate args
    ap.add_argument("--invariant_gate", type=lambda s: s.lower() in ["true","1","yes","y"], default=True,
                    help="If True, require compression+imbalance invariant to pass")
    ap.add_argument("--imbalance_min", type=float, default=0.15,
                    help="Minimum imbalance_score to pass invariant gate (e.g., 0.15)")
    ap.add_argument("--compression_pct", type=float, default=20.0,
                    help="Percentile threshold for compression flag (e.g., 20 → bottom 20%%)")
    ap.add_argument("--range_pct_window", type=int, default=20,
                    help="Rolling window (days) for percentile ranks")
    ap.add_argument("--debug", type=lambda s: s.lower() in ["true","1","yes","y"], default=False,
                    help="If True, write a merged sample for inspection")
    args = ap.parse_args()

    # Parse days
    days = parse_days(args.days)

    backtest(
        events_path=args.events,
        ohlcv_path=args.csv,
        out_dir=args.out,
        rr=args.rr,
        pip_size=args.pip_size,
        tp_mode=args.tp_mode,
        tp_q1=args.tp_q1, tp_q2=args.tp_q2, tp_q3=args.tp_q3, tp_q4=args.tp_q4,
        days=days,
        restrict_early=args.restrict_early,
        skip_q4=args.skip_q4,
        retest_tol=args.retest_tol,
        # regime gate
        min_contain_streak=args.min_contain_streak,
        avoid_resolution=args.avoid_resolution,
        imbalance_path=args.imbalance,
        # invariant gate
        invariant_gate=args.invariant_gate,
        imbalance_min=args.imbalance_min,
        compression_pct=args.compression_pct,
        range_pct_window=args.range_pct_window,
        debug=args.debug
    )
