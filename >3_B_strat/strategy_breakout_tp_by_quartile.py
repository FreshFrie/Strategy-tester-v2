#!/usr/bin/env python3
"""
Breakout Strategy — Fixed RR with Quartile-Based TP + NY→Asia Containment Gate (FX-aware)
----------------------------------------------------------------------------------------
• Keeps RR fixed (default 1.5R).
• Sets TP distance (in pips) by Asia-range quartile (Q1..Q4).
  - tp_mode=auto   → compute median(max_extension) per quartile from events (filtered by day/time)
  - tp_mode=manual → use CLI-provided quartile TP distances in pips
• SL distance = TP_distance / RR (so RR stays constant).
• **NEW:** Pre-entry regime **gate** using NY→Asia imbalance features
  (contain_streak / resolve_streak / asia_contains_prev_ny / asia_resolves_prev_ny),
  merged from `ny_asia_daily.csv` produced by `ny_asia_imbalance_tracker.py`.
  - Only trade if `contain_streak >= --min_contain_streak` (default 3).
  - Optional `--avoid_resolution True` skips days with `resolve_streak >= k`.
  - Optional check that the **current day** is a containment day.
• Optional timing filter: restrict to early sweeps (01:00/01:30) and weekday filter.
• Logs MFE in R and per-quartile summary + debug sample after filters.

Outputs:
  out_dir/strategy_breakout_results.csv
  out_dir/strategy_breakout_summary.csv
  out_dir/strategy_breakout_by_quartile.csv
  out_dir/strategy_breakout_tp_table.csv (TP pips actually used per quartile)
  out_dir/events_after_filters_sample.csv (when --debug True)

Example:
  python strategy_breakout_tp_by_quartile.py \
    --events results/harvest_events.csv \
    --csv results/DAT_MT_EURUSD_M1_2024.csv \
    --out results/breakout_tp_ctx_filtered \
    --tp_mode auto --rr 1.5 --pip_size 0.0001 \
    --days TueWed --restrict_early True \
    --imbalance results/imbalance_test/ny_asia_daily.csv \
    --min_contain_streak 3 --avoid_resolution True --debug True
"""

import pandas as pd
import numpy as np
import os
import argparse

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

# ------------------------------ TP TABLE LOGIC -----------------------------
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

# --------------------------------- BACKTEST --------------------------------
def backtest(events_path: str, ohlcv_path: str, out_dir: str,
             rr: float = 1.5, pip_size: float = 0.0001, tp_mode: str = "auto",
             tp_q1: float | None = None, tp_q2: float | None = None, tp_q3: float | None = None, tp_q4: float | None = None,
             days: tuple[str,...] = ("Tuesday","Wednesday"),
             restrict_early: bool = True,
             skip_q4: bool = False,
             retest_tol: float = 0.0002,
             # NEW: regime gate args
             min_contain_streak: int = 3,
             avoid_resolution: bool = False,
             imbalance_path: str | None = None,
             debug: bool = False) -> None:
    # Force runs to write into a consistent parent folder but create a per-dataset subfolder
    # based on the OHLCV filename to avoid overwriting when running multiple CSVs.
    base_out_parent = os.path.join("results", "breakout_tp_run")
    csv_base = os.path.splitext(os.path.basename(ohlcv_path))[0]
    out_dir = os.path.join(base_out_parent, csv_base)
    os.makedirs(out_dir, exist_ok=True)

    # Load events and enrich
    ev = pd.read_csv(events_path)
    ev["sweep_time"]   = pd.to_datetime(ev["sweep_time"])
    ev["london_end"]   = pd.to_datetime(ev["london_end"])
    ev["trading_day_id"] = pd.to_datetime(ev["trading_day_id"]).dt.date
    ev["dow"] = ev["sweep_time"].dt.day_name()
    ev["hour"] = ev["sweep_time"].dt.hour
    ev, q1, q2, q3 = assign_quartiles(ev)

    # --- Merge imbalance features (containment / resolution) ---
    if not imbalance_path or not os.path.exists(imbalance_path):
        raise FileNotFoundError("--imbalance path is required and must point to ny_asia_daily.csv")
    imb = pd.read_csv(imbalance_path)
    imb["trading_day_id"] = pd.to_datetime(imb["trading_day_id"]).dt.date
    needed_cols = {"contain_streak","resolve_streak","asia_contains_prev_ny","asia_resolves_prev_ny"}
    miss = [c for c in needed_cols if c not in imb.columns]
    if miss:
        raise ValueError(f"imbalance CSV missing columns: {miss}. Did you pass ny_asia_daily.csv?")
    ev = pd.merge(
        ev,
        imb[["trading_day_id","contain_streak","resolve_streak","asia_contains_prev_ny","asia_resolves_prev_ny"]],
        on="trading_day_id",
        how="left"
    )

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

    # --------------------------- MAIN LOOP --------------------------- #
    for _, r in ev.iterrows():
        # Time / DOW filters
        if restrict_early and r["hour"] >= 2:
            continue
        if r["dow"] not in days:
            continue
        if skip_q4 and r["asia_quartile"] == "Q4":
            continue
        # Only trade Q1, Q3, Q4 — skip Q2 explicitly
        if r.get("asia_quartile") == "Q2":
            continue

        # --- Regime gate: require containment streak ---
        if pd.isna(r.get("contain_streak")) or int(r["contain_streak"]) < int(min_contain_streak):
            continue
        # Optional: avoid resolution streaks
        if avoid_resolution and not pd.isna(r.get("resolve_streak")) and int(r["resolve_streak"]) >= int(min_contain_streak):
            continue
        # Optional: require the *current day* to be containment (safer)
        if r.get("asia_contains_prev_ny") is not None and not bool(r["asia_contains_prev_ny"]):
            continue

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

            # Fixed-R framework
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
        })

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
                "contain_streak","resolve_streak","asia_contains_prev_ny","asia_resolves_prev_ny"]
        try:
            ev_pass = pd.merge(ev, trades_df["trading_day_id"].drop_duplicates(), on="trading_day_id", how="inner")
            ev_pass[cols].to_csv(dbg_path, index=False)
        except Exception:
            # if merge fails for any reason, just dump ev with cols
            ev[cols].to_csv(dbg_path, index=False)
        print(f"[debug] wrote merged events sample → {dbg_path}")

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
    ap.add_argument("--days", default="TueWed")
    ap.add_argument("--restrict_early", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)
    # removed: --skip_q4 and --retest_tol to simplify CLI (uses sensible defaults)
    # NEW: regime gate args
    ap.add_argument("--imbalance", type=str, required=True,
                    help="Path to ny_asia_daily.csv from NY→Asia tracker")
    ap.add_argument("--min_contain_streak", type=int, default=3,
                    help="Minimum Asia containment streak required to trade (default 3)")
    ap.add_argument("--avoid_resolution", type=lambda s: s.lower() in ["true","1","yes","y"], default=False,
                    help="If True, skip days with resolve_streak >= min_contain_streak")
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
    # using default quartile TP args (tp_mode auto) or function defaults
        days=days,
        restrict_early=args.restrict_early,
    # using default skip_q4 and retest_tol from function signature
        # regime gate
        min_contain_streak=args.min_contain_streak,
        avoid_resolution=args.avoid_resolution,
        imbalance_path=args.imbalance,
        debug=args.debug
    )
