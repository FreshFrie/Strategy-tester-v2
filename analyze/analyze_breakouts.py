#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
import json

def bucket_minutes(dt, bucket=30):
    h = dt.hour
    m = dt.minute - (dt.minute % bucket)
    return f"{h:02d}:{m:02d}"

def load_events(path):
    ev = pd.read_csv(path)
    ev["sweep_time"] = pd.to_datetime(ev["sweep_time"])
    ev["reverted"] = ev["reverted_inside_by_end"].astype(int)
    ev["breakout"] = 1 - ev["reverted"]
    ev["dow"] = ev["sweep_time"].dt.day_name()
    ev["time_bucket_30"] = ev["sweep_time"].apply(lambda x: bucket_minutes(x, 30))
    ev["time_bucket_15"] = ev["sweep_time"].apply(lambda x: bucket_minutes(x, 15))
    return ev

def analyze(ev, out_dir, bucket=30):
    # choose bucket
    tb_col = f"time_bucket_{bucket}"
    if tb_col not in ev.columns:
        ev[tb_col] = ev["sweep_time"].apply(lambda x: f"{x.hour:02d}:{x.minute - (x.minute % bucket):02d}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) By time bucket (overall)
    by_time = ev.groupby(tb_col).agg(
        sweeps=("breakout","count"),
        breakout_rate=("breakout","mean"),
        fakeout_rate=("reverted","mean")
    ).reset_index().sort_values(tb_col)
    by_time_path = os.path.join(out_dir, "breakout_by_timebucket.csv")
    by_time.to_csv(by_time_path, index=False)

    # 2) By DOW (overall)
    by_dow = ev.groupby("dow").agg(
        sweeps=("breakout","count"),
        breakout_rate=("breakout","mean"),
        fakeout_rate=("reverted","mean")
    ).reset_index().sort_values("breakout_rate", ascending=False)
    by_dow_path = os.path.join(out_dir, "breakout_by_dow.csv")
    by_dow.to_csv(by_dow_path, index=False)

    # 3) Time x DOW pivot
    time_dow = ev.pivot_table(index=tb_col, columns="dow", values="breakout", aggfunc=["mean","count"])
    time_dow.columns = [f"{a}_{b}" for a,b in time_dow.columns]
    time_dow = time_dow.reset_index().sort_values(tb_col)
    time_dow_path = os.path.join(out_dir, "breakout_by_time_and_dow.csv")
    time_dow.to_csv(time_dow_path, index=False)

    # 4) Time x Side pivot
    time_side = ev.pivot_table(index=tb_col, columns="side", values="breakout", aggfunc=["mean","count"])
    time_side.columns = [f"{a}_{b}" for a,b in time_side.columns]
    time_side = time_side.reset_index().sort_values(tb_col)
    time_side_path = os.path.join(out_dir, "breakout_by_time_and_side.csv")
    time_side.to_csv(time_side_path, index=False)

    # 5) Rank top windows (breakout-leaning and fakeout-leaning) for Tue/Wed vs Thu/Fri
    def rank_windows(sub, label):
        grp = sub.groupby(tb_col).agg(
            sweeps=("breakout","count"),
            breakout_rate=("breakout","mean")
        ).reset_index()
        grp = grp[grp["sweeps"] >= 5]  # require at least 5 sweeps for stability
        top_breakout = grp.sort_values("breakout_rate", ascending=False).head(5).to_dict(orient="records")
        top_fakeout = grp.sort_values("breakout_rate", ascending=True).head(5).to_dict(orient="records")
        return {label: {"top_breakout": top_breakout, "top_fakeout": top_fakeout}}

    ranks = {}
    ranks.update(rank_windows(ev[ev["dow"].isin(["Tuesday","Wednesday"])], "TueWed"))
    ranks.update(rank_windows(ev[ev["dow"].isin(["Thursday","Friday"])], "ThuFri"))
    ranks_path = os.path.join(out_dir, "top_windows.json")
    with open(ranks_path, "w") as f:
        json.dump(ranks, f, indent=2)

    # 6) Excel pack
    xlsx_path = os.path.join(out_dir, "breakout_analysis.xlsx")
    with pd.ExcelWriter(xlsx_path) as xw:
        ev.to_excel(xw, sheet_name="events_raw", index=False)
        by_time.to_excel(xw, sheet_name="by_time", index=False)
        by_dow.to_excel(xw, sheet_name="by_dow", index=False)
        time_dow.to_excel(xw, sheet_name="time_x_dow", index=False)
        time_side.to_excel(xw, sheet_name="time_x_side", index=False)

    print("Wrote:")
    print(" -", by_time_path)
    print(" -", by_dow_path)
    print(" -", time_dow_path)
    print(" -", time_side_path)
    print(" -", ranks_path)
    print(" -", xlsx_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="Path to harvest_events.csv")
    ap.add_argument("--out", default="./results/analysis", help="Output directory")
    ap.add_argument("--bucket", type=int, default=30, choices=[15,30], help="Time bucket minutes (15 or 30)")
    args = ap.parse_args()

    ev = load_events(args.events)
    analyze(ev, args.out, bucket=args.bucket)

if __name__ == "__main__":
    main()
