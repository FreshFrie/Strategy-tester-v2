# backtester.py
# SOJR-only runner (clean, minimal). Expects:
# - features.py with: add_core_features(), load_sessions()
# - configs/params.yaml and configs/sessions.yaml
# - CSV at data/<file>.csv (or pass a path arg)

import os, sys, csv, json, time
import numpy as np
import pandas as pd
import yaml

from features import add_core_features, load_sessions
from strategies.sojr import detect_sojr


# --------- helpers ---------
def load_params(path="configs/params.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path,"r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("params.yaml parsed to non-dict")
    return data

def load_sessions(path="configs/sessions.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path,"r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("sessions.yaml parsed to non-dict")
    # minimal schema check
    data.setdefault("fx", {})
    data["fx"].setdefault("london_open", [{"start":"03:00","end":"04:00"}])
    data["fx"].setdefault("ny_open",     [{"start":"08:00","end":"09:00"}])
    data["fx"].setdefault("overlap",     [{"start":"08:00","end":"11:00"}])
    data.setdefault("patch_marks", {"minutes":[7,13,27,41,53],"tolerance_min":2})
    return data

def load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    """
    Handles both:
      (A) headerless: date,time,open,high,low,close,volume
      (B) headered:   time,open,high,low,close,volume  (UTC-4, naive)
    Returns df with columns: time, open, high, low, close, volume
    """
    # peek first line to guess format
    with open(csv_path, "r") as f:
        first = f.readline().strip()

    # if first token looks like "YYYY.MM.DD,HH:MM,..." -> headerless
    if "." in first.split(",")[0] and ":" in first.split(",")[1]:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["date", "clock", "open", "high", "low", "close", "volume"],
        )
        df["time"] = pd.to_datetime(df["date"] + " " + df["clock"], format="%Y.%m.%d %H:%M")
        df = df.drop(columns=["date", "clock"])
    else:
        # assume already has 'time,open,high,low,close,volume'
        df = pd.read_csv(csv_path)
        if "time" not in df.columns:
            raise ValueError("CSV must contain a 'time' column or provide date+time columns headerless.")

        # parse time safely
        df["time"] = pd.to_datetime(df["time"])

    # ensure numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["time", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    return df


# --------- SOJR backtest ---------

def run_sojr(csv_path: str, asset: str = "EURUSD"):
    # 1) load data
    df = load_ohlcv_csv(csv_path)

    # 2) configs / params
    sessions_cfg = load_sessions("configs/sessions.yaml")
    all_params = load_params("configs/params.yaml")
    p = {**all_params["common"], **all_params["SOJR"]}

    # 3) features (ATR/EMAs + session/opening range tagging)
    df = add_core_features(df, p, sessions_cfg)

    # 4) iterate and trade
    os.makedirs("results", exist_ok=True)
    out_csv = "results/trades_sojr.csv"
    out_json = "results/summary_sojr.json"

    trades_pnl = []
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        # DR11 logging format
        w.writerow(["STRAT","ASSET","TIME_UTC-4","EVENT_TYPE","ANOMALY_%","ENTRY","EXIT","P/L","STATUS"])

        warmup = max(100, 2 * p["ATR_N"])
        total_iter = max(0, len(df) - 1 - warmup)
        print(f"[INFO] bars={len(df)} warmup={warmup} to_process={total_iter}")
        sys.stdout.flush()

        start_time = time.time()
        print_every = max(500, total_iter // 100)

        try:
            for i in range(warmup, len(df) - 1):
                # lightweight live progress
                if (i - warmup) % print_every == 0:
                    elapsed = time.time() - start_time
                    done = (i - warmup)
                    pct = 100.0 * done / total_iter if total_iter else 100.0
                    print(f"[PROGRESS] {done}/{total_iter} ({pct:.1f}%) elapsed={elapsed:.1f}s trades={len(trades_pnl)}")
                    sys.stdout.flush()

                row = df.iloc[i]
                sig = detect_sojr(row, p)
                if not sig:
                    continue

                entry = float(row["close"])
                side  = int(sig["side"])  # +1 long, -1 short
                atrN  = float(row["atr"])
                sl    = entry - side * p["sl_atr_mult"] * atrN

                # simple management: walk forward up to time_stop_min or until SL hits
                j = i + 1
                hold = 0
                status = "TIME_EXIT"
                exit_px = float(df["close"].iloc[j])
                while j < len(df) - 1 and hold < p["time_stop_min"]:
                    lo = float(df["low"].iloc[j])
                    hi = float(df["high"].iloc[j])
                    if side > 0 and lo <= sl:
                        exit_px = sl
                        status = "SL_HIT"
                        break
                    if side < 0 and hi >= sl:
                        exit_px = sl
                        status = "SL_HIT"
                        break
                    exit_px = float(df["close"].iloc[j])
                    j += 1
                    hold += 1

                pnl = (exit_px - entry) * side
                trades_pnl.append(pnl)

                # anomaly % (simple: prior-close change magnitude)
                prev_close = float(df["close"].iloc[i - 1])
                anomaly_pct = 100.0 * abs(entry - prev_close) / max(prev_close, 1e-12)

                w.writerow([
                    "SOJR",
                    asset,
                    str(df["time"].iloc[i]),
                    sig["event"],
                    round(anomaly_pct, 4),
                    round(entry, 6),
                    round(exit_px, 6),
                    round(pnl, 6),
                    status
                ])
                # ensure the row is written to disk for live visibility
                try:
                    f.flush()
                except Exception:
                    pass

        except KeyboardInterrupt:
            print("[INFO] Interrupted by user — finishing up and writing summary...")
            sys.stdout.flush()

    # 5) metrics summary
    wins   = [x for x in trades_pnl if x > 0]
    losses = [abs(x) for x in trades_pnl if x <= 0]
    total  = len(trades_pnl)
    win_rate = (len(wins) / total) if total else 0.0
    profit_factor = (sum(wins) / max(sum(losses), 1e-12)) if total else 0.0

    # crude Sharpe (minute bars → annualized approx for FX sessions)
    if total > 1:
        mean = float(np.mean(trades_pnl))
        std  = float(np.std(trades_pnl, ddof=0)) + 1e-12
        # 24h*60m ~ 1440/mins, but FX active hours smaller; keep simple scalar:
        sharpe = (mean / std) * np.sqrt(252 * 24 * 60)
    else:
        sharpe = 0.0

    summary = {
        "trades": total,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_approx": sharpe
    }
    with open(out_json, "w") as g:
        json.dump(summary, g, indent=2)

    return summary


# --------- entry point ---------

if __name__ == "__main__":
    # default path or CLI arg
    path = "data/DAT_MT_EURUSD_M1_2024.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    print(f"[RUN] SOJR on {path}")
    print(run_sojr(path, asset="EURUSD"))
