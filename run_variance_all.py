#!/usr/bin/env python3
"""Run variance_phase_tracker.py over all CSVs in the `csv/` directory.

For each file matching the default pattern it will create a per-year output
folder under the given `--out-root` and run the tracker, saving console logs
and the CSV outputs there.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path

DEFAULT_CSV_DIR = "csv"
DEFAULT_PATTERN = r"DAT_MT_.*_(\d{4})\.csv"


def run_one(csv_path: Path, out_root: Path):
    m = re.search(DEFAULT_PATTERN, csv_path.name)
    year = m.group(1) if m else csv_path.stem
    out_dir = out_root / year
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "variance_phase_tracker.py",
        "--csv",
        str(csv_path),
        "--out",
        str(out_dir),
        "--pip_size",
        "0.0001",
        "--roll",
        "12",
        "--z_up",
        "0.75",
        "--z_down",
        "-0.75",
        "--week_end",
        "SUN",
    ]

    print(f"Running for {csv_path.name} -> {out_dir} ...")
    p = subprocess.run(cmd, capture_output=True, text=True)

    # Save logs
    (out_dir / "run_stdout.log").write_text(p.stdout)
    (out_dir / "run_stderr.log").write_text(p.stderr)

    success = (p.returncode == 0) and (out_dir.joinpath("asia_weekly_metrics.csv").exists())
    return {
        "year": year,
        "csv": str(csv_path),
        "out": str(out_dir),
        "returncode": p.returncode,
        "success": success,
        "stdout": p.stdout[:200],
        "stderr": p.stderr[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=DEFAULT_CSV_DIR, help="Directory containing CSV files")
    ap.add_argument("--out-root", default="results/variance_all", help="Root output directory")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="Regex to extract year from filename")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in csv_dir.glob("*.csv") if re.search(args.pattern, p.name)])
    if not files:
        print("No matching CSV files found in", csv_dir)
        return

    results = []
    for f in files:
        res = run_one(f, out_root)
        results.append(res)

    # Print summary
    print("\nSummary:")
    for r in results:
        status = "OK" if r["success"] else f"FAIL(code={r['returncode']})"
        print(f"{r['year']}: {status} -> {r['out']}")


if __name__ == "__main__":
    main()
