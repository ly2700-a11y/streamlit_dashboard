#!/usr/bin/env python3
"""Export per-BP-level DQN metrics to CSVs for Exhibit 5.

Reads ensemble_table2_bp/{asset_slug}/bp{XX}/metrics.json and
ensemble_table2_bp/bp{XX}/table2_metrics.json (for "All").

Writes one CSV per BP level: figures/data/exhibit5_bp{XX}.csv
Each CSV: Asset, E(R), std(R), Sharpe, Sortino, MDD, % +ve, Ave P/L
"""
import csv, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "drl/dqn/reports/ensemble_table2_bp"
DATA_DIR = REPO_ROOT / "drl/dqn/figures/data"
ASSETS = [
    ("Commodity", "Commodity"),
    ("Equity Index", "Equity_Index"),
    ("Fixed Income", "Fixed_Income"),
    ("Forex", "Forex"),
]
HEADERS = ["Asset", "E(R)", "std(R)", "Sharpe", "Sortino", "MDD", "% +ve", "Ave P/L"]

def _bp_from_dir(name):
    for p in name.split("_"):
        if p.startswith("bp") and p[2:].isdigit():
            return int(p[2:])
    return None

def read_metrics(filepath, dirname=""):
    with open(filepath) as f:
        d = json.load(f)
    if "metrics" in d:
        m, bp = d["metrics"], d.get("bp")
    else:
        m = {k: v for k, v in d.items() if k not in ("n_contracts", "bp")}
        bp = d.get("bp")
    bp_bps = int(bp * 10000) if bp is not None else _bp_from_dir(dirname)
    return m, bp_bps

def collect_bps():
    bps = set()
    for _, slug in ASSETS:
        adir = REPORTS_ROOT / slug
        if not adir.exists(): continue
        for bpd in adir.iterdir():
            if bpd.is_dir():
                bp_bps = _bp_from_dir(bpd.name)
                if bp_bps: bps.add(bp_bps)
    # Also check for standalone bp dirs with table2_metrics.json
    for bpd in REPORTS_ROOT.iterdir():
        if bpd.is_dir():
            bp_bps = _bp_from_dir(bpd.name)
            if bp_bps and (bpd / "table2_metrics.json").exists():
                bps.add(bp_bps)
    return sorted(bps)

def export():
    bps = collect_bps()
    if not bps:
        print("No BP data.")
        return
    for bp_bps in bps:
        rows = []
        # Per-asset
        for name, slug in ASSETS:
            mf = REPORTS_ROOT / slug / f"bp{bp_bps}" / "metrics.json"
            if not mf.exists():
                print(f"  bp{bp_bps} {name}: missing"); continue
            try:
                m, _ = read_metrics(mf, f"bp{bp_bps}")
            except Exception as e:
                print(f"  bp{bp_bps} {name}: ERROR {e}"); continue
            row = {"Asset": name}
            for h in HEADERS[1:]:
                row[h] = m.get(h)
            rows.append(row)
        # All from per-BP table2_metrics.json
        tf = REPORTS_ROOT / f"bp{bp_bps}" / "table2_metrics.json"
        if tf.exists():
            d = json.load(open(tf))
            if "All" in d:
                all_m = d["All"].get("metrics", {})
                row = {"Asset": "All"}
                for h in HEADERS[1:]:
                    row[h] = all_m.get(h)
                rows.append(row)
        if rows:
            csv_path = DATA_DIR / f"exhibit5_bp{bp_bps}.csv"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=HEADERS)
                w.writeheader(); w.writerows(rows)
            assets = [r["Asset"] for r in rows]
            print(f"  bp{bp_bps}: {assets} -> exhibit5_bp{bp_bps}.csv")
    # Manifest
    with open(DATA_DIR / "exhibit5_manifest.json", "w") as f:
        json.dump({"bp_levels": bps}, f, indent=2)
    print(f"  Manifest: bp_levels={bps}")

if __name__ == "__main__":
    print("=" * 50)
    print("Export BP metrics -> per-BP CSVs")
    print("=" * 50)
    export()
    print("\nDone!")
