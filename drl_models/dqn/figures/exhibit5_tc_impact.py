#!/usr/bin/env python3
"""
Exhibit 5: Transaction Cost Impact — Sharpe Ratio & Daily Cost vs BP Level

Reads per-BP CSVs from figures/data/exhibit5_bp{XX}.csv.
Auto-discovers BP levels from exhibit5_manifest.json.
Output: exhibit5_tc_impact.pdf
"""
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "drl/dqn/figures/data"
OUTPUT = REPO_ROOT / "drl/dqn/figures/exhibit5_tc_impact.pdf"

ASSET_ORDER = ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]
COLORS = {"Commodity": "#1f77b4", "Equity Index": "#ff7f0e",
          "Fixed Income": "#2ca02c", "Forex": "#d62728", "All": "#9467bd"}
MARKERS = {"Commodity": "o", "Equity Index": "s",
           "Fixed Income": "^", "Forex": "D", "All": "P"}

def load_data():
    manifest = DATA_DIR / "exhibit5_manifest.json"
    if not manifest.exists():
        print("No manifest.")
        return {}
    bps = json.load(open(manifest)).get("bp_levels", [])
    bps.sort()
    data = {a: {} for a in ASSET_ORDER}
    for bp in bps:
        cf = DATA_DIR / f"exhibit5_bp{bp}.csv"
        if not cf.exists():
            continue
        with open(cf, newline="") as f:
            for row in csv.DictReader(f):
                a = row["Asset"].strip()
                entry = {}
                for k, v in row.items():
                    if k == "Asset": continue
                    try: entry[k] = float(v) if v else None
                    except ValueError: entry[k] = None
                data[a][bp] = entry
    return {a: d for a, d in data.items() if d}

def plot():
    data = load_data()
    if not data:
        print("No data.")
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8, 10), dpi=300)
    n_bp = max(len(d) for d in data.values())
    is_partial = n_bp < 5
    for asset in ASSET_ORDER:
        if asset not in data: continue
        bps = sorted(data[asset].keys())
        xs_s, ys_s, xs_c, ys_c = [], [], [], []
        for bp in bps:
            m = data[asset][bp]
            s, e = m.get("Sharpe"), m.get("E(R)")
            if s is not None: xs_s.append(bp); ys_s.append(s)
            if e is not None: xs_c.append(bp); ys_c.append(-e)
        if xs_s:
            ax_a.plot(xs_s, ys_s, color=COLORS[asset], marker=MARKERS[asset],
                      markersize=7, linewidth=2, label=asset)
        if xs_c:
            ax_b.plot(xs_c, ys_c, color=COLORS[asset], marker=MARKERS[asset],
                      markersize=7, linewidth=2, label=asset)
    for ax, title, ylabel in [
        (ax_a, "Panel A: Sharpe Ratio vs Transaction Cost", "Sharpe Ratio"),
        (ax_b, "Panel B: Avg Daily Cost Per Contract vs Transaction Cost", "Cost Proxy (\u2212E(R))")]:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("BP Level (bps)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
    if is_partial:
        for ax in (ax_a, ax_b):
            ax.text(0.5, 0.5, f"Partial \u2014 {n_bp} BP level(s)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray", alpha=0.3, rotation=30)
    fig.suptitle("Exhibit 5: Transaction Cost Impact on DQN Performance",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    print("Exhibit 5: Transaction Cost Impact")
    plot()
