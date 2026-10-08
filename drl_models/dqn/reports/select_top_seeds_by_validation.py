#!/usr/bin/env python3
"""Select top-5 seeds by validation reward from training logs.

Parses training log tqdm output: val=+X.XXXX
Returns best seeds per (asset, round, bp_level).
"""
import re, json
from pathlib import Path

LOG_DIR = Path("drl/dqn/logs")

def get_top_seeds(asset_path, rnd, bp_label="bp1", n=5):
    """Return list of (max_val_reward, seed) sorted best-first."""
    results = []
    for s in range(42, 52):
        log = LOG_DIR / f"train_{bp_label}_{asset_path}_r{rnd}_s{s}.log"
        if not log.exists():
            continue
        vals = re.findall(r'val=([+-]?[0-9.]+)', log.read_text())
        if vals:
            results.append((max(float(v) for v in vals), s))
    results.sort(reverse=True)
    return [(v, s) for v, s in results[:n]]

def select_all(bp_level=0.0001, n=5):
    """Return {asset_name: {round_num: [seed, ...]}} for given BP."""
    bp_label = f"bp{int(bp_level * 10000)}"
    assets = {
        "Commodity": "Commodity",
        "Equity_Index": "Equity Index",
        "Fixed_Income": "Fixed Income",
        "Forex": "Forex",
    }
    result = {}
    for slug, name in assets.items():
        result[name] = {}
        for rnd in [1, 2]:
            seeds = [s for _, s in get_top_seeds(slug, rnd, bp_label, n)]
            result[name][rnd] = seeds
    return result

if __name__ == "__main__":
    for bp_val in [0.0001, 0.0020]:
        sel = select_all(bp_val)
        print(f"\n=== bp{int(bp_val*10000)} validation top-5 ===")
        for asset, rounds in sel.items():
            print(f"  {asset}:")
            for rnd, seeds in sorted(rounds.items()):
                print(f"    r{rnd}: {seeds}")
