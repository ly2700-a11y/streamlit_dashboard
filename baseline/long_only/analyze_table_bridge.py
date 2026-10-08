#!/usr/bin/env python3
"""
analyze_table_bridge.py

Diagnose how Table 2 relates to Table 3 in both:
1. the paper's published targets, and
2. our current implementation.

This is useful because a pure constant rescale from Table 3 -> Table 2 should
leave Sharpe, Sortino, % +ve, and Ave P/L almost unchanged. The paper tables do
not always behave that way, so we want a compact report before tuning the
pipeline further.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import (
    DEFAULT_SIGMA_TGT,
    apply_portfolio_vol_scaling,
    compute_metrics,
    compute_portfolio_returns,
    load_contracts,
)
from config import ASSET_CLASSES, PAPER_TABLE2, PAPER_TABLE3


ASSETS = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All']
INVARIANT_METRICS = ['Sharpe', 'Sortino', '% +ve', 'Ave P/L']


def safe_ratio(v2, v3):
    if abs(v3) < 1e-12:
        return np.nan
    return v2 / v3


def ours_for_asset(asset, sigma_tgt, test_start, test_end, port_vol_target):
    if asset == 'All':
        raw = []
        for name in ASSET_CLASSES:
            raw.extend(load_contracts(name, test_start=test_start, test_end=test_end))
    else:
        raw = load_contracts(asset, test_start=test_start, test_end=test_end)
    if not raw:
        return None

    r_t3 = compute_portfolio_returns(raw, 'Long', sigma_tgt)
    r_t2 = apply_portfolio_vol_scaling(r_t3, port_vol_target)

    m3 = dict(zip(
        ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
         'MDD', 'Calmar', '% +ve', 'Ave P/L'],
        compute_metrics(r_t3),
    ))
    m2 = dict(zip(
        ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
         'MDD', 'Calmar', '% +ve', 'Ave P/L'],
        compute_metrics(r_t2),
    ))
    return m3, m2


def format_ratio(value):
    if np.isnan(value):
        return '   NA'
    return f'{value:5.3f}'


def print_block(title, t3, t2):
    k_std = safe_ratio(t2['std(R)'], t3['std(R)'])
    k_er = safe_ratio(t2['E(R)'], t3['E(R)'])
    k_dd = safe_ratio(t2['DD'], t3['DD'])
    invariant_deltas = {k: t2[k] - t3[k] for k in INVARIANT_METRICS}

    print(title)
    print(
        f"  k_std={format_ratio(k_std)}  "
        f"k_er={format_ratio(k_er)}  "
        f"k_dd={format_ratio(k_dd)}"
    )
    print(
        "  deltas: "
        + "  ".join(f"{k}={invariant_deltas[k]:+0.3f}" for k in INVARIANT_METRICS)
    )


def main():
    parser = argparse.ArgumentParser(description='Analyze Table 2 vs Table 3 bridge')
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA_TGT)
    parser.add_argument('--test-start', default='2011-01-01')
    parser.add_argument('--test-end', default='2019-12-31')
    parser.add_argument('--port-vol-target', type=float, default=0.97)
    args = parser.parse_args()

    print('=' * 88)
    print('Table 2 vs Table 3 Bridge Diagnostic')
    print(
        f"sigma={args.sigma}  test={args.test_start}..{args.test_end}  "
        f"port_vol_target={args.port_vol_target}"
    )
    print('=' * 88)

    for asset in ASSETS:
        print(f"\n{asset}")
        print('-' * 88)

        paper_t3 = PAPER_TABLE3[asset]['Long']
        paper_t2 = PAPER_TABLE2[asset]['Long']
        print_block('Paper', paper_t3, paper_t2)

        ours = ours_for_asset(
            asset,
            sigma_tgt=args.sigma,
            test_start=args.test_start,
            test_end=args.test_end,
            port_vol_target=args.port_vol_target,
        )
        if ours is None:
            print('Ours')
            print('  no data loaded')
            continue

        ours_t3, ours_t2 = ours
        print_block('Ours ', ours_t3, ours_t2)


if __name__ == '__main__':
    main()
