#!/usr/bin/env python3
"""Run the legacy experimental Sign(R) configuration against both Table 3 and Table 2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS_SIGNR = ROOT / "baseline" / "signr"
if str(TESTS_SIGNR) not in sys.path:
    sys.path.insert(0, str(TESTS_SIGNR))

from baseline_run import (  # noqa: E402
    CORE_METRICS,
    DEFAULT_REPORT_SOURCE,
    DEFAULT_SIGMA_TGT,
    METRIC_NAMES,
    PAPER_TABLE2,
    PAPER_TABLE3,
    load_contracts,
    run_table,
)
import frontier_40plus_enumeration_signr as fe  # noqa: E402


ASSETS = ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]


def load_raw(asset: str, test_start: str, test_end: str):
    if asset == "All":
        raw = []
        for asset_name in ["Commodity", "Equity Index", "Fixed Income", "Forex"]:
            raw.extend(
                load_contracts(
                    asset_name,
                    test_start,
                    test_end,
                    excluded_contracts=fe.LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR,
                    source_overrides=fe.LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR,
                )
            )
        return raw
    return load_contracts(
        asset,
        test_start,
        test_end,
        excluded_contracts=fe.LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR,
        source_overrides=fe.LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR,
    )


def main():
    parser = argparse.ArgumentParser(description="Run legacy Sign(R) configuration for Table 2 and/or Table 3")
    parser.add_argument('--table', choices=['2', '3', 'both'], default='both')
    parser.add_argument('--asset', default=None)
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA_TGT)
    parser.add_argument('--test-start', default='2011-01-01')
    parser.add_argument('--test-end', default='2019-12-31')
    parser.add_argument('--port-vol-target', type=float, default=0.97)
    parser.add_argument('--all-metrics', action='store_true')
    parser.add_argument('--aggregation', choices=['variable_n', 'dropna'], default='variable_n')
    parser.add_argument('--report-source', choices=['trade', 'RISK_PRICE_SIGMA0'], default=DEFAULT_REPORT_SOURCE)
    args = parser.parse_args()

    metric_names = list(METRIC_NAMES) if args.all_metrics else CORE_METRICS
    asset_classes = [args.asset] if args.asset else ASSETS
    tables = []
    if args.table in ('3', 'both'):
        tables.append(('Table 3', PAPER_TABLE3, None))
    if args.table in ('2', 'both'):
        tables.append(('Table 2', PAPER_TABLE2, args.port_vol_target))

    print('Legacy experimental Sign(R) configuration')
    print('=' * 60)
    print(f'sigma_tgt: {args.sigma}')
    print(f'test window: {args.test_start} -> {args.test_end}')
    print(f'table mode: {args.table}')
    print(f'portfolio vol target (Table 2): {args.port_vol_target}')
    print(f"metric set: {'ALL 9' if args.all_metrics else 'CORE 5'}")
    print()
    print('source overrides:')
    for tk, src in sorted(fe.LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR.items()):
        print(f'  {tk}: {src}')
    print()
    print('excluded:')
    print('  ' + ', '.join(sorted(fe.LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR)))
    print()

    grand_n10, grand_n15, grand_total = 0, 0, 0
    for table_label, paper_table, port_vol in tables:
        for asset in asset_classes:
            raw = load_raw(asset, args.test_start, args.test_end)
            n10, n15, total = run_table(
                raw,
                asset,
                args.sigma,
                paper_table,
                table_label,
                port_vol_target=port_vol,
                metric_names=metric_names,
                aggregation_mode=args.aggregation,
                report_source=args.report_source,
                test_start=args.test_start,
                test_end=args.test_end,
                strategies=['Sign(R)'],
            )
            grand_n10 += n10
            grand_n15 += n15
            grand_total += total

    if grand_total > 0:
        print(f"\n{'=' * 60}")
        print(f"  GRAND TOTAL: ≤10%: {grand_n10}/{grand_total} | ≤15%: {grand_n15}/{grand_total}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
