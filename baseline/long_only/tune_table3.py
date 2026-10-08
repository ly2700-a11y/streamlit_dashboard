#!/usr/bin/env python3
"""
tune_table3.py

Table 3-only tuning harness.

Purpose:
- Freeze attention on the upstream reproduction problem (Table 3 only).
- Compare candidate exclusion sets, aggregation modes, and sigma_tgt values.
- Rank scenarios by closeness to the paper on the Long baseline.

Usage:
  python tests/tune_table3.py
  python tests/tune_table3.py --sigmas 0.062,0.0627,0.063,0.064
  python tests/tune_table3.py --presets memory_5,current_config --top 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import compute_metrics, compute_portfolio_returns, load_contracts
from config import EXCLUDED_CONTRACTS, PAPER_TABLE3


ASSETS = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All']
CORE_METRICS = ['E(R)', 'std(R)', 'Sharpe', '% +ve', 'Ave P/L']
ALL_METRICS = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
               'MDD', 'Calmar', '% +ve', 'Ave P/L']

EXCLUSION_PRESETS = {
    'none': [],
    'memory_5': ['LB', 'JO', 'ZO', 'CC', 'FB'],
    'memory_5_plus_us': ['LB', 'JO', 'ZO', 'CC', 'FB', 'US'],
    'memory_5_plus_us_zh': ['LB', 'JO', 'ZO', 'CC', 'FB', 'US', 'ZH'],
    'current_config': list(EXCLUDED_CONTRACTS),
}


def parse_csv_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def load_asset(asset, test_start, test_end, excluded_contracts):
    if asset == 'All':
        raw = []
        for name in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
            raw.extend(load_contracts(
                name,
                test_start=test_start,
                test_end=test_end,
                excluded_contracts=excluded_contracts,
            ))
        return raw
    return load_contracts(
        asset,
        test_start=test_start,
        test_end=test_end,
        excluded_contracts=excluded_contracts,
    )


def pct_err(ours, paper):
    if abs(paper) < 1e-12:
        return 0.0 if abs(ours) < 1e-12 else np.inf
    return abs((ours - paper) / abs(paper)) * 100.0


def evaluate_asset(asset, sigma, aggregation_mode, excluded_contracts, metric_names,
                   test_start, test_end):
    raw = load_asset(asset, test_start, test_end, excluded_contracts)
    if not raw:
        return None

    r_port = compute_portfolio_returns(
        raw,
        'Long',
        sigma,
        aggregation_mode=aggregation_mode,
    )
    metrics = dict(zip(ALL_METRICS, compute_metrics(r_port)))
    paper = PAPER_TABLE3[asset]['Long']
    errs = {name: pct_err(metrics[name], paper[name]) for name in metric_names}
    return {
        'contracts': len(raw),
        'metrics': metrics,
        'errs': errs,
        'n10': sum(1 for e in errs.values() if e < 10),
        'n15': sum(1 for e in errs.values() if e < 15),
        'mae': float(np.mean(list(errs.values()))),
        'max_err': float(np.max(list(errs.values()))),
    }


def evaluate_scenario(preset_name, sigma, aggregation_mode, metric_names,
                      test_start, test_end):
    excluded_contracts = EXCLUSION_PRESETS[preset_name]
    asset_results = {}
    for asset in ASSETS:
        result = evaluate_asset(
            asset,
            sigma=sigma,
            aggregation_mode=aggregation_mode,
            excluded_contracts=excluded_contracts,
            metric_names=metric_names,
            test_start=test_start,
            test_end=test_end,
        )
        if result is None:
            return None
        asset_results[asset] = result

    all_errs = [e for result in asset_results.values() for e in result['errs'].values()]
    return {
        'preset': preset_name,
        'excluded_contracts': excluded_contracts,
        'sigma': sigma,
        'aggregation': aggregation_mode,
        'asset_results': asset_results,
        'n10': sum(result['n10'] for result in asset_results.values()),
        'n15': sum(result['n15'] for result in asset_results.values()),
        'total_metrics': len(metric_names) * len(ASSETS),
        'mae': float(np.mean(all_errs)),
        'median_err': float(np.median(all_errs)),
        'max_err': float(np.max(all_errs)),
    }


def sort_key(result):
    return (-result['n15'], -result['n10'], result['mae'], result['median_err'], result['max_err'])


def print_top_results(results, top_n):
    print('=' * 130)
    print('TOP TABLE 3 SCENARIOS')
    print('=' * 130)
    print(f"{'#':>2} {'Preset':20s} {'Agg':10s} {'Sigma':>7s} {'n10':>7s} {'n15':>7s} {'MAE%':>8s} {'Med%':>8s} {'Max%':>8s}")
    print('-' * 130)
    for idx, result in enumerate(results[:top_n], start=1):
        print(
            f"{idx:>2} {result['preset']:20s} {result['aggregation']:10s} "
            f"{result['sigma']:>7.4f} "
            f"{result['n10']:>3d}/{result['total_metrics']:<3d} "
            f"{result['n15']:>3d}/{result['total_metrics']:<3d} "
            f"{result['mae']:>7.2f}% {result['median_err']:>7.2f}% {result['max_err']:>7.2f}%"
        )


def print_best_detail(result, metric_names):
    print('\n' + '=' * 130)
    print('BEST SCENARIO DETAIL')
    print('=' * 130)
    print(
        f"preset={result['preset']}  aggregation={result['aggregation']}  "
        f"sigma={result['sigma']:.4f}  excluded={','.join(result['excluded_contracts']) or '(none)'}"
    )
    print(
        f"score: n10={result['n10']}/{result['total_metrics']}  "
        f"n15={result['n15']}/{result['total_metrics']}  "
        f"MAE={result['mae']:.2f}%"
    )
    print('-' * 130)
    print(f"{'Asset':14s} {'#':>3s} {'n10':>5s} {'n15':>5s} {'MAE%':>8s}  Metric errors")
    print('-' * 130)
    for asset in ASSETS:
        asset_result = result['asset_results'][asset]
        metric_blob = '  '.join(
            f"{name}={asset_result['errs'][name]:.1f}%"
            for name in metric_names
        )
        print(
            f"{asset:14s} {asset_result['contracts']:>3d} "
            f"{asset_result['n10']:>3d}/{len(metric_names):<1d} "
            f"{asset_result['n15']:>3d}/{len(metric_names):<1d} "
            f"{asset_result['mae']:>7.2f}%  {metric_blob}"
        )


def main():
    parser = argparse.ArgumentParser(description='Tune Table 3 reproduction settings')
    parser.add_argument(
        '--sigmas',
        default='0.0620,0.0627,0.0630,0.0640,0.0650',
        help='Comma-separated sigma_tgt values to test',
    )
    parser.add_argument(
        '--presets',
        default='memory_5,memory_5_plus_us,memory_5_plus_us_zh,current_config',
        help='Comma-separated exclusion presets to test',
    )
    parser.add_argument(
        '--aggregation-modes',
        default='variable_n,dropna',
        help='Comma-separated aggregation modes to test',
    )
    parser.add_argument('--all-metrics', action='store_true',
                        help='Tune on all 9 metrics instead of the core 5')
    parser.add_argument('--test-start', default='2011-01-01')
    parser.add_argument('--test-end', default='2019-12-31')
    parser.add_argument('--top', type=int, default=8)
    args = parser.parse_args()

    sigmas = [float(x) for x in parse_csv_list(args.sigmas)]
    presets = parse_csv_list(args.presets)
    aggregation_modes = parse_csv_list(args.aggregation_modes)
    metric_names = ALL_METRICS if args.all_metrics else CORE_METRICS

    unknown = [name for name in presets if name not in EXCLUSION_PRESETS]
    if unknown:
        raise SystemExit(f'Unknown preset(s): {", ".join(unknown)}')

    results = []
    for preset in presets:
        for aggregation_mode in aggregation_modes:
            for sigma in sigmas:
                result = evaluate_scenario(
                    preset_name=preset,
                    sigma=sigma,
                    aggregation_mode=aggregation_mode,
                    metric_names=metric_names,
                    test_start=args.test_start,
                    test_end=args.test_end,
                )
                if result is not None:
                    results.append(result)

    if not results:
        raise SystemExit('No scenarios produced results')

    results.sort(key=sort_key)
    print_top_results(results, top_n=min(args.top, len(results)))
    print_best_detail(results[0], metric_names=metric_names)


if __name__ == '__main__':
    main()
