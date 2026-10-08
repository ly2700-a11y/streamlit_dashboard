#!/usr/bin/env python3
"""
Generate an E(R)-focused attribution report across all 4 asset classes.

The report is designed to answer:
1. Where does realized E(R) come from mathematically?
2. Which included contracts drive current asset-class and All-row gaps?
3. Which currently excluded contracts are worth adding back?
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ASSET_CLASSES, EXCLUDED_CONTRACTS, PAPER_TABLE3
from repro_analysis import (
    METRIC_DEFINITIONS,
    evaluate_table,
    load_asset_contracts,
    realized_er_contributions,
)


DOC_PATH = ROOT / 'docs' / 'er_attribution_report.md'
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
SIGMA = 0.0627
METRIC_DEF = METRIC_DEFINITIONS['additive_subset']
ASSET_CLASSES_4 = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']
BASE_EXCLUDED = list(EXCLUDED_CONTRACTS)


def md_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(lines)


def scenario_compare(excluded_contracts):
    results = {}
    for asset in ASSET_CLASSES_4 + ['All']:
        results[asset] = evaluate_table(
            asset,
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=excluded_contracts,
            sigma_tgt=SIGMA,
            aggregation_mode='variable_n',
            test_start=TEST_START,
            test_end=TEST_END,
        )
    return results


def summarize_current_baseline(results):
    rows = []
    pass_10 = 0
    pass_15 = 0
    total = 0
    metrics = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']
    for asset in ASSET_CLASSES_4 + ['All']:
        r = results[asset]
        errs = r['percent_errors']
        if asset == 'All':
            # still report all 9, but near-zero ER/Sharpe/Sortino/Calmar explode
            n10 = sum(1 for m in metrics if errs[m] < 10)
            n15 = sum(1 for m in metrics if errs[m] < 15)
        else:
            n10 = sum(1 for m in metrics if errs[m] < 10)
            n15 = sum(1 for m in metrics if errs[m] < 15)
        pass_10 += n10
        pass_15 += n15
        total += len(metrics)
        rows.append([
            asset,
            r['contracts'],
            n10,
            n15,
            f"{r['metrics']['E(R)']:+.3f}",
            f"{r['paper']['E(R)']:+.3f}",
            f"{r['absolute_gaps']['E(R)']:.3f}",
            f"{r['metrics']['Sharpe']:+.3f}",
            f"{r['paper']['Sharpe']:+.3f}",
            f"{r['absolute_gaps']['Sharpe']:.3f}",
        ])
    return rows, pass_10, pass_15, total


def asset_contribution_rows(asset, excluded_contracts):
    raw = load_asset_contracts(
        asset,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=excluded_contracts,
    )
    rows = realized_er_contributions(raw, sigma_tgt=SIGMA)
    rows.sort(key=lambda r: r['er_contrib'])
    neg = rows[:3]
    pos = list(reversed(rows[-3:]))
    return neg, pos, rows


def leave_one_out_rows(asset, excluded_contracts, candidate_tickers):
    baseline_asset = evaluate_table(
        asset, PAPER_TABLE3, METRIC_DEF,
        excluded_contracts=excluded_contracts, sigma_tgt=SIGMA,
        aggregation_mode='variable_n', test_start=TEST_START, test_end=TEST_END,
    )
    baseline_all = evaluate_table(
        'All', PAPER_TABLE3, METRIC_DEF,
        excluded_contracts=excluded_contracts, sigma_tgt=SIGMA,
        aggregation_mode='variable_n', test_start=TEST_START, test_end=TEST_END,
    )
    raw = load_asset_contracts(
        asset,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=excluded_contracts,
    )
    raw_by_ticker = {rd['tk']: rd for rd in raw}
    rows = []
    for tk in candidate_tickers:
        if tk not in raw_by_ticker:
            continue
        new_excluded = list(excluded_contracts) + [tk]
        asset_result = evaluate_table(
            asset, PAPER_TABLE3, METRIC_DEF,
            excluded_contracts=new_excluded, sigma_tgt=SIGMA,
            aggregation_mode='variable_n', test_start=TEST_START, test_end=TEST_END,
        )
        all_result = evaluate_table(
            'All', PAPER_TABLE3, METRIC_DEF,
            excluded_contracts=new_excluded, sigma_tgt=SIGMA,
            aggregation_mode='variable_n', test_start=TEST_START, test_end=TEST_END,
        )
        rows.append({
            'ticker': tk,
            'asset_gap_delta': asset_result['absolute_gaps']['E(R)'] - baseline_asset['absolute_gaps']['E(R)'],
            'asset_sharpe_gap_delta': asset_result['absolute_gaps']['Sharpe'] - baseline_asset['absolute_gaps']['Sharpe'],
            'all_gap_delta': all_result['absolute_gaps']['E(R)'] - baseline_all['absolute_gaps']['E(R)'],
            'all_sharpe_gap_delta': all_result['absolute_gaps']['Sharpe'] - baseline_all['absolute_gaps']['Sharpe'],
        })
    rows.sort(key=lambda r: (r['asset_gap_delta'], r['asset_sharpe_gap_delta']))
    return rows


def add_back_rows(excluded_contracts):
    base = scenario_compare(excluded_contracts)
    rows = []
    for tk in excluded_contracts:
        asset = next(name for name, tickers in ASSET_CLASSES.items() if tk in tickers)
        new_excluded = [x for x in excluded_contracts if x != tk]
        asset_result = evaluate_table(
            asset, PAPER_TABLE3, METRIC_DEF,
            excluded_contracts=new_excluded, sigma_tgt=SIGMA,
            aggregation_mode='variable_n', test_start=TEST_START, test_end=TEST_END,
        )
        all_result = evaluate_table(
            'All', PAPER_TABLE3, METRIC_DEF,
            excluded_contracts=new_excluded, sigma_tgt=SIGMA,
            aggregation_mode='variable_n', test_start=TEST_START, test_end=TEST_END,
        )
        base_asset = base[asset]
        base_all = base['All']
        rows.append([
            tk,
            asset,
            f"{asset_result['metrics']['E(R)']:+.3f}",
            f"{asset_result['absolute_gaps']['E(R)'] - base_asset['absolute_gaps']['E(R)']:+.3f}",
            f"{asset_result['absolute_gaps']['Sharpe'] - base_asset['absolute_gaps']['Sharpe']:+.3f}",
            f"{all_result['absolute_gaps']['E(R)'] - base_all['absolute_gaps']['E(R)']:+.3f}",
            f"{all_result['absolute_gaps']['Sharpe'] - base_all['absolute_gaps']['Sharpe']:+.3f}",
        ])
    return rows


def section_for_asset(asset, excluded_contracts):
    neg, pos, all_rows = asset_contribution_rows(asset, excluded_contracts)
    candidate_tickers = []
    for r in neg + pos:
        if r['ticker'] not in candidate_tickers:
            candidate_tickers.append(r['ticker'])
    loo = leave_one_out_rows(asset, excluded_contracts, candidate_tickers)

    contrib_rows = []
    for r in neg + pos:
        contrib_rows.append([
            r['ticker'],
            f"{r['er_contrib']:+.3f}",
            f"{r['signal_contrib']:+.3f}",
            f"{r['tc_contrib']:+.3f}",
            r['n_obs'],
        ])

    loo_rows = []
    for r in loo:
        loo_rows.append([
            r['ticker'],
            f"{r['asset_gap_delta']:+.3f}",
            f"{r['asset_sharpe_gap_delta']:+.3f}",
            f"{r['all_gap_delta']:+.3f}",
            f"{r['all_sharpe_gap_delta']:+.3f}",
        ])

    return '\n'.join([
        f'## {asset}',
        '',
        '### Realized E(R) Contributors',
        '',
        md_table(
            ['Ticker', 'Trade contrib', 'Signal contrib', 'TC contrib', 'Obs'],
            contrib_rows,
        ),
        '',
        '### Best Leave-One-Out Diagnostics',
        '',
        md_table(
            ['Ticker', 'Δ asset |E(R) gap|', 'Δ asset |Sharpe gap|', 'Δ All |E(R) gap|', 'Δ All |Sharpe gap|'],
            loo_rows,
        ),
    ])


def main():
    current = scenario_compare(BASE_EXCLUDED)
    baseline_rows, pass_10, pass_15, total = summarize_current_baseline(current)
    addback = add_back_rows(BASE_EXCLUDED)

    text = '\n'.join([
        '# E(R) Attribution Report',
        '',
        '- Focus: continue Table 3 work toward the `40/45` target by proving where `E(R)` comes from and which contracts move the gap.',
        f'- Metric definition: `{METRIC_DEF.name}`',
        f'- Active excluded set: `{", ".join(BASE_EXCLUDED)}`',
        f'- Sigma: `{SIGMA}`',
        '',
        '## Math Identity',
        '',
        'For variable-N aggregation,',
        '',
        '```',
        'R_port,t = (1 / N_t) * Σ_i R_i,t',
        'E(R_port) = 252 * mean_t[(1 / N_t) * Σ_i R_i,t]',
        '```',
        '',
        'So each contract has realized annualized contribution',
        '',
        '```',
        'contrib_i = 252 * mean_t[I_i,t * R_i,t / N_t]',
        '```',
        '',
        'and because `R_i,t = signal_i,t - tc_i,t`, the same identity holds for the signal and transaction-cost pieces.',
        '',
        '## Current 45-Comparison Context',
        '',
        f'- Current full 9-metric baseline score: `n10={pass_10}/{total}`, `n15={pass_15}/{total}`',
        '- This confirms we are **not** at the `40/45` target yet and should keep pushing Table 3.',
        '',
        md_table(
            ['Asset', '#', 'n10', 'n15', 'E(R) ours', 'E(R) paper', '|E(R) gap|', 'Sharpe ours', 'Sharpe paper', '|Sharpe gap|'],
            baseline_rows,
        ),
        '',
        '## Add-Back Candidates From Current Excluded Set',
        '',
        md_table(
            ['Ticker', 'Asset', 'Asset E(R) after add-back', 'Δ asset |E(R) gap|', 'Δ asset |Sharpe gap|', 'Δ All |E(R) gap|', 'Δ All |Sharpe gap|'],
            addback,
        ),
        '',
        section_for_asset('Commodity', BASE_EXCLUDED),
        '',
        section_for_asset('Equity Index', BASE_EXCLUDED),
        '',
        section_for_asset('Fixed Income', BASE_EXCLUDED),
        '',
        section_for_asset('Forex', BASE_EXCLUDED),
        '',
        '## Interpretation',
        '',
        '- `E(R)` remains the bottleneck; std / %+ve / Ave P/L are still the stable metrics.',
        '- The add-back table tells us which excluded contracts are promising candidates under the current metric/scaling understanding.',
        '- The leave-one-out tables identify where the current included universe is still structurally fighting the paper, especially in Equity and the All row.',
        '- Next work should use these generated deltas to justify any future contract add-back or data-path investigation.',
    ])

    DOC_PATH.write_text(text + '\n', encoding='utf-8')
    print(DOC_PATH)


if __name__ == '__main__':
    main()
