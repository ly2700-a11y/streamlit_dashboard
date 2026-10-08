#!/usr/bin/env python3
"""
Search for evidence-backed per-contract data-source overrides that improve
Table 3 without changing the active universe.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ASSET_CLASSES, EXCLUDED_CONTRACTS, PAPER_TABLE3
from repro_analysis import METRIC_DEFINITIONS, evaluate_table


DOC_PATH = ROOT / 'docs' / 'source_override_search_report.md'
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
SIGMA = 0.0627
METRIC_DEF = METRIC_DEFINITIONS['additive_subset']
ASSETS = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All']
SEARCH_ASSETS = ['Commodity', 'Equity Index']
SOURCES = ['REV', 'RAD_REGEN']
GREEDY_POOL = 12


def md_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(lines)


def scenario_results(source_overrides):
    results = {}
    for asset in ASSETS:
        results[asset] = evaluate_table(
            asset,
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=EXCLUDED_CONTRACTS,
            sigma_tgt=SIGMA,
            aggregation_mode='variable_n',
            source_overrides=source_overrides,
            test_start=TEST_START,
            test_end=TEST_END,
        )
    return results


def score_results(results):
    total_10 = 0
    total_15 = 0
    focus_er = 0.0
    focus_sharpe = 0.0
    all_er = results['All']['absolute_gaps']['E(R)']
    for asset in ASSETS:
        errs = results[asset]['percent_errors']
        total_10 += sum(1 for name in results[asset]['metrics'] if errs[name] < 10)
        total_15 += sum(1 for name in results[asset]['metrics'] if errs[name] < 15)
    for asset in ['Commodity', 'Equity Index', 'All']:
        focus_er += results[asset]['absolute_gaps']['E(R)']
        focus_sharpe += results[asset]['absolute_gaps']['Sharpe']
    return {
        'n10': total_10,
        'n15': total_15,
        'focus_er': round(focus_er, 3),
        'focus_sharpe': round(focus_sharpe, 3),
        'all_er': round(all_er, 3),
    }


def better_than(left, right):
    return (
        left['n15'],
        left['n10'],
        -left['focus_er'],
        -left['focus_sharpe'],
        -left['all_er'],
    ) > (
        right['n15'],
        right['n10'],
        -right['focus_er'],
        -right['focus_sharpe'],
        -right['all_er'],
    )


def candidate_contracts():
    out = []
    for asset in SEARCH_ASSETS:
        for tk in ASSET_CLASSES[asset]:
            if tk not in EXCLUDED_CONTRACTS:
                out.append((asset, tk))
    return out


def one_by_one_search(base_results):
    baseline = score_results(base_results)
    rows = []
    for asset, tk in candidate_contracts():
        for source in SOURCES:
            overrides = {tk: source}
            results = scenario_results(overrides)
            score = score_results(results)
            rows.append({
                'asset': asset,
                'ticker': tk,
                'source': source,
                'delta_n15': score['n15'] - baseline['n15'],
                'delta_n10': score['n10'] - baseline['n10'],
                'delta_focus_er': round(score['focus_er'] - baseline['focus_er'], 3),
                'delta_focus_sharpe': round(score['focus_sharpe'] - baseline['focus_sharpe'], 3),
                'delta_all_er': round(score['all_er'] - baseline['all_er'], 3),
                'asset_er': results[asset]['metrics']['E(R)'],
                'asset_er_gap': results[asset]['absolute_gaps']['E(R)'],
                'asset_sharpe_gap': results[asset]['absolute_gaps']['Sharpe'],
                'all_er_gap': results['All']['absolute_gaps']['E(R)'],
            })
    rows.sort(
        key=lambda r: (
            -r['delta_n15'],
            -r['delta_n10'],
            r['delta_focus_er'],
            r['delta_focus_sharpe'],
            r['delta_all_er'],
        )
    )
    return rows


def greedy_search(base_results, one_by_one_rows):
    current_overrides = {}
    current_results = base_results
    current_score = score_results(current_results)
    steps = []

    shortlist = []
    seen = set()
    for row in one_by_one_rows:
        key = (row['ticker'], row['source'])
        if key in seen:
            continue
        shortlist.append(row)
        seen.add(key)
        if len(shortlist) >= GREEDY_POOL:
            break

    remaining = {(row['ticker'], row['asset'], row['source']) for row in shortlist}
    while remaining:
        best = None
        for tk, asset, source in sorted(remaining):
            trial_overrides = dict(current_overrides)
            trial_overrides[tk] = source
            trial_results = scenario_results(trial_overrides)
            trial_score = score_results(trial_results)
            candidate = {
                'ticker': tk,
                'asset': asset,
                'source': source,
                'overrides': trial_overrides,
                'results': trial_results,
                'score': trial_score,
            }
            if best is None or better_than(candidate['score'], best['score']):
                best = candidate
        if best is None or not better_than(best['score'], current_score):
            break
        current_overrides = best['overrides']
        current_results = best['results']
        current_score = best['score']
        remaining = {entry for entry in remaining if entry[0] != best['ticker']}
        steps.append({
            'ticker': best['ticker'],
            'asset': best['asset'],
            'source': best['source'],
            'score': current_score,
            'commodity_er_gap': current_results['Commodity']['absolute_gaps']['E(R)'],
            'equity_er_gap': current_results['Equity Index']['absolute_gaps']['E(R)'],
            'all_er_gap': current_results['All']['absolute_gaps']['E(R)'],
        })
    return current_overrides, current_results, steps


def scenario_summary_rows(results):
    rows = []
    for asset in ASSETS:
        r = results[asset]
        rows.append([
            asset,
            r['contracts'],
            f"{r['metrics']['E(R)']:+.3f}",
            f"{r['paper']['E(R)']:+.3f}",
            f"{r['absolute_gaps']['E(R)']:.3f}",
            f"{r['metrics']['Sharpe']:+.3f}",
            f"{r['paper']['Sharpe']:+.3f}",
            f"{r['absolute_gaps']['Sharpe']:.3f}",
            sum(1 for e in r['percent_errors'].values() if e < 10),
            sum(1 for e in r['percent_errors'].values() if e < 15),
        ])
    return rows


def main():
    baseline_results = scenario_results({})
    baseline_score = score_results(baseline_results)
    one_by_one = one_by_one_search(baseline_results)
    best_overrides, best_results, steps = greedy_search(baseline_results, one_by_one)
    best_score = score_results(best_results)

    top_rows = []
    for row in one_by_one[:20]:
        top_rows.append([
            row['asset'],
            row['ticker'],
            row['source'],
            f"{row['delta_n15']:+d}",
            f"{row['delta_n10']:+d}",
            f"{row['delta_focus_er']:+.3f}",
            f"{row['delta_focus_sharpe']:+.3f}",
            f"{row['delta_all_er']:+.3f}",
            f"{row['asset_er']:+.3f}",
            f"{row['asset_er_gap']:.3f}",
            f"{row['all_er_gap']:.3f}",
        ])

    if steps:
        step_rows = [
            [
                idx,
                step['asset'],
                step['ticker'],
                step['source'],
                step['score']['n15'],
                step['score']['n10'],
                f"{step['score']['focus_er']:.3f}",
                f"{step['commodity_er_gap']:.3f}",
                f"{step['equity_er_gap']:.3f}",
                f"{step['all_er_gap']:.3f}",
            ]
            for idx, step in enumerate(steps, start=1)
        ]
    else:
        step_rows = [['-', '-', '-', '-', baseline_score['n15'], baseline_score['n10'], f"{baseline_score['focus_er']:.3f}", '-', '-', '-']]

    override_lines = ['- None']
    if best_overrides:
        override_lines = [f"- `{tk}` -> `{source}`" for tk, source in sorted(best_overrides.items())]

    interpretation = [
        '- Baseline uses the active 46-contract universe and current additive metric path.',
        '- One-by-one results show whether a single contract-level source swap improves the full 45-comparison score or at least reduces the key `E(R)` gaps.',
        '- Greedy results show whether those improvements stack, which is the real test for a hybrid source-map fix.',
    ]
    if best_overrides:
        interpretation.append(
            f"- Best greedy scenario reached `n15={best_score['n15']}/45` and `n10={best_score['n10']}/45`, versus baseline `n15={baseline_score['n15']}/45`, `n10={baseline_score['n10']}/45`."
        )
    else:
        interpretation.append('- No greedy source override improved the baseline objective under the current search scope.')

    body = '\n'.join([
        '# Source Override Search Report',
        '',
        '- Goal: improve Table 3 by changing data-source interpretation for included contracts, not by shrinking the universe.',
        f"- Metric definition: `{METRIC_DEF.name}`",
        f"- Search scope: `{', '.join(SEARCH_ASSETS)}` contracts only",
        f"- Candidate sources: `{', '.join(SOURCES)}`",
        f"- Greedy stack shortlist size: `{GREEDY_POOL}` best one-by-one candidates",
        '',
        '## Baseline Score',
        '',
        md_table(
            ['Scenario', 'n15', 'n10', 'Focus |E(R)| gap', 'Focus |Sharpe| gap', 'All |E(R)| gap'],
            [['Baseline', baseline_score['n15'], baseline_score['n10'], f"{baseline_score['focus_er']:.3f}", f"{baseline_score['focus_sharpe']:.3f}", f"{baseline_score['all_er']:.3f}"]],
        ),
        '',
        '## Baseline Asset Summary',
        '',
        md_table(
            ['Asset', '#', 'E(R) ours', 'E(R) paper', '|E(R) gap|', 'Sharpe ours', 'Sharpe paper', '|Sharpe gap|', 'n10', 'n15'],
            scenario_summary_rows(baseline_results),
        ),
        '',
        '## Best One-By-One Overrides',
        '',
        md_table(
            ['Asset', 'Ticker', 'Source', 'Δn15', 'Δn10', 'Δ focus |E(R)|', 'Δ focus |Sharpe|', 'Δ All |E(R)|', 'Asset E(R)', 'Asset |E(R)|', 'All |E(R)|'],
            top_rows,
        ),
        '',
        '## Greedy Accepted Overrides',
        '',
        '\n'.join(override_lines),
        '',
        md_table(
            ['Step', 'Asset', 'Ticker', 'Source', 'n15', 'n10', 'Focus |E(R)| gap', 'Commodity |E(R)|', 'Equity |E(R)|', 'All |E(R)|'],
            step_rows,
        ),
        '',
        '## Greedy Final Asset Summary',
        '',
        md_table(
            ['Asset', '#', 'E(R) ours', 'E(R) paper', '|E(R) gap|', 'Sharpe ours', 'Sharpe paper', '|Sharpe gap|', 'n10', 'n15'],
            scenario_summary_rows(best_results),
        ),
        '',
        '## Interpretation',
        '',
        '\n'.join(interpretation),
        '',
    ])

    DOC_PATH.write_text(body)
    print(f'Wrote {DOC_PATH}')
    print(f'Baseline: n15={baseline_score["n15"]}/45 n10={baseline_score["n10"]}/45 focus_er={baseline_score["focus_er"]:.3f}')
    print(f'Best    : n15={best_score["n15"]}/45 n10={best_score["n10"]}/45 focus_er={best_score["focus_er"]:.3f}')
    print(f'Overrides: {best_overrides}')


if __name__ == '__main__':
    main()
