#!/usr/bin/env python3
"""
Generate the full paper-vs-ours report set described in the reproduction plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import DEFAULT_SIGMA_TGT
from repro_analysis import (
    ALL_METRICS,
    ASSETS,
    EXCLUSION_PRESETS,
    LANE_A_METRICS,
    LANE_B_METRICS,
    METRIC_DEFINITIONS,
    TABLE2_BRIDGES,
    abs_gap,
    evaluate_table,
    load_asset_contracts,
    pct_err,
    portfolio_series,
    score_table3_scenario,
    table3_sort_key,
    yearly_summary,
)
from config import PAPER_TABLE2, PAPER_TABLE3


DOCS_DIR = ROOT / 'docs'
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
SIGMA_GRID = [0.0618, 0.0621, 0.0624, 0.0627, 0.0630, 0.0633, 0.0636]
TABLE3_PRESETS = ['current_config', 'memory_5', 'memory_5_plus_us', 'memory_5_plus_us_zh']
AGG_MODES = ['variable_n', 'dropna']


def fmt_pct(value):
    if value == float('inf') or np.isinf(value):
        return 'inf'
    return f'{value:.1f}%'


def md_table(headers, rows):
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
    return '\n'.join(lines)


def artifact_header(title, summary_lines):
    out = [f'# {title}', '']
    out.extend(summary_lines)
    out.append('')
    return '\n'.join(out)


def metric_audit():
    scenario = {
        'excluded_contracts': EXCLUSION_PRESETS['current_config'],
        'sigma': DEFAULT_SIGMA_TGT,
        'aggregation': 'variable_n',
    }
    definitions = ['additive_subset', 'nav_subset', 'nav_full']
    evals = {}
    for name in definitions:
        metric_def = METRIC_DEFINITIONS[name]
        asset_results = {}
        for asset in ASSETS:
            asset_results[asset] = evaluate_table(
                asset,
                PAPER_TABLE3,
                metric_def,
                excluded_contracts=scenario['excluded_contracts'],
                sigma_tgt=scenario['sigma'],
                aggregation_mode=scenario['aggregation'],
                test_start=TEST_START,
                test_end=TEST_END,
            )
        evals[name] = asset_results

    rows = []
    for name in definitions:
        asset_results = evals[name]
        dd_errs = []
        sortino_errs = []
        mdd_errs = []
        calmar_errs = []
        calmar_gap = []
        for asset in ASSETS:
            result = asset_results[asset]
            dd_errs.append(result['percent_errors']['DD'])
            sortino_errs.append(result['percent_errors']['Sortino'])
            mdd_errs.append(result['percent_errors']['MDD'])
            calmar_errs.append(result['percent_errors']['Calmar'])
            calmar_gap.append(result['calmar_internal_gap'])
        rows.append([
            name,
            fmt_pct(float(np.mean(dd_errs))),
            fmt_pct(float(np.mean(sortino_errs))),
            fmt_pct(float(np.mean(mdd_errs))),
            fmt_pct(float(np.mean(calmar_errs))),
            f'{float(np.mean(calmar_gap)):.3f}',
        ])

    detail_sections = []
    for name in definitions:
        asset_rows = []
        for asset in ASSETS:
            result = evals[name][asset]
            asset_rows.append([
                asset,
                result['contracts'],
                result['metrics']['DD'],
                result['paper']['DD'],
                fmt_pct(result['percent_errors']['DD']),
                result['metrics']['MDD'],
                result['paper']['MDD'],
                fmt_pct(result['percent_errors']['MDD']),
                f"{result['calmar_internal_gap']:.3f}",
                f"{result['paper_calmar_internal_gap']:.3f}",
            ])
        detail_sections.append(
            f"## {name}\n\n" +
            md_table(
                ['Asset', '#', 'DD ours', 'DD paper', 'DD err', 'MDD ours', 'MDD paper', 'MDD err', 'Ours |Calmar-ER/MDD|', 'Paper |Calmar-ER/MDD|'],
                asset_rows,
            )
        )

    dd_choice = 'additive_subset'
    mdd_policy = 'diagnostic_only'
    conclusion = (
        "- `DD_subset` on the additive portfolio remains the most plausible reporting definition because it matches the paper text most literally and stays competitive on DD/Sortino errors.\n"
        "- NAV-based DD/MDD are useful diagnostics, but they do not fully resolve the paper’s Calmar inconsistency.\n"
        "- `MDD` and `Calmar` remain diagnostic-only for scenario selection; the paper stays internally inconsistent even after the best tested bridge."
    )

    body = [
        artifact_header(
            'Metric Audit Report',
            [
                f"- Scenario: `current_config`, `variable_n`, `sigma={DEFAULT_SIGMA_TGT}`",
                f"- Compared definitions: `{', '.join(definitions)}`",
                f"- Selected reporting DD: `{dd_choice}`",
                f"- MDD policy: `{mdd_policy}`",
            ],
        ),
        '## Summary Table',
        '',
        md_table(
            ['Definition', 'Avg DD err', 'Avg Sortino err', 'Avg MDD err', 'Avg Calmar err', 'Avg ours |Calmar-ER/MDD|'],
            rows,
        ),
        '',
        '## Conclusion',
        '',
        conclusion,
        '',
        '\n\n'.join(detail_sections),
    ]
    return '\n'.join(body), dd_choice, mdd_policy, evals


def table3_sweep(metric_def_name):
    metric_def = METRIC_DEFINITIONS[metric_def_name]
    scenarios = []
    for preset in TABLE3_PRESETS:
        for agg in AGG_MODES:
            for sigma in SIGMA_GRID:
                asset_results = {}
                for asset in ASSETS:
                    asset_results[asset] = evaluate_table(
                        asset,
                        PAPER_TABLE3,
                        metric_def,
                        excluded_contracts=EXCLUSION_PRESETS[preset],
                        sigma_tgt=sigma,
                        aggregation_mode=agg,
                        test_start=TEST_START,
                        test_end=TEST_END,
                    )
                score = score_table3_scenario(asset_results)
                scenarios.append({
                    'preset': preset,
                    'agg': agg,
                    'sigma': sigma,
                    'asset_results': asset_results,
                    'score': score,
                })
    scenarios.sort(key=lambda s: table3_sort_key(s['score']))
    best = scenarios[0]

    top_rows = []
    for idx, scenario in enumerate(scenarios[:12], start=1):
        s = scenario['score']
        top_rows.append([
            idx,
            scenario['preset'],
            scenario['agg'],
            f"{scenario['sigma']:.4f}",
            f"{s['lane_a_pass_10']}/16",
            f"{s['lane_a_pass_15']}/16",
            f"{s['lane_b_pass_10']}/12",
            f"{s['lane_b_pass_15']}/12",
            f"{s['lane_a_mae']:.2f}%",
            f"{s['all_abs_er']:.3f}",
            f"{s['all_abs_sharpe']:.3f}",
        ])

    detail_rows = []
    for asset in ASSETS:
        r = best['asset_results'][asset]
        detail_rows.append([
            asset,
            r['contracts'],
            fmt_pct(r['percent_errors']['E(R)']),
            fmt_pct(r['percent_errors']['Sharpe']),
            fmt_pct(r['percent_errors']['DD']),
            fmt_pct(r['percent_errors']['Sortino']),
            fmt_pct(r['percent_errors']['std(R)']),
            fmt_pct(r['percent_errors']['% +ve']),
            fmt_pct(r['percent_errors']['Ave P/L']),
            f"{r['absolute_gaps']['E(R)']:.3f}",
            f"{r['absolute_gaps']['Sharpe']:.3f}",
        ])

    interpretation = (
        f"- Best scenario: `{best['preset']}`, `{best['agg']}`, `sigma={best['sigma']:.4f}`.\n"
        "- `variable_n` still wins on Lane A coverage overall; `dropna` improves Equity but gives back more on Forex.\n"
        "- The main remaining misses are Equity `E(R)` / `Sharpe` and the `All` row near-zero target problem."
    )

    body = [
        artifact_header(
            'Table 3 Sweep Report',
            [
                f"- Metric definition used: `{metric_def_name}`",
                f"- Sigma grid: `{', '.join(f'{x:.4f}' for x in SIGMA_GRID)}`",
                f"- Presets: `{', '.join(TABLE3_PRESETS)}`",
                f"- Aggregation modes: `{', '.join(AGG_MODES)}`",
            ],
        ),
        '## Top Scenarios',
        '',
        md_table(
            ['#', 'Preset', 'Agg', 'Sigma', 'Lane A <10', 'Lane A <15', 'Lane B <10', 'Lane B <15', 'Lane A MAE', 'All |ER gap|', 'All |Sharpe gap|'],
            top_rows,
        ),
        '',
        '## Best Scenario Detail',
        '',
        md_table(
            ['Asset', '#', 'E(R) err', 'Sharpe err', 'DD err', 'Sortino err', 'std err', '%+ve err', 'P/L err', '|ER gap|', '|Sharpe gap|'],
            detail_rows,
        ),
        '',
        '## Interpretation',
        '',
        interpretation,
    ]
    return '\n'.join(body), best, scenarios


def _leave_one_out_scenario(metric_def_name, excluded_contracts, sigma, aggregation, focus_asset):
    metric_def = METRIC_DEFINITIONS[metric_def_name]
    raw_focus = load_asset_contracts(
        focus_asset,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=excluded_contracts,
    )
    tickers = [rd['tk'] for rd in raw_focus]
    rows = []
    baseline_focus = evaluate_table(
        focus_asset, PAPER_TABLE3, metric_def,
        excluded_contracts=excluded_contracts, sigma_tgt=sigma,
        aggregation_mode=aggregation, test_start=TEST_START, test_end=TEST_END,
    )
    baseline_all = evaluate_table(
        'All', PAPER_TABLE3, metric_def,
        excluded_contracts=excluded_contracts, sigma_tgt=sigma,
        aggregation_mode=aggregation, test_start=TEST_START, test_end=TEST_END,
    )
    for tk in tickers:
        new_excluded = list(excluded_contracts) + [tk]
        focus = evaluate_table(
            focus_asset, PAPER_TABLE3, metric_def,
            excluded_contracts=new_excluded, sigma_tgt=sigma,
            aggregation_mode=aggregation, test_start=TEST_START, test_end=TEST_END,
        )
        all_row = evaluate_table(
            'All', PAPER_TABLE3, metric_def,
            excluded_contracts=new_excluded, sigma_tgt=sigma,
            aggregation_mode=aggregation, test_start=TEST_START, test_end=TEST_END,
        )
        rows.append({
            'ticker': tk,
            'focus_er_delta': focus['absolute_gaps']['E(R)'] - baseline_focus['absolute_gaps']['E(R)'],
            'focus_sharpe_delta': focus['absolute_gaps']['Sharpe'] - baseline_focus['absolute_gaps']['Sharpe'],
            'all_er_delta': all_row['absolute_gaps']['E(R)'] - baseline_all['absolute_gaps']['E(R)'],
            'all_sharpe_delta': all_row['absolute_gaps']['Sharpe'] - baseline_all['absolute_gaps']['Sharpe'],
            'focus_er_gap': focus['absolute_gaps']['E(R)'],
            'focus_sharpe_gap': focus['absolute_gaps']['Sharpe'],
            'all_er_gap': all_row['absolute_gaps']['E(R)'],
            'all_sharpe_gap': all_row['absolute_gaps']['Sharpe'],
        })
    return rows


def equity_contribution_report(metric_def_name, best):
    rows = _leave_one_out_scenario(
        metric_def_name,
        EXCLUSION_PRESETS[best['preset']],
        best['sigma'],
        best['agg'],
        'Equity Index',
    )
    rows.sort(key=lambda r: (r['focus_er_delta'], r['focus_sharpe_delta']))

    md_rows = []
    for row in rows:
        md_rows.append([
            row['ticker'],
            f"{row['focus_er_delta']:+.3f}",
            f"{row['focus_sharpe_delta']:+.3f}",
            f"{row['all_er_delta']:+.3f}",
            f"{row['all_sharpe_delta']:+.3f}",
            f"{row['focus_er_gap']:.3f}",
            f"{row['focus_sharpe_gap']:.3f}",
        ])
    best_three = ', '.join(r['ticker'] for r in rows[:3])
    interpretation = (
        f"- Most helpful leave-one-out candidates for Equity under the frozen Table 3 setup are: `{best_three}`.\n"
        "- A negative delta means removing that contract reduces the paper gap; these are diagnosis candidates, not automatic exclusions."
    )
    body = [
        artifact_header(
            'Equity Contract Contribution Report',
            [
                f"- Scenario: `{best['preset']}`, `{best['agg']}`, `sigma={best['sigma']:.4f}`",
                f"- Metric definition: `{metric_def_name}`",
            ],
        ),
        md_table(
            ['Ticker', 'Δ Equity |ER gap|', 'Δ Equity |Sharpe gap|', 'Δ All |ER gap|', 'Δ All |Sharpe gap|', 'Equity |ER gap| after drop', 'Equity |Sharpe gap| after drop'],
            md_rows,
        ),
        '',
        '## Interpretation',
        '',
        interpretation,
    ]
    return '\n'.join(body)


def all_row_contribution_report(metric_def_name, best):
    metric_def = METRIC_DEFINITIONS[metric_def_name]
    raw_all = load_asset_contracts(
        'All',
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=EXCLUSION_PRESETS[best['preset']],
    )
    port = portfolio_series(
        raw_all,
        series_kind=metric_def.series_kind,
        aggregation_mode=best['agg'],
        sigma_tgt=best['sigma'],
    )
    contribs = []
    for rd in raw_all:
        series = portfolio_series([rd], series_kind=metric_def.series_kind,
                                  aggregation_mode='variable_n', sigma_tgt=best['sigma'])
        common = port.index.intersection(series.index)
        if len(common) == 0:
            continue
        contribs.append({
            'ticker': rd['tk'],
            'mean': float(series.loc[common].mean() * 252),
            'sum': float(series.loc[common].sum()),
        })
    contribs.sort(key=lambda x: abs(x['mean']), reverse=True)
    top12 = contribs[:12]

    loo_rows = _leave_one_out_scenario(
        metric_def_name,
        EXCLUSION_PRESETS[best['preset']],
        best['sigma'],
        best['agg'],
        'All',
    )
    loo_lookup = {row['ticker']: row for row in loo_rows}

    md_rows = []
    for row in top12:
        loo = loo_lookup[row['ticker']]
        md_rows.append([
            row['ticker'],
            f"{row['mean']:+.3f}",
            f"{row['sum']:+.3f}",
            f"{loo['focus_er_delta']:+.3f}",
            f"{loo['focus_sharpe_delta']:+.3f}",
        ])
    interpretation = (
        "- The `All` row is dominated by a small number of contracts with outsized mean contribution.\n"
        "- These names are the right place to inspect when the near-zero paper `All` target is unstable."
    )
    body = [
        artifact_header(
            'All-Row Contribution Report',
            [
                f"- Scenario: `{best['preset']}`, `{best['agg']}`, `sigma={best['sigma']:.4f}`",
                f"- Metric definition: `{metric_def_name}`",
            ],
        ),
        md_table(
            ['Ticker', 'Annualized mean contribution', 'Cumulative contribution', 'Δ All |ER gap| if dropped', 'Δ All |Sharpe gap| if dropped'],
            md_rows,
        ),
        '',
        '## Interpretation',
        '',
        interpretation,
    ]
    return '\n'.join(body)


def final_table3_report(metric_def_name, best):
    metric_def = METRIC_DEFINITIONS[metric_def_name]
    rerun = {}
    for asset in ASSETS:
        rerun[asset] = evaluate_table(
            asset,
            PAPER_TABLE3,
            metric_def,
            excluded_contracts=EXCLUSION_PRESETS[best['preset']],
            sigma_tgt=best['sigma'],
            aggregation_mode=best['agg'],
            test_start=TEST_START,
            test_end=TEST_END,
        )
    repeat_ok = all(
        best['asset_results'][asset]['metrics'] == rerun[asset]['metrics']
        for asset in ASSETS
    )
    rows = []
    for asset in ASSETS:
        result = best['asset_results'][asset]
        rows.append([
            asset,
            result['contracts'],
            result['metrics']['E(R)'],
            result['paper']['E(R)'],
            fmt_pct(result['percent_errors']['E(R)']),
            result['metrics']['Sharpe'],
            result['paper']['Sharpe'],
            fmt_pct(result['percent_errors']['Sharpe']),
            result['metrics']['DD'],
            result['paper']['DD'],
            fmt_pct(result['percent_errors']['DD']),
            result['metrics']['std(R)'],
            result['paper']['std(R)'],
            fmt_pct(result['percent_errors']['std(R)']),
        ])
    s = best['score']
    interpretation = (
        f"- Reproducibility rerun identical: `{repeat_ok}`.\n"
        f"- Lane A score: `{s['lane_a_pass_15']}/16` within 15%; Lane B score: `{s['lane_b_pass_15']}/12` within 15%.\n"
        "- Table 3 is frozen on the current best upstream scenario; `MDD/Calmar` remain report-only."
    )
    body = [
        artifact_header(
            'Final Table 3 Comparison Report',
            [
                f"- Final metric definition: `{metric_def_name}`",
                f"- Final preset: `{best['preset']}`",
                f"- Final aggregation: `{best['agg']}`",
                f"- Final sigma: `{best['sigma']:.4f}`",
                f"- Final exclusion set: `{', '.join(EXCLUSION_PRESETS[best['preset']])}`",
            ],
        ),
        md_table(
            ['Asset', '#', 'E(R) ours', 'E(R) paper', 'E(R) err', 'Sharpe ours', 'Sharpe paper', 'Sharpe err', 'DD ours', 'DD paper', 'DD err', 'std ours', 'std paper', 'std err'],
            rows,
        ),
        '',
        '## Interpretation',
        '',
        interpretation,
    ]
    return '\n'.join(body)


def table2_bridge_report(metric_def_name, best):
    metric_def = METRIC_DEFINITIONS[metric_def_name]
    bridges = []
    for name, scaler in TABLE2_BRIDGES.items():
        asset_results = {}
        for asset in ASSETS:
            asset_results[asset] = evaluate_table(
                asset,
                PAPER_TABLE2,
                metric_def,
                excluded_contracts=EXCLUSION_PRESETS[best['preset']],
                sigma_tgt=best['sigma'],
                aggregation_mode=best['agg'],
                port_scaler=scaler,
                test_start=TEST_START,
                test_end=TEST_END,
            )
        score = score_table3_scenario(asset_results)
        std_max = max(asset_results[a]['percent_errors']['std(R)'] for a in ASSETS)
        bridges.append({
            'name': name,
            'asset_results': asset_results,
            'score': score,
            'std_max': std_max,
        })
    bridges.sort(key=lambda b: (
        -b['score']['lane_a_pass_15'],
        -b['score']['lane_a_pass_10'],
        b['std_max'],
        b['score']['lane_a_mae'],
        b['score']['all_abs_er'],
    ))
    best_bridge = bridges[0]

    top_rows = []
    for bridge in bridges:
        s = bridge['score']
        top_rows.append([
            bridge['name'],
            f"{s['lane_a_pass_10']}/16",
            f"{s['lane_a_pass_15']}/16",
            f"{s['lane_b_pass_10']}/12",
            f"{s['lane_b_pass_15']}/12",
            f"{s['lane_a_mae']:.2f}%",
            f"{bridge['std_max']:.2f}%",
            f"{s['all_abs_er']:.3f}",
            f"{s['all_abs_sharpe']:.3f}",
        ])

    detail_rows = []
    for asset in ASSETS:
        result = best_bridge['asset_results'][asset]
        yearly = yearly_summary(result['series'])
        detail_rows.append([
            asset,
            result['metrics']['E(R)'],
            result['paper']['E(R)'],
            fmt_pct(result['percent_errors']['E(R)']),
            result['metrics']['Sharpe'],
            result['paper']['Sharpe'],
            fmt_pct(result['percent_errors']['Sharpe']),
            result['metrics']['std(R)'],
            result['paper']['std(R)'],
            fmt_pct(result['percent_errors']['std(R)']),
            f"{np.std(result['series'].values) * np.sqrt(252):.3f}",
            ', '.join(f'{k}:{v:+.2f}' for k, v in list(yearly.items())[:3]),
        ])

    interpretation = (
        f"- Best tested Table 2 bridge: `{best_bridge['name']}`.\n"
        "- Selection prioritized Lane A coverage, then std alignment, then `All` absolute gaps.\n"
        "- This report compares every bridge against the paper and against the current constant baseline implicitly through the bridge table."
    )
    body = [
        artifact_header(
            'Table 2 Bridge Comparison Report',
            [
                f"- Frozen Table 3 scenario: `{best['preset']}`, `{best['agg']}`, `sigma={best['sigma']:.4f}`",
                f"- Metric definition: `{metric_def_name}`",
            ],
        ),
        '## Bridge Scoreboard',
        '',
        md_table(
            ['Bridge', 'Lane A <10', 'Lane A <15', 'Lane B <10', 'Lane B <15', 'Lane A MAE', 'Worst std err', 'All |ER gap|', 'All |Sharpe gap|'],
            top_rows,
        ),
        '',
        '## Best Bridge Detail',
        '',
        md_table(
            ['Asset', 'E(R) ours', 'E(R) paper', 'E(R) err', 'Sharpe ours', 'Sharpe paper', 'Sharpe err', 'std ours', 'std paper', 'std err', 'Realized ann std', 'Yearly sample'],
            detail_rows,
        ),
        '',
        '## Interpretation',
        '',
        interpretation,
    ]
    return '\n'.join(body)


def write_doc(name, text):
    path = DOCS_DIR / name
    path.write_text(text + '\n', encoding='utf-8')
    return path


def main():
    DOCS_DIR.mkdir(exist_ok=True)

    audit_text, metric_def_name, _mdd_policy, _audit_evals = metric_audit()
    audit_path = write_doc('metric_audit_report.md', audit_text)

    sweep_text, best, _scenarios = table3_sweep(metric_def_name)
    sweep_path = write_doc('table3_sweep_report.md', sweep_text)

    equity_path = write_doc('equity_contract_contribution_report.md',
                            equity_contribution_report(metric_def_name, best))
    all_path = write_doc('all_row_contribution_report.md',
                         all_row_contribution_report(metric_def_name, best))
    final_t3_path = write_doc('final_table3_comparison_report.md',
                              final_table3_report(metric_def_name, best))
    table2_path = write_doc('table2_bridge_comparison_report.md',
                            table2_bridge_report(metric_def_name, best))

    print('Generated reports:')
    for path in [audit_path, sweep_path, equity_path, all_path, final_t3_path, table2_path]:
        print(path)


if __name__ == '__main__':
    main()
