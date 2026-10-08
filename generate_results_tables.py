#!/usr/bin/env python3
"""
generate_results_tables.py
==========================
汇总所有策略（Long / Sign(R) / MACD / DQN / A2C）的指标，
生成论文风格的 Table 2（含组合波动率缩放）和 Table 3（原始信号）。

输出：伦钰婷做PPT要用的素材/results_tables.md

Usage:
    python3 generate_results_tables.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baseline_run import (
    load_contracts,
    compute_portfolio_returns,
    apply_portfolio_vol_scaling,
)
from config import (
    LEGACY_EXPERIMENTAL_OVERRIDES_LONG, LEGACY_EXPERIMENTAL_EXCLUDED_LONG,
    LEGACY_EXPERIMENTAL_OVERRIDES_MACD, LEGACY_EXPERIMENTAL_EXCLUDED_MACD,
    LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR, LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR,
    SOURCE_OVERRIDES, EXCLUDED_CONTRACTS,
)
from rl_models.eval_utils import compute_paper_metrics

# ── Parameters ────────────────────────────────────────────────────────────────
TEST_START      = '2011-01-01'
TEST_END        = '2019-12-31'
SIGMA_TGT       = 0.063          # per-contract vol target
PORT_VOL_TARGET = 0.97           # Table 2 portfolio-level vol target
TRADING_DAYS    = 252

ASSET_CLASSES = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']
ASSET_PATH_MAP = {
    'Commodity': 'Commodity',
    'Equity Index': 'Equity_Index',
    'Fixed Income': 'Fixed_Income',
    'Forex': 'Forex',
}

COLS = ['E(R)', 'Std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +Ret', 'Ave.P/Ave.L']

# ── Helper: portfolio daily returns from raw_data ──────────────────────────────
def get_port_returns(asset_name, strat, overrides, excluded, port_vol=None):
    raw = load_contracts(
        asset_name,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=list(excluded),
        source_overrides=overrides,
    )
    if not raw:
        return None
    R = compute_portfolio_returns(raw, strat, SIGMA_TGT)
    if port_vol is not None:
        R = apply_portfolio_vol_scaling(R, port_vol)
    return R


def get_port_returns_all(strat, overrides, excluded, port_vol=None):
    """All = equal-weight mean of all 4 asset classes pooled."""
    all_raw = []
    for ac in ASSET_CLASSES:
        raw = load_contracts(
            ac,
            test_start=TEST_START,
            test_end=TEST_END,
            excluded_contracts=list(excluded),
            source_overrides=overrides,
        )
        all_raw.extend(raw)
    if not all_raw:
        return None
    R = compute_portfolio_returns(all_raw, strat, SIGMA_TGT)
    if port_vol is not None:
        R = apply_portfolio_vol_scaling(R, port_vol)
    return R


# ── Helper: metrics dict from array ───────────────────────────────────────────
def metrics_from_array(R):
    if R is None or len(R) == 0:
        return {c: float('nan') for c in COLS}
    m = compute_paper_metrics(pd.Series(R))
    return {c: m.get(c, float('nan')) for c in COLS}


# ── DQN: load from ensemble_table2 npz (already scaled to 0.97) ───────────────
def get_dqn_series(asset_name, port_vol=None):
    from vol_scaling import scale_portfolio
    path_name = ASSET_PATH_MAP[asset_name]
    npz_path  = ROOT / f'drl/dqn/reports/ensemble_table2/{path_name}/top5_ensemble_R.npz'
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    # portfolio_returns are already the daily return series at port_vol=0.97
    raw = data['portfolio_returns'].astype(float)
    if port_vol is not None:
        return scale_portfolio(raw, target_std=port_vol)
    return raw


def get_dqn_series_all(port_vol=None):
    """DQN All = equal-weight average of the 4 asset-class series."""
    series_list = []
    for ac in ASSET_CLASSES:
        s = get_dqn_series(ac, port_vol=None)  # load unscaled first
        if s is not None:
            series_list.append(pd.Series(s))
    if not series_list:
        return None
    # align on positional index (they share the same test window)
    min_len = min(len(s) for s in series_list)
    avg = np.mean([s.values[:min_len] for s in series_list], axis=0)
    if port_vol is not None:
        from vol_scaling import scale_portfolio
        avg = scale_portfolio(avg, target_std=port_vol)
    return avg


# ── A2C: load from a2c_results_wide.csv ───────────────────────────────────────
_A2C_COL_MAP = {
    'Commodity': 'Commodity',
    'Equity Index': 'Equity Index',
    'Fixed Income': 'Fixed Income',
    'Forex': 'Forex',   # CSV uses 'Forex'
    'All': 'All',
}

def _load_a2c_wide():
    csv_path = ROOT / 'rl_models/a2c_results_wide.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def get_a2c_returns(asset_name, port_vol=None, df_wide=None):
    if df_wide is None:
        df_wide = _load_a2c_wide()
    if df_wide is None:
        return None
    col = _A2C_COL_MAP.get(asset_name)
    if col not in df_wide.columns:
        return None
    wealth = df_wide[col].dropna()
    if len(wealth) < 10:
        return None
    # cum_wealth → daily returns
    R = wealth.diff().dropna().values
    if port_vol is not None:
        from vol_scaling import scale_portfolio
        R = scale_portfolio(R, target_std=port_vol)
    return R


# ── Build one table (dict of dicts) ───────────────────────────────────────────
def build_table(port_vol=None):
    """
    port_vol=0.97  → Table 2 (portfolio-level vol targeting)
    port_vol=None  → Table 3 (raw signal)
    """
    tag = f"Table 2 (port_vol={port_vol})" if port_vol else "Table 3 (raw signal)"
    print(f"\n{'='*60}\nBuilding {tag}\n{'='*60}")

    df_a2c = _load_a2c_wide()

    # strategy → overrides/excluded
    strat_cfg = {
        'Long':    (LEGACY_EXPERIMENTAL_OVERRIDES_LONG,  LEGACY_EXPERIMENTAL_EXCLUDED_LONG),
        'Sign(R)': (LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR, LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR),
        'MACD':    (LEGACY_EXPERIMENTAL_OVERRIDES_MACD,  LEGACY_EXPERIMENTAL_EXCLUDED_MACD),
    }

    results = {}  # asset → strategy → metrics_dict

    all_assets = ASSET_CLASSES + ['All']
    for asset in all_assets:
        results[asset] = {}
        print(f"  {asset}:")

        # Long / Sign(R) / MACD
        for strat, (ov, ex) in strat_cfg.items():
            if asset == 'All':
                R = get_port_returns_all(strat, ov, ex, port_vol)
            else:
                R = get_port_returns(asset, strat, ov, ex, port_vol)
            results[asset][strat] = metrics_from_array(R)
            sharpe = results[asset][strat]['Sharpe']
            print(f"    {strat:8s} Sharpe={sharpe:+.3f}")

        # DQN
        if asset == 'All':
            R_dqn = get_dqn_series_all(port_vol)
        else:
            R_dqn = get_dqn_series(asset, port_vol)
        results[asset]['DQN'] = metrics_from_array(R_dqn)
        print(f"    {'DQN':8s} Sharpe={results[asset]['DQN']['Sharpe']:+.3f}")

        # A2C
        R_a2c = get_a2c_returns(asset, port_vol, df_a2c)
        results[asset]['A2C'] = metrics_from_array(R_a2c)
        print(f"    {'A2C':8s} Sharpe={results[asset]['A2C']['Sharpe']:+.3f}")

    return results


# ── Format as Markdown table ───────────────────────────────────────────────────
STRAT_ORDER = ['Long', 'Sign(R)', 'MACD', 'DQN', 'A2C']

def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.3f}'


def render_md_table(results, title):
    lines = []
    lines.append(f'## {title}')
    lines.append('')

    # header
    header = '| Strategy | ' + ' | '.join(COLS) + ' |'
    sep    = '|---|' + '|'.join(['---:'] * len(COLS)) + '|'
    lines.append(header)
    lines.append(sep)

    asset_display = {
        'Commodity': 'Commodity',
        'Equity Index': 'Equity Index',
        'Fixed Income': 'Fixed Income',
        'Forex': 'FX',
        'All': 'All',
    }

    for asset in ASSET_CLASSES + ['All']:
        lines.append(f'| **{asset_display[asset]}** |{"| ".join([""] * len(COLS))}|')
        for strat in STRAT_ORDER:
            m = results.get(asset, {}).get(strat, {})
            vals = ' | '.join(_fmt(m.get(c)) for c in COLS)
            lines.append(f'| {strat} | {vals} |')
        lines.append('| |' + ' |'.join([''] * len(COLS)) + '|')

    return '\n'.join(lines)


def render_paper_style_md(results, title, note=''):
    """Grouped layout matching the paper's visual style."""
    lines = []
    lines.append(f'## {title}')
    if note:
        lines.append(f'> {note}')
    lines.append('')

    col_header = ' | '.join(f'**{c}**' for c in COLS)
    lines.append(f'|  | {col_header} |')
    lines.append('|---|' + '|'.join(['---:'] * len(COLS)) + '|')

    asset_display = {
        'Commodity': 'Commodity',
        'Equity Index': 'Equity Index',
        'Fixed Income': 'Fixed Income',
        'Forex': 'FX',
        'All': 'All',
    }

    for asset in ASSET_CLASSES + ['All']:
        disp = asset_display[asset]
        lines.append(f'| ***{disp}*** | {" | ".join([""] * len(COLS))} |')
        for strat in STRAT_ORDER:
            m = results.get(asset, {}).get(strat, {})
            vals = ' | '.join(_fmt(m.get(c)) for c in COLS)
            lines.append(f'| {strat} | {vals} |')

    return '\n'.join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Generating Table 2 and Table 3 metrics for all strategies...")

    table2_results = build_table(port_vol=PORT_VOL_TARGET)
    table3_results = build_table(port_vol=None)

    md_table2 = render_paper_style_md(
        table2_results,
        'Table 2: Experiment results for the portfolio-level volatility targeting',
        note='Additional portfolio-level vol scaling applied (target σ = 0.97 annualised).'
    )
    md_table3 = render_paper_style_md(
        table3_results,
        'Table 3: Experiment Results for the Raw Signal',
        note='No additional portfolio-level vol scaling (raw per-contract signal only).'
    )

    output_dir = ROOT / '伦钰婷做PPT要用的素材'
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / 'results_tables.md'

    content = f"""# Strategy Performance Tables
*Generated from reproduced models — test period {TEST_START} to {TEST_END}*

Strategies included: **Long Only**, **Sign(R)**, **MACD**, **DQN (Top-5 Ensemble)**, **A2C**

Metrics:
- **E(R)**: Annualised mean return
- **Std(R)**: Annualised standard deviation
- **DD**: Annualised downside deviation (std of negative returns)
- **Sharpe**: E(R) / Std(R)
- **Sortino**: E(R) / DD
- **MDD**: Maximum drawdown
- **Calmar**: E(R) / MDD
- **% +Ret**: Fraction of days with positive return
- **Ave.P/Ave.L**: Mean gain / |Mean loss|

---

{md_table2}

---

{md_table3}
"""

    out_path.write_text(content, encoding='utf-8')
    print(f'\n✅ Tables saved to: {out_path}')


if __name__ == '__main__':
    main()
