#!/usr/bin/env python3
"""
baseline_run.py — Table 2 & Table 3 baseline reproduction (single entry point)

Paper: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"

Usage:
    python baseline_run.py                  # Table 3 (all asset classes)
    python baseline_run.py --table 2        # Table 2 (with portfolio vol scaling)
    python baseline_run.py --table both     # Both tables
    python baseline_run.py --asset Forex    # Single asset class
    python baseline_run.py --sigma 0.064    # Custom σ_tgt
    python baseline_run.py --test-start 2015-01-01 --test-end 2019-12-31  # Custom period
    python baseline_run.py --source-mode working --all-metrics # Use unified SOURCE_OVERRIDES from config.py
    python baseline_run.py --source-mode pure-rad --all-metrics # 不用任何 override，所有合约默认走 RAD
    python baseline_run.py --source-mode strategy-legacy --all-metrics # Use legacy overrides for each strategy                                                  
                                                            Long 用 LEGACY_EXPERIMENTAL_OVERRIDES_LONG
                                                            Sign(R) 用 LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR
                                                            MACD 用 LEGACY_EXPERIMENTAL_OVERRIDES_MACD
    
    
    """
import argparse
from functools import lru_cache
import os
import numpy as np
import pandas as pd
from data_loader import load_clc_full
from strategies import strategy_sign_r, strategy_macd
from metrics import (
    cagr_from_path,
    compute_metrics,
    max_drawdown_from_path,
)
from config import (
    ASSET_CLASSES, BP, TRADING_DAYS, SIGN_LOOKBACK,
    PAPER_TABLE2, PAPER_TABLE3, METRIC_NAMES, EXCLUDED_CONTRACTS,
    SOURCE_OVERRIDES,
    LEGACY_EXPERIMENTAL_OVERRIDES_LONG,
    LEGACY_EXPERIMENTAL_EXCLUDED_LONG,
    LEGACY_EXPERIMENTAL_OVERRIDES_MACD,
    LEGACY_EXPERIMENTAL_EXCLUDED_MACD,
    LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR,
    LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR,
)

# Core 5 metrics for summary table
CORE_METRICS = ['E(R)', 'std(R)', 'Sharpe', '% +ve', 'Ave P/L']
CORE_METRIC_IDX = [METRIC_NAMES.index(n) for n in CORE_METRICS]

# ─── Parameters ───────────────────────────────────────────────────
DEFAULT_SIGMA_TGT = 0.0630   # Working frontier after adjusted-only MDD-first v4 re-search
EWMA_SPAN = 60              # EWMA span for σ_t [Paper Section 3.2]
T = TRADING_DAYS            # 252
W0 = 1.0                    # Initial wealth per contract
DEFAULT_REPORT_SOURCE = 'RISK_PRICE_SIGMA0'  # current split-world start: 7 trade metrics + risk/price t0 bridge for MDD/Calmar



# ─── Data Loading ─────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _prepare_contract_cached(ticker, test_start, test_end, source):
    df = load_clc_full(ticker, source=source, anchor_date=test_start)
    if df is None:
        return None
    prices = df['Close'].values.astype(float)
    if len(prices) < 500:
        return None

    rt = np.zeros(len(prices))
    rt[1:] = prices[1:] - prices[:-1]
    sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values

    mask_s = df['Date'] >= test_start
    mask_e = df['Date'] <= test_end
    if not mask_s.any() or not mask_e.any():
        return None
    t0 = mask_s.idxmax()
    t1 = len(df) - 1 - mask_e[::-1].values.argmax()
    start = max(t0, SIGN_LOOKBACK)
    # `t1` is the last in-range index, so include it explicitly.
    dates = df['Date'].iloc[start:t1 + 1].values
    return {
        'tk': ticker,
        'rt': rt,
        'sigma': sigma,
        'prices': prices,
        'start': start,
        't1': t1,
        'dates': dates,
        'source': source,
        'macd_pos': strategy_macd(prices),
    }


def load_contracts(ac_name, test_start='2011-01-01', test_end='2019-12-31',
                   excluded_contracts=None, source_overrides=None):
    """Load and prepare all contracts for an asset class."""
    tickers = ASSET_CLASSES.get(ac_name, [])
    if excluded_contracts is None:
        excluded_contracts = EXCLUDED_CONTRACTS
    if source_overrides is None:
        source_overrides = SOURCE_OVERRIDES
    raw = []
    for tk in tickers:
        if tk in excluded_contracts:
            continue
        source = source_overrides.get(tk, 'RAD')
        prepared = _prepare_contract_cached(tk, test_start, test_end, source)
        if prepared is not None:
            raw.append(prepared)
    return raw


def parse_exclusion_arg(exclude_arg):
    """Parse a comma-separated exclusion override from the CLI."""
    if not exclude_arg:
        return list(EXCLUDED_CONTRACTS)
    tokens = [tk.strip().upper() for tk in exclude_arg.split(',')]
    tokens = [tk for tk in tokens if tk]
    return sorted(set(tokens))


def build_reporting_portfolio_risk_price_sigma0(raw_data, sigma_tgt, strat='Long'):
    """Build the fixed reporting-world sleeve-capital wealth path.

    Each contract sleeve starts with capital:
        C_i,0 = p_i,0 * sigma_tgt / sigma_i,0

    Sleeve wealth then accumulates normalized Eq. 4 rewards:
        w_i,t = 1 + cumsum(R_i,t / C_i,0)

    The reporting portfolio is the equal-weight average of sleeve wealth paths.
    """
    sleeve_paths = []
    for rd in raw_data:
        detail = compute_contract_returns(rd, strat, sigma_tgt, detail=True)
        start, t1 = rd['start'], rd['t1']
        Rt = detail['Rt'][start:t1 + 1]
        prices = detail['prices'][start:t1 + 1]
        sigma = detail['sigma'][start:t1 + 1]
        if len(Rt) == 0 or len(prices) == 0 or len(sigma) == 0:
            continue
        p0 = float(prices[0])
        sigma0 = float(sigma[0])
        if not np.isfinite(p0) or not np.isfinite(sigma0) or sigma0 <= 0:
            continue
        capital0 = p0 * sigma_tgt / sigma0
        if not np.isfinite(capital0) or capital0 <= 0:
            continue
        sleeve_paths.append(1.0 + np.cumsum(Rt / capital0))

    if not sleeve_paths:
        return None

    min_len = min(len(path) for path in sleeve_paths)
    if min_len == 0:
        return None

    sleeves = np.column_stack([path[:min_len] for path in sleeve_paths])
    portfolio = sleeves.mean(axis=1)

    sleeve_simple = np.full_like(sleeves, np.nan, dtype=float)
    if min_len > 1:
        sleeve_simple[1:, :] = sleeves[1:, :] / sleeves[:-1, :] - 1.0

    portfolio_simple = np.full(min_len, np.nan, dtype=float)
    portfolio_log = np.full(min_len, np.nan, dtype=float)
    if min_len > 1:
        portfolio_simple[1:] = portfolio[1:] / portfolio[:-1] - 1.0
        portfolio_log[1:] = np.log(portfolio[1:] / portfolio[:-1])

    return {
        'portfolio_path': portfolio,
        'portfolio_simple_returns': portfolio_simple,
        'portfolio_log_returns': portfolio_log,
        'sleeve_paths': sleeves,
        'sleeve_simple_returns': sleeve_simple,
        'length': min_len,
    }


def compute_reporting_mdd_calmar_risk_price_sigma0(raw_data, sigma_tgt, strat='Long'):
    """
    Reporting-world bridge:

    - keep Eq. 4 trade rewards R_i,t
    - define sleeve initial capital as p_i,0 * sigma_tgt / sigma_i,0
    - accumulate normalized sleeve wealth:
          w_i,t = 1 + cumsum(R_i,t / C_i,0)
    - equal-weight sleeve wealth paths into a portfolio path

    This keeps the trade lane untouched while giving MDD/Calmar a portfolio
    wealth object that uses both initial price and initial risk scale.
    """
    reporting = build_reporting_portfolio_risk_price_sigma0(
        raw_data,
        sigma_tgt=sigma_tgt,
        strat=strat,
    )
    if reporting is None:
        return None
    port = reporting['portfolio_path']
    mdd = max_drawdown_from_path(port)
    cagr = cagr_from_path(port)
    calmar = cagr / mdd if mdd > 0 else 0.0
    return round(mdd, 3), round(calmar, 3), reporting['length']


# ─── Eq 4: Trade Return ──────────────────────────────────────────
def compute_contract_returns(rd, strat, sigma_tgt, detail=False):
    """Compute daily R_t for one contract using Paper Eq 4:

    R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t
        − bp × p_{t-1} × |(σ_tgt/σ_{t-1})×A_{t-1} − (σ_tgt/σ_{t-2})×A_{t-2}|

    Args:
        detail: if True, return dict with Rt + position/scale/TC details
    Returns:
        Rt array (default), or dict with full diagnostic info (detail=True)
    """
    rt, sigma, prices = rd['rt'], rd['sigma'], rd['prices']
    n = len(rt)

    # Position signal A_t
    if strat == 'Long':
        pos = np.ones(n)
    elif strat == 'Sign(R)':
        pos = strategy_sign_r(rt, SIGN_LOOKBACK)
    else:
        pos = rd['macd_pos']

    Rt = np.zeros(n)
    scaled_pos = np.zeros(n)   # A_{t-1} × σ_tgt/σ_{t-1}
    gross_pnl = np.zeros(n)    # sp × r_t  (before TC)
    tc_cost = np.zeros(n)      # bp × p_{t-1} × |Δscaled_pos|

    for t in range(1, n):
        if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
            a_prev = pos[t - 1] if strat != 'Long' else 1.0
            a_prev2 = pos[t - 2] if strat != 'Long' else 1.0
            sp = a_prev * sigma_tgt / sigma[t - 1]
            spp = a_prev2 * sigma_tgt / sigma[t - 2] if t >= 2 else 0.0
            scaled_pos[t] = sp
            gross_pnl[t] = sp * rt[t]
            tc_cost[t] = BP * prices[t - 1] * abs(sp - spp)
            Rt[t] = gross_pnl[t] - tc_cost[t]

    if detail:
        return {'Rt': Rt, 'A_t': pos, 'scaled_pos': scaled_pos,
                'gross_pnl': gross_pnl, 'tc_cost': tc_cost,
                'sigma': sigma, 'prices': prices, 'rt': rt}
    return Rt


# ─── Eq 13: Portfolio Return ─────────────────────────────────────
def compute_portfolio_returns(raw_data, strat, sigma_tgt,
                              aggregation_mode='variable_n'):
    """Eq 13: R_port = (1/N) × Σ R_i  (equal-weight average)."""
    series = []
    for rd in raw_data:
        Rt = compute_contract_returns(rd, strat, sigma_tgt)
        start, t1, dates = rd['start'], rd['t1'], rd['dates']
        slc = Rt[start:t1 + 1]
        series.append(pd.Series(slc[:len(dates)], index=dates[:len(slc)]))
    df_all = pd.DataFrame(series)
    if aggregation_mode == 'dropna':
        port = df_all.T.dropna().mean(axis=1)
    elif aggregation_mode == 'variable_n':
        # Average only over contracts with data on each date. This preserves
        # dates across exchanges with different holiday calendars.
        port = df_all.T.mean(axis=1)
    else:
        raise ValueError(f'Unknown aggregation_mode: {aggregation_mode}')
    return port.values

# ─── Table 2: Portfolio-level vol scaling ─────────────────────────
def apply_portfolio_vol_scaling(R_eq, target_std):
    """Scale R_eq so annualized std = target_std."""
    current_std = np.std(R_eq) * np.sqrt(T)
    if current_std > 0:
        return R_eq * (target_std / current_std)
    return R_eq


# ─── Output ───────────────────────────────────────────────────────
def fmt(vals):
    return "  ".join(f"{v:>+7.3f}" for v in vals)


def run_table(raw_data, ac_name, sigma_tgt, paper_table, table_label,
              port_vol_target=None, metric_names=None,
              aggregation_mode='variable_n',
              report_source=DEFAULT_REPORT_SOURCE,
              test_start='2011-01-01',
              test_end='2019-12-31',
              strategies=None):
    """Run one table (Table 2 or 3) for one asset class."""
    if isinstance(raw_data, dict):
        strategy_counts = {strat: len(rows) for strat, rows in raw_data.items()}
        if not any(strategy_counts.values()):
            return 0, 0, 0
        unique_counts = sorted(set(strategy_counts.values()))
        n_label = str(unique_counts[0]) if len(unique_counts) == 1 else "mixed"
    else:
        N = len(raw_data)
        if N == 0:
            return 0, 0, 0
        n_label = str(N)

    if isinstance(raw_data, dict):
        N = max((len(rows) for rows in raw_data.values()), default=0)
    if N == 0:
        return 0, 0, 0

    if metric_names is None:
        metric_names = METRIC_NAMES

    # Get indices for the metrics we want to display
    all_names = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
                 'MDD', 'Calmar', '% +ve', 'Ave P/L']
    metric_idx = [all_names.index(n) for n in metric_names]
    n_metrics = len(metric_names)

    port_str = f" | port_vol→{port_vol_target}" if port_vol_target else ""
    report_str = f" | report={report_source}" if report_source != 'trade' else ""
    print(f"\n{'=' * 90}")
    print(f"  {table_label} — {ac_name} ({n_label} contracts)")
    print(f"  σ_tgt={sigma_tgt} | EWMA({EWMA_SPAN}) | bp={BP}{port_str}{report_str}")
    print(f"  Metrics: {', '.join(metric_names)}")
    print(f"{'=' * 90}")

    if strategies is None:
        strategies = ['Long', 'Sign(R)', 'MACD']

    available_targets = paper_table.get(ac_name, {})
    total_n10, total_n15, total = 0, 0, 0
    for strat in strategies:
        if strat not in available_targets:
            print(f"\n  {strat:8s} (skipped: no paper target for {ac_name} in {table_label})")
            continue
        strat_raw_data = raw_data.get(strat, []) if isinstance(raw_data, dict) else raw_data
        N = len(strat_raw_data)
        if N == 0:
            print(f"\n  {strat:8s} (skipped: no contracts loaded for {ac_name})")
            continue
        R = compute_portfolio_returns(strat_raw_data, strat, sigma_tgt,
                                      aggregation_mode=aggregation_mode)
        if port_vol_target is not None:
            R = apply_portfolio_vol_scaling(R, port_vol_target)
        m_all = compute_metrics(R, n_contracts=N)
        if report_source != 'trade':
            reporting = compute_reporting_mdd_calmar_risk_price_sigma0(
                strat_raw_data,
                sigma_tgt=sigma_tgt,
                strat=strat,
            )
            if reporting is not None:
                mdd_rep, calmar_rep, _ = reporting
                m_all[all_names.index('MDD')] = mdd_rep
                m_all[all_names.index('Calmar')] = calmar_rep
        # Extract only the metrics we care about
        m = [m_all[i] for i in metric_idx]
        pv_dict = available_targets[strat]
        pv = [pv_dict[k] for k in metric_names]
        errs = [abs((m[i] - pv[i]) / abs(pv[i])) * 100 if pv[i] != 0 else 0
                for i in range(n_metrics)]
        n10 = sum(1 for e in errs if e < 10)
        n15 = sum(1 for e in errs if e < 15)
        total_n10 += n10
        total_n15 += n15
        total += n_metrics

        print(f"\n  {strat:8s} (≤10%:{n10}/{n_metrics}  ≤15%:{n15}/{n_metrics})")
        print(f"  Ours  : {fmt(m)}")
        print(f"  Paper : {fmt(pv)}")
        print(f"  %Err  : {'  '.join(f'{e:>6.1f}%' for e in errs)}")

    return total_n10, total_n15, total


# ─── Main ─────────────────────────────────────────────────────────
def run_diagnostics(raw_data, ac_name, sigma_tgt, test_start='2011-01-01', test_end='2019-12-31'):
    """Output full portfolio diagnostic with per-contract positions.
    Saves a CSV with daily scaled_pos, gross_pnl, tc_cost for each contract
    plus portfolio-level R_port and cumsum.
    """
    all_details = {}
    for rd in raw_data:
        tk = rd['tk']
        det = compute_contract_returns(rd, 'Long', sigma_tgt, detail=True)
        start, t1, dates = rd['start'], rd['t1'], rd['dates']
        slc = slice(start, t1 + 1)
        df = pd.DataFrame({
            f'{tk}_price': det['prices'][slc],
            f'{tk}_rt': det['rt'][slc],
            f'{tk}_sigma': det['sigma'][slc],
            f'{tk}_scaled_pos': det['scaled_pos'][slc],
            f'{tk}_gross_pnl': det['gross_pnl'][slc],
            f'{tk}_tc': det['tc_cost'][slc],
            f'{tk}_Rt': det['Rt'][slc],
        }, index=dates[:t1 - start + 1])
        all_details[tk] = df

    # Merge all contracts on date index
    merged = pd.concat(all_details.values(), axis=1)

    # Portfolio-level columns
    rt_cols = [c for c in merged.columns if c.endswith('_Rt')]
    pos_cols = [c for c in merged.columns if c.endswith('_scaled_pos')]
    pnl_cols = [c for c in merged.columns if c.endswith('_gross_pnl')]
    tc_cols = [c for c in merged.columns if c.endswith('_tc')]

    merged['port_Rt'] = merged[rt_cols].mean(axis=1)
    merged['port_gross_pnl'] = merged[pnl_cols].mean(axis=1)
    merged['port_tc'] = merged[tc_cols].mean(axis=1)
    merged['port_cumsum'] = merged['port_Rt'].cumsum()
    merged['N_contracts'] = merged[rt_cols].notna().sum(axis=1)
    # Per-contract mean stats
    merged['mean_scaled_pos'] = merged[pos_cols].mean(axis=1)
    merged['port_daily_std'] = merged['port_Rt'].rolling(60).std()

    outdir = f'diagnostics/{ac_name.replace(" ", "_")}'
    os.makedirs(outdir, exist_ok=True)
    fname = f'{outdir}/portfolio_diagnostic_{sigma_tgt}.csv'
    merged.to_csv(fname)
    print(f'  → Saved {fname} ({len(merged)} rows × {len(merged.columns)} cols)')

    # Summary
    Rt = merged['port_Rt'].dropna().values
    if len(Rt) > 100:
        er = np.mean(Rt) * T
        std = np.std(Rt) * np.sqrt(T)
        print(f'  Summary: E(R)={er:+.3f}  std={std:.3f}  Sharpe={er/std:.3f}')
        print(f'  Cumsum range: [{merged["port_cumsum"].min():.2f}, {merged["port_cumsum"].max():.2f}]')
        print(f'  Mean scaled_pos: {merged["mean_scaled_pos"].mean():.4f}')
        print(f'  Mean TC/contract: {merged["port_tc"].mean():.6f}')
        print(f'  TC as % of gross: {abs(merged["port_tc"].mean()) / abs(merged["port_gross_pnl"].mean()) * 100:.2f}%')
    return fname


def main():
    parser = argparse.ArgumentParser(description='Baseline reproduction')
    parser.add_argument('--table', choices=['2', '3', 'both'], default='3',
                        help='Which table to run (default: 3)')
    parser.add_argument('--asset', default=None,
                        help='Single asset class (e.g. "Equity Index")')
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA_TGT,
                        help=f'σ_tgt per contract (default: {DEFAULT_SIGMA_TGT})')
    parser.add_argument('--test-start', default='2011-01-01')
    parser.add_argument('--test-end', default='2019-12-31')
    parser.add_argument('--port-vol-target', type=float, default=0.97,
                        help='Portfolio vol target for Table 2 (default: 0.97)')
    parser.add_argument('--all-metrics', action='store_true',
                        help='Show all 9 metrics (default: 5 core metrics)')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Save full portfolio diagnostic CSV with positions')
    parser.add_argument('--aggregation', choices=['variable_n', 'dropna'],
                        default='variable_n',
                        help='Portfolio aggregation mode (default: variable_n)')
    parser.add_argument('--report-source',
                        choices=['trade', 'RISK_PRICE_SIGMA0'],
                        default=DEFAULT_REPORT_SOURCE,
                        help='Reporting source / bridge for MDD and Calmar only')
    parser.add_argument('--source-mode', choices=['working', 'pure-rad', 'strategy-legacy'],
                        default='working',
                        help='Data source policy: current working overrides, pure RAD only, or strategy-specific legacy overrides (default: working)')
    parser.add_argument('--strategies', default='Long,Sign(R),MACD',
                        help='Comma-separated strategies to run (default: Long,Sign(R),MACD)')
    parser.add_argument('--exclude-contracts', default=None,
                        help='Comma-separated exclusion override, e.g. FB,ZA,ZO')
    args = parser.parse_args()

    excluded_contracts = parse_exclusion_arg(args.exclude_contracts)
    strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]

    strategy_config_map = {
        'Long': (LEGACY_EXPERIMENTAL_OVERRIDES_LONG, LEGACY_EXPERIMENTAL_EXCLUDED_LONG),
        'Sign(R)': (LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR, LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR),
        'MACD': (LEGACY_EXPERIMENTAL_OVERRIDES_MACD, LEGACY_EXPERIMENTAL_EXCLUDED_MACD),
    }

    if args.source_mode == 'working':
        source_overrides = SOURCE_OVERRIDES
    else:
        source_overrides = {}

    asset_classes = [args.asset] if args.asset else [
        'Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All'
    ]

    # Select metric set
    if args.all_metrics:
        metric_names = list(METRIC_NAMES)  # all 9
    else:
        metric_names = CORE_METRICS  # 5 core
    print(f"Using {'ALL 9' if args.all_metrics else 'CORE 5'} metrics: {metric_names}")
    print(f"Using strategies: {strategies}")
    print(f"Using source mode: {args.source_mode}")

    tables = []
    if args.table in ('3', 'both'):
        tables.append(('Table 3', PAPER_TABLE3, None))
    if args.table in ('2', 'both'):
        tables.append(('Table 2', PAPER_TABLE2, args.port_vol_target))

    grand_n10, grand_n15, grand_total = 0, 0, 0

    for table_label, paper_table, port_vol in tables:
        for ac in asset_classes:
            if args.source_mode == 'strategy-legacy':
                raw = {}
                for strat in strategies:
                    strat_overrides, strat_excluded = strategy_config_map.get(strat, ({}, []))
                    if ac == 'All':
                        strat_raw = []
                        for a in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
                            strat_raw.extend(load_contracts(
                                a,
                                args.test_start,
                                args.test_end,
                                excluded_contracts=list(strat_excluded),
                                source_overrides=strat_overrides,
                            ))
                    else:
                        strat_raw = load_contracts(
                            ac,
                            args.test_start,
                            args.test_end,
                            excluded_contracts=list(strat_excluded),
                            source_overrides=strat_overrides,
                        )
                    raw[strat] = strat_raw
            elif ac == 'All':
                # All = combine all asset classes (excluding EXCLUDED_CONTRACTS)
                raw = []
                for a in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
                    raw.extend(load_contracts(
                        a,
                        args.test_start,
                        args.test_end,
                        excluded_contracts=excluded_contracts,
                        source_overrides=source_overrides,
                    ))
            else:
                raw = load_contracts(
                    ac,
                    args.test_start,
                    args.test_end,
                    excluded_contracts=excluded_contracts,
                    source_overrides=source_overrides,
                )
            n10, n15, tot = run_table(raw, ac, args.sigma, paper_table,
                                      table_label, port_vol,
                                      metric_names=metric_names,
                                      aggregation_mode=args.aggregation,
                                      report_source=args.report_source,
                                      test_start=args.test_start,
                                      test_end=args.test_end,
                                      strategies=strategies)
            if args.diagnostic and port_vol is None:  # only Table 3
                run_diagnostics(raw, ac, args.sigma, args.test_start, args.test_end)
            grand_n10 += n10
            grand_n15 += n15
            grand_total += tot

    if grand_total > 0:
        print(f"\n{'=' * 60}")
        print(f"  GRAND TOTAL: ≤10%: {grand_n10}/{grand_total}"
              f" | ≤15%: {grand_n15}/{grand_total}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
