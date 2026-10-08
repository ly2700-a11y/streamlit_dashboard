#!/usr/bin/env python3
"""
Export all figure data to CSV files.

Usage:
    python drl/dqn/figures/export_figure_data.py
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path

from baseline_run import load_contracts, compute_portfolio_return_series
from drl_models.dqn.figures.figure_data import load_scaled_ensemble_series, scale_return_series, get_ensemble_npz_path
from drl_shared.spec import current_source_policy

# ─── Configuration ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / 'drl' / 'dqn' / 'figures' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']
ASSET_PATH_MAP = {
    'Commodity': 'Commodity',
    'Equity Index': 'Equity_Index',
    'Fixed Income': 'Fixed_Income',
    'Forex': 'Forex'
}
SIGMA_TGT = 0.058
PORT_VOL_TARGET = 0.97
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
R1_R2_BOUNDARY = '2016-01-01'
SOURCE_POLICY = current_source_policy()
BP_LEVEL = None

# Exhibit 4
SEEDS = list(range(42, 52))  # 42-51
PER_CONTRACT_DIR = REPO_ROOT / 'drl' / 'dqn' / 'reports' / 'per_contract'

# Exhibit 5
REPORTS_BP_ROOT = REPO_ROOT / 'drl' / 'dqn' / 'reports' / 'ensemble_table2_bp'


# ─── Shared helpers (same as figure scripts) ──────────────────────────────────

def load_dqn_ensemble_returns(asset_name):
    npz_path = get_ensemble_npz_path(asset_name, bp=BP_LEVEL)
    series = load_scaled_ensemble_series(npz_path, PORT_VOL_TARGET)
    return series.index, series.values


def compute_long_only_returns(asset_name):
    raw_data = load_contracts(
        asset_name,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=SOURCE_POLICY['excluded_contracts'],
        source_overrides=SOURCE_POLICY['source_overrides'],
    )
    series = compute_portfolio_return_series(raw_data, 'Long', SIGMA_TGT)
    series = scale_return_series(series, PORT_VOL_TARGET)
    return series.index, series.values


def compute_rolling_sharpe(returns, window=252):
    returns_series = pd.Series(returns)
    rolling_mean = returns_series.rolling(window=window, min_periods=window).mean()
    rolling_std = returns_series.rolling(window=window, min_periods=window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    return rolling_sharpe.values


def compute_drawdown(returns, initial_wealth):
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.asarray([], dtype=float)
    wealth = float(initial_wealth) + np.cumsum(values)
    running_peak = np.maximum.accumulate(wealth)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (wealth - running_peak) / running_peak * 100
    return np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)


def asset_contract_count(asset_name):
    raw_data = load_contracts(
        asset_name,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=SOURCE_POLICY['excluded_contracts'],
        source_overrides=SOURCE_POLICY['source_overrides'],
    )
    return len(raw_data)


# ─── Figure 1: Cumulative Returns ─────────────────────────────────────────────

def export_paper_figure1():
    bp_suffix = f"_bp{int(BP_LEVEL * 10000)}" if BP_LEVEL else ""
    print(f"Exporting paper_figure1_data{bp_suffix}.csv ...")
    records = []
    for asset in ASSETS:
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_cum = np.cumsum(dqn_returns)

        long_dates, long_returns = compute_long_only_returns(asset)
        long_cum = np.cumsum(long_returns)

        dqn_dt = pd.to_datetime(dqn_dates)
        long_dt = pd.to_datetime(long_dates)
        long_series = pd.Series(long_cum, index=long_dt)

        asset_records = []
        for i, (dt, dqn_val) in enumerate(zip(dqn_dt, dqn_cum)):
            long_val = long_series.loc[dt] if dt in long_series.index else np.nan
            row = {
                'date': dt.strftime('%Y-%m-%d'),
                'asset': asset,
                'DQN_cum_return': round(dqn_val, 6),
                'Long_cum_return': round(long_val, 6),
            }
            records.append(row)
            asset_records.append(row)

        # Per-asset CSV (match existing naming: "equity index" with space)
        asset_slug = asset.lower()
        per_csv = DATA_DIR / f'paper_figure1_{asset_slug}{bp_suffix}.csv'
        pd.DataFrame(asset_records).to_csv(per_csv, index=False)
        print(f"  {asset}: {len(asset_records)} rows → {per_csv.name}")

    df = pd.DataFrame(records)
    out_path = DATA_DIR / f'paper_figure1_data{bp_suffix}.csv'
    df.to_csv(out_path, index=False)
    print(f"  Combined: {len(df)} rows → {out_path.name}")
    return df


# ─── Exhibit 4: Per-Contract Sharpe ────────────────────────────────────────────

def compute_annualized_sharpe(returns):
    if len(returns) == 0 or np.std(returns) == 0:
        return np.nan
    mean_daily = np.mean(returns)
    std_daily = np.std(returns)
    return mean_daily * 252 / (std_daily * np.sqrt(252))


def export_exhibit4():
    print("Exporting exhibit4_data.csv ...")
    records = []

    for asset in ASSETS:
        asset_slug = ASSET_PATH_MAP[asset]
        asset_dir = PER_CONTRACT_DIR / asset_slug

        if not asset_dir.exists():
            print(f"  Warning: {asset_dir} does not exist, skipping")
            continue

        # Get contracts for this asset to know the ticker list
        raw_data = load_contracts(
            asset,
            test_start=TEST_START,
            test_end=TEST_END,
            excluded_contracts=SOURCE_POLICY['excluded_contracts'],
            source_overrides=SOURCE_POLICY['source_overrides'],
        )
        ticker_set = {rd['tk'] for rd in raw_data}

        for seed in SEEDS:
            for rd in raw_data:
                ticker = rd['tk']
                cache_path = asset_dir / f"{ticker}_s{seed}.npz"

                if not cache_path.exists():
                    continue

                try:
                    data = np.load(cache_path, allow_pickle=True)
                    returns = data['returns']
                    sharpe = compute_annualized_sharpe(returns)
                    records.append({
                        'ticker': ticker,
                        'asset': asset,
                        'seed': seed,
                        'sharpe': round(sharpe, 6) if np.isfinite(sharpe) else np.nan,
                    })
                except Exception as e:
                    print(f"  Error loading {cache_path}: {e}")
                    continue

    df = pd.DataFrame(records)
    out_path = DATA_DIR / 'exhibit4_data.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


# ─── Supplementary: Rolling Sharpe ─────────────────────────────────────────────

def export_supp_rolling_sharpe():
    print("Exporting supp_rolling_sharpe_data.csv ...")
    records = []
    for asset in ASSETS:
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_dt = pd.to_datetime(dqn_dates)
        dqn_sharpe = compute_rolling_sharpe(dqn_returns)

        long_dates, long_returns = compute_long_only_returns(asset)
        long_dt = pd.to_datetime(long_dates)
        long_sharpe = compute_rolling_sharpe(long_returns)

        # Align on DQN dates
        long_series = pd.Series(long_sharpe, index=long_dt)

        for i, (dt, ds) in enumerate(zip(dqn_dt, dqn_sharpe)):
            ls = long_series.loc[dt] if dt in long_series.index else np.nan
            records.append({
                'date': dt.strftime('%Y-%m-%d'),
                'asset': asset,
                'DQN_sharpe': round(ds, 6) if np.isfinite(ds) else np.nan,
                'Long_sharpe': round(ls, 6) if np.isfinite(ls) else np.nan,
            })

    df = pd.DataFrame(records)
    out_path = DATA_DIR / 'supp_rolling_sharpe_data.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


# ─── Supplementary: Drawdown ───────────────────────────────────────────────────

def export_supp_drawdown():
    print("Exporting supp_drawdown_data.csv ...")
    records = []
    for asset in ASSETS:
        n_contracts = asset_contract_count(asset)
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_dt = pd.to_datetime(dqn_dates)
        dqn_dd = compute_drawdown(dqn_returns, n_contracts)

        long_dates, long_returns = compute_long_only_returns(asset)
        long_dt = pd.to_datetime(long_dates)
        long_dd = compute_drawdown(long_returns, n_contracts)

        # Align on DQN dates
        long_series = pd.Series(long_dd, index=long_dt)

        for i, (dt, dd_val) in enumerate(zip(dqn_dt, dqn_dd)):
            long_val = long_series.loc[dt] if dt in long_series.index else np.nan
            records.append({
                'date': dt.strftime('%Y-%m-%d'),
                'asset': asset,
                'DQN_drawdown': round(dd_val, 6),
                'Long_drawdown': round(long_val, 6) if np.isfinite(long_val) else np.nan,
            })

    df = pd.DataFrame(records)
    out_path = DATA_DIR / 'supp_drawdown_data.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


# ─── Supplementary: Monthly Heatmap ───────────────────────────────────────────

def export_supp_monthly_heatmap():
    print("Exporting supp_monthly_heatmap_data.csv ...")
    records = []
    for asset in ASSETS:
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_series = pd.Series(dqn_returns, index=pd.to_datetime(dqn_dates))
        monthly_dqn = dqn_series.resample('ME').sum()

        long_dates, long_returns = compute_long_only_returns(asset)
        long_series = pd.Series(long_returns, index=pd.to_datetime(long_dates))
        monthly_long = long_series.resample('ME').sum()

        # Align on DQN monthly dates
        for dt, dqn_val in zip(monthly_dqn.index, monthly_dqn.values):
            long_val = monthly_long.loc[dt] if dt in monthly_long.index else np.nan
            records.append({
                'year': dt.year,
                'month': dt.month,
                'asset': asset,
                'DQN_return': round(dqn_val, 6),
                'Long_return': round(long_val, 6) if np.isfinite(long_val) else np.nan,
            })

    df = pd.DataFrame(records)
    out_path = DATA_DIR / 'supp_monthly_heatmap_data.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


# ─── Supplementary: Yearly Bars ────────────────────────────────────────────────

def export_supp_yearly_bars():
    print("Exporting supp_yearly_bars_data.csv ...")

    def annual_sharpe(x):
        if len(x) == 0 or x.std() == 0:
            return np.nan
        return x.mean() / x.std() * np.sqrt(252)

    records = []
    for asset in ASSETS:
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_series = pd.Series(dqn_returns, index=pd.to_datetime(dqn_dates))
        dqn_annual = dqn_series.resample('YE').apply(annual_sharpe)

        long_dates, long_returns = compute_long_only_returns(asset)
        long_series = pd.Series(long_returns, index=pd.to_datetime(long_dates))
        long_annual = long_series.resample('YE').apply(annual_sharpe)

        for dt, dqn_val in zip(dqn_annual.index, dqn_annual.values):
            long_val = long_annual.loc[dt] if dt in long_annual.index else np.nan
            records.append({
                'year': dt.year,
                'asset': asset,
                'DQN_return': round(dqn_val, 6) if np.isfinite(dqn_val) else np.nan,
                'Long_return': round(long_val, 6) if np.isfinite(long_val) else np.nan,
            })

    df = pd.DataFrame(records)
    out_path = DATA_DIR / 'supp_yearly_bars_data.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


# ─── Exhibit 5: BP Impact ──────────────────────────────────────────────────────

def export_exhibit5():
    print("Exporting exhibit5_data.csv ...")
    records = []

    for asset in ASSETS:
        slug = ASSET_PATH_MAP[asset]
        asset_dir = REPORTS_BP_ROOT / slug

        if not asset_dir.exists():
            print(f"  Warning: {asset_dir} does not exist, skipping")
            continue

        for bp_dir in sorted(asset_dir.iterdir()):
            if not bp_dir.is_dir() or not bp_dir.name.startswith('bp'):
                continue

            try:
                bp_bps = int(bp_dir.name[2:])
            except ValueError:
                continue

            metrics_path = bp_dir / 'metrics.json'
            if not metrics_path.exists():
                continue

            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"  Error loading {metrics_path}: {e}")
                continue

            sharpe = None
            if isinstance(metrics, dict):
                m = metrics.get('metrics', metrics)
                if isinstance(m, dict):
                    s = m.get('Sharpe')
                    if s is not None:
                        try:
                            sharpe = float(s)
                        except (TypeError, ValueError):
                            sharpe = None

            avg_daily_cost = None
            daily_costs_path = bp_dir / 'daily_costs.npz'
            if daily_costs_path.exists():
                try:
                    cost_data = np.load(daily_costs_path, allow_pickle=True)
                    if 'avg_daily_cost' in cost_data:
                        avg_daily_cost = float(cost_data['avg_daily_cost'])
                    elif 'daily_costs' in cost_data:
                        dc = cost_data['daily_costs']
                        if len(dc) > 0:
                            avg_daily_cost = float(np.mean(dc))
                except Exception:
                    pass

            records.append({
                'bp_level': bp_bps,
                'asset': asset,
                'sharpe': round(sharpe, 6) if sharpe is not None and np.isfinite(sharpe) else np.nan,
                'avg_daily_cost': round(avg_daily_cost, 6) if avg_daily_cost is not None else np.nan,
            })

    df = pd.DataFrame(records)
    out_path = DATA_DIR / 'exhibit5_data.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Exporting all figure data to CSV")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 60)
    print()

    export_paper_figure1()
    print()
    export_exhibit4()
    print()
    export_supp_rolling_sharpe()
    print()
    export_supp_drawdown()
    print()
    export_supp_monthly_heatmap()
    print()
    export_supp_yearly_bars()
    print()
    export_exhibit5()
    print()

    print("=" * 60)
    print("All CSVs exported successfully!")
    print("=" * 60)


if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc-bp", type=float, default=None)
    args = parser.parse_args()
    sys.modules['__main__'].BP_LEVEL = args.tc_bp
    main()