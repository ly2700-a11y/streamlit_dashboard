#!/usr/bin/env python3
"""
generate_ensemble_table2.py — Top-5 Q-value ensemble backtest for Table 2 metrics

Generates ensemble portfolio returns using Q-value averaging (NOT majority voting)
and computes Table 2 metrics with port_vol_target=0.97.

"All" portfolio: equal-weighted across ALL contracts (not equal-weighted by asset).
Formula: (1/N) * sum(R_t^i) where N = total contracts across all 4 asset classes.
Implemented as contract-count-weighted average of per-asset portfolios:
    all_port = (sum(n_asset * portfolio_asset)) / total_contracts

Usage:
    python drl/dqn/reports/generate_ensemble_table2.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import BP, PAPER_TABLE2, METRIC_NAMES
from baseline_run import load_contracts, compute_contract_returns_from_positions
from drl_models.dqn.model import DQNAgent
from drl_models.dqn.spec import RETRAIN_ROUNDS, SIGMA_TGT, WARMUP, contract_data_path
from drl_shared.spec import SEQ_LEN, SIGMA_TGT_DEFAULT, current_source_policy, universe_tickers
from drl_shared.state_space import action_id_to_position, get_feature_window
from vol_scaling import get_portfolio_bridge
from metrics import compute_metrics

# Model root directory
MODEL_ROOT = ROOT / "drl" / "dqn" / "models"
REPORTS_ROOT = ROOT / "drl" / "dqn" / "reports" / "ensemble_table2_bp"

TC_BP_LEVELS = [0.0020, 0.0010, 0.0005]

# Top-5 seeds by validation ranking from earlier analysis
# Format: asset -> {r1: [seeds], r2: [seeds]}
TOP5_SEEDS = {
    "Commodity": {
        "r1": [42, 43, 44, 45, 46],  # Top 5 from r1 validation
        "r2": [42, 43, 44, 45, 46],  # Top 5 from r2 validation
    },
    "Equity Index": {
        "r1": [42, 43, 44, 45, 46],
        "r2": [42, 43, 44, 45, 46],
    },
    "Fixed Income": {
        "r1": [42, 43, 44, 45, 46],
        "r2": [42, 43, 44, 45, 46],
    },
    "Forex": {
        "r1": [42, 43, 44, 45, 46],
        "r2": [42, 43, 44, 45, 46],
    },
}

SOURCE_POLICY = current_source_policy()
SOURCE_OVERRIDES = dict(SOURCE_POLICY["source_overrides"])
EXCLUDED_CONTRACTS = set(SOURCE_POLICY["excluded_contracts"])


def find_latest_bundle(asset_slug: str, round_name: str, seed: int, bp: float | None = None) -> Path | None:
    """Find the latest model bundle for a given asset, round, and seed.

    Args:
        asset_slug: Asset identifier (e.g., 'Commodity')
        round_name: Round identifier (e.g., 'r1')
        seed: Random seed
        bp: If specified, only match bundles with this bp prefix (e.g., 0.0020 -> 'bp20_')
           If None, match any bundle ending with _s{seed}
    """
    asset_dir = MODEL_ROOT / asset_slug / round_name
    if not asset_dir.exists():
        return None

    # Find all bundles for this seed
    seed_bundles = []
    for child in asset_dir.iterdir():
        if not child.is_dir():
            continue
        # Must end with _s{seed}
        if f"_s{seed}" not in child.name:
            continue
        # BP filter: if bp specified, must have bp{XX}_ prefix
        if bp is not None:
            bp_prefix = f"bp{int(bp * 10000)}_"
            if not child.name.startswith(bp_prefix):
                continue
        checkpoint = child / "checkpoint.pt"
        manifest = child / "manifest.json"
        if checkpoint.exists():
            seed_bundles.append(child)

    if not seed_bundles:
        return None

    # Sort by name (timestamp) and return the latest
    return sorted(seed_bundles)[-1]


def load_model_from_bundle(bundle_dir: Path, device: str = "cpu") -> DQNAgent | None:
    """Load a DQN model from a bundle directory."""
    checkpoint_path = bundle_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        return None
    
    try:
        agent = DQNAgent(device=device)
        agent.load(checkpoint_path)
        agent.q_net.eval()
        return agent
    except Exception as e:
        print(f"  Warning: Failed to load model from {bundle_dir}: {e}")
        return None


def find_available_seeds(asset_slug: str, round_name: str, seed_range: range, bp: float | None = None) -> list[int]:
    """Find which seeds have available bundles for a given asset/round."""
    asset_dir = MODEL_ROOT / asset_slug / round_name
    if not asset_dir.exists():
        return []

    available = []
    for seed in seed_range:
        bundle = find_latest_bundle(asset_slug, round_name, seed, bp=bp)
        if bundle is not None:
            available.append(seed)
    return available


def load_ensemble_models(asset_name: str, round_num: int, device: str = "cpu", bp: float | None = None) -> tuple[list[DQNAgent], float]:
    """Load the top-5 models for an asset and round.

    Returns:
        (models, bp) where bp is read from the first model's manifest.json
    """
    asset_slug = asset_name.replace(" ", "_")
    round_name = f"r{round_num}"

    print(f"  Looking for available models for {asset_name} {round_name}...")
    available_seeds = find_available_seeds(asset_slug, round_name, range(42, 52), bp=bp)
    print(f"    Available seeds: {available_seeds}")

    if len(available_seeds) < 5:
        print(f"    Warning: Only {len(available_seeds)} seeds available (need 5)")
        seeds = available_seeds
    else:
        seeds = available_seeds[:5]

    if len(seeds) == 0:
        print(f"    ERROR: No bundles found for {asset_name} {round_name}")
        return [], bp

    print(f"  Loading {len(seeds)} models for {asset_name} {round_name} (seeds: {seeds})...")

    models = []
    for seed in seeds:
        bundle = find_latest_bundle(asset_slug, round_name, seed, bp=bp)
        if bundle is None:
            print(f"    Warning: No bundle found for seed {seed}")
            continue

        manifest_path = bundle / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            bp_manifest = manifest.get("reward_spec", {}).get("bp")
            if bp_manifest is not None:
                bp = bp_manifest

        agent = load_model_from_bundle(bundle, device=device)
        if agent is not None:
            models.append(agent)
            print(f"    ✓ Seed {seed}: {bundle.name}")

    return models, bp


def ensemble_q_value_predict(models: list[DQNAgent], states: np.ndarray) -> np.ndarray:
    """
    Q-value ensemble: average Q-values from all models, then argmax.
    
    Returns action_ids (0, 1, 2) for each state.
    """
    if len(models) == 0:
        raise ValueError("No models loaded for ensemble")
    
    if len(states) == 0:
        return np.array([], dtype=np.int64)
    
    all_q_values = []
    
    for model in models:
        model.q_net.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(model.device)
            q_values = model.q_net(tensor).cpu().numpy()  # (N, 3)
            all_q_values.append(q_values)
    
    # Average Q-values across all models
    mean_q = np.mean(all_q_values, axis=0)  # (N, 3)
    
    # Argmax to get action_ids
    action_ids = mean_q.argmax(axis=1).astype(np.int64)
    
    return action_ids


def backtest_contract_ensemble(
    ticker: str,
    round_num: int,
    models: list[DQNAgent],
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    bp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Backtest a single contract using Q-ensemble.
    
    Returns:
        (returns, dates, positions) or None if failed
    """
    # Load contract data
    source = SOURCE_OVERRIDES.get(ticker, 'RAD')
    npz_path = contract_data_path(round_num, ticker)
    
    if not npz_path.exists():
        print(f"    Warning: No feature data for {ticker} r{round_num}")
        return None
    
    # Load NPZ data
    npz_data = np.load(npz_path, allow_pickle=True)
    features = npz_data["features"]
    dates = npz_data["dates"]
    prices = npz_data["prices"]
    
    # Load raw data for returns/sigma calculation
    from data_loader import load_clc_full
    round_info = RETRAIN_ROUNDS[round_num]
    df = load_clc_full(ticker, source=source, anchor_date=round_info["test_start"])
    
    if df is None:
        print(f"    Warning: Could not load raw data for {ticker}")
        return None
    
    # Compute returns and sigma
    prices_full = df["Close"].values.astype(float)
    returns = np.zeros(len(prices_full))
    returns[1:] = prices_full[1:] - prices_full[:-1]
    sigma = pd.Series(returns).ewm(span=60, adjust=False).std().values
    
    # Get test period mask
    test_mask = (df["Date"] >= round_info["test_start"]) & (df["Date"] <= round_info["test_end"])
    test_indices = np.where(test_mask.values)[0]
    
    if len(test_indices) == 0:
        print(f"    Warning: No test data for {ticker}")
        return None
    
    # Align dates between NPZ and raw data
    npz_dates = pd.to_datetime(dates)
    raw_dates = pd.to_datetime(df["Date"].values)
    
    # Build positions array (full length of raw data)
    positions = np.zeros(len(prices_full), dtype=float)
    
    # For each test date, get ensemble prediction
    states_list = []
    position_indices = []
    
    for raw_idx in test_indices:
        if raw_idx < WARMUP:
            continue
        
        # Find corresponding NPZ index
        date = raw_dates[raw_idx]
        npz_mask = npz_dates == date
        if not npz_mask.any():
            continue
        
        npz_idx = np.where(npz_mask)[0][0]
        
        # Get feature window
        if npz_idx < SEQ_LEN:
            continue
        
        state = get_feature_window(features, npz_idx)
        states_list.append(state)
        position_indices.append(raw_idx)
    
    if len(states_list) == 0:
        print(f"    Warning: No valid states for {ticker}")
        return None
    
    # Batch predict using ensemble
    states_array = np.stack(states_list).astype(np.float32)
    action_ids = ensemble_q_value_predict(models, states_array)
    
    # Convert action_ids to positions (-1, 0, 1)
    for i, raw_idx in enumerate(position_indices):
        positions[raw_idx] = action_id_to_position(int(action_ids[i]))
    
    # Compute Eq.4 returns
    prepared = {
        "tk": ticker,
        "rt": returns,
        "sigma": sigma,
        "prices": prices_full,
        "start": test_indices[0],
        "t1": test_indices[-1],
        "dates": df["Date"].values,
    }
    
    Rt = compute_contract_returns_from_positions(prepared, positions, sigma_tgt, bp=bp)
    
    # Extract test period returns and dates
    test_returns = Rt[test_indices]
    test_dates = df["Date"].values[test_indices]
    test_positions = positions[test_indices]
    
    # Validate that some positions are non-zero
    non_zero_pct = (test_positions != 0).mean()
    if non_zero_pct < 0.01:
        print(f"    Warning: Only {non_zero_pct*100:.1f}% non-zero positions for {ticker}")
    
    return test_returns, test_dates, test_positions


def compute_portfolio_returns(
    asset_name: str,
    round_num: int,
    models: list[DQNAgent],
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    bp: float | None = None,
) -> tuple[pd.Series, int]:
    """
    Compute portfolio returns for an asset and round using Q-ensemble.
    
    Returns:
        (portfolio_return_series, n_contracts)
    """
    tickers = [
        ticker
        for ticker in universe_tickers(asset_name)
        if ticker not in EXCLUDED_CONTRACTS
    ]
    contract_series = []
    n_contracts = 0
    
    for ticker in tickers:
        result = backtest_contract_ensemble(ticker, round_num, models, sigma_tgt, bp=bp)
        if result is None:
            continue
        
        returns, dates, positions = result
        series = pd.Series(returns, index=pd.to_datetime(dates))
        contract_series.append(series)
        n_contracts += 1
        
        # Print position statistics
        non_zero_pct = (positions != 0).mean()
        print(f"    {ticker}: {len(returns)} days, {non_zero_pct*100:.1f}% non-zero positions")
    
    if n_contracts == 0:
        raise ValueError(f"No valid contracts for {asset_name} r{round_num}")
    
    # Align and compute portfolio returns (variable_n = mean across contracts)
    df = pd.DataFrame(contract_series).T.sort_index()
    portfolio = df.mean(axis=1)
    
    return portfolio, n_contracts


def run_asset_ensemble_backtest(
    asset_name: str,
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    port_vol_target: float = 0.97,
    bp_level: float | None = None,
) -> dict:
    """
    Run full ensemble backtest for an asset across both rounds.
    
    Args:
        bp_level: If specified, save results to bp{int(bp_level*10000)} subdirectory
    """
    print(f"\n{'='*70}")
    print(f"Asset: {asset_name}")
    print(f"{'='*70}")
    
    models_r1, bp_r1 = load_ensemble_models(asset_name, 1, bp=bp_level)
    models_r2, bp_r2 = load_ensemble_models(asset_name, 2, bp=bp_level)
    
    if len(models_r1) == 0 or len(models_r2) == 0:
        raise ValueError(f"Could not load enough models for {asset_name}")
    
    bp = bp_r1 if bp_r1 != BP else bp_r2
    
    print(f"\n  Round 1 (2011-2015):")
    port_r1, n1 = compute_portfolio_returns(asset_name, 1, models_r1, sigma_tgt, bp=bp)
    
    print(f"\n  Round 2 (2016-2019):")
    port_r2, n2 = compute_portfolio_returns(asset_name, 2, models_r2, sigma_tgt, bp=bp)
    
    portfolio_full = pd.concat([port_r1, port_r2]).sort_index()
    n_contracts_avg = (n1 + n2) / 2
    
    print(f"\n  Combined: {len(portfolio_full)} days, avg {n_contracts_avg:.1f} contracts")
    
    asset_slug = asset_name.replace(" ", "_")
    
    if bp_level is not None:
        save_dir = REPORTS_ROOT / asset_slug / f"bp{int(bp_level * 10000)}"
    else:
        save_dir = REPORTS_ROOT / asset_slug
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    npz_path = save_dir / "top5_ensemble_R.npz"
    np.savez(
        npz_path,
        portfolio_returns=portfolio_full.values.astype(np.float64),
        dates=np.array([d.strftime("%Y-%m-%d") for d in portfolio_full.index]),
    )
    print(f"  Saved: {npz_path}")
    
    R_scaled = get_portfolio_bridge("constant_posthoc", port_vol_target)(portfolio_full.values)
    metrics = compute_metrics(R_scaled, n_contracts=int(n_contracts_avg))
    metrics_dict = dict(zip(METRIC_NAMES, metrics))
    
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics_dict, "n_contracts": n_contracts_avg, "bp": bp}, f, indent=2)
    print(f"  Saved: {metrics_path}")
    
    return {
        "metrics": metrics_dict,
        "portfolio": portfolio_full,
        "n_contracts": n_contracts_avg,
    }


def compute_all_portfolio(
    results_dict: dict,
    port_vol_target: float = 0.97,
) -> dict:
    """Compute the 'All' portfolio: equal-weighted across ALL contracts.

    Per paper Section 4.3-4.4, "All" = (1/N) * sum(R_t^i) where N is the
    total number of contracts across all 4 asset classes. This weights each
    contract equally, NOT each asset class equally.

    Since per-asset portfolios are already (1/n_asset) averages of their
    contracts, we reconstruct the all-contracts average by weighting each
    per-asset portfolio by its contract count:
        all_port = (sum(n_asset * portfolio_asset)) / total_contracts

    Args:
        results_dict: dict mapping asset_name -> result dict with keys
            "metrics", "portfolio" (pd.Series), "n_contracts" (float)
        port_vol_target: portfolio volatility target for constant_posthoc bridge

    Returns:
        dict with keys "metrics", "portfolio", "n_contracts"
    """
    import logging
    logger = logging.getLogger(__name__)

    asset_names = ["Commodity", "Equity Index", "Fixed Income", "Forex"]

    portfolios = {}
    weights = {}
    for asset_name in asset_names:
        if asset_name not in results_dict:
            logger.warning("All portfolio: %s missing from results, skipping", asset_name)
            continue
        result = results_dict[asset_name]
        n = result["n_contracts"]
        if n == 0:
            logger.warning("All portfolio: %s has 0 contracts, skipping", asset_name)
            continue
        portfolios[asset_name] = result["portfolio"]
        weights[asset_name] = n

    if not portfolios:
        raise ValueError("No valid asset results for 'All' portfolio computation")

    total_contracts = sum(weights.values())

    port_df = pd.DataFrame(portfolios).dropna()
    weight_series = pd.Series(weights)
    all_port = port_df.dot(weight_series.reindex(port_df.columns).fillna(0)) / total_contracts

    R_scaled = get_portfolio_bridge("constant_posthoc", port_vol_target)(all_port.values)
    metrics = compute_metrics(R_scaled, n_contracts=int(total_contracts))
    metrics_dict = dict(zip(METRIC_NAMES, metrics))

    return {
        "metrics": metrics_dict,
        "portfolio": all_port,
        "n_contracts": total_contracts,
    }


def format_comparison_table(results: dict) -> str:
    """Format results as a comparison table."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("Table 2: Ensemble Q-Value Averaging vs Paper Table 2 DQN")
    lines.append("="*100)
    
    # Header
    header = f"{'Asset':<15} {'E(R)':>8} {'std(R)':>8} {'Sharpe':>8} {'Sortino':>8} {'MDD':>8} {'%+ve':>8} {'Ave P/L':>8}"
    lines.append(header)
    lines.append("-"*100)
    
    display_metrics = ["E(R)", "std(R)", "Sharpe", "Sortino", "MDD", "% +ve", "Ave P/L"]
    
    asset_order = ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]
    for asset_name in asset_order:
        if asset_name not in results:
            continue
        
        result = results[asset_name]
        our_metrics = result["metrics"]
        paper_metrics = PAPER_TABLE2[asset_name]["DQN"]
        
        # Our metrics
        our_values = [our_metrics[m] for m in display_metrics]
        lines.append(f"{asset_name:<15} " + " ".join([f"{v:>8.3f}" for v in our_values]))
        
        # Paper metrics
        paper_values = [paper_metrics[m] for m in display_metrics]
        lines.append(f"{'Paper T2 DQN':<15} " + " ".join([f"{v:>8.3f}" for v in paper_values]))
        
        # Diff
        diffs = [our_metrics[m] - paper_metrics[m] for m in display_metrics]
        lines.append(f"{'Diff':<15} " + " ".join([f"{v:>+8.3f}" for v in diffs]))
        lines.append("")
    
    lines.append("="*100)
    return "\n".join(lines)


import argparse

def main():
    parser = argparse.ArgumentParser(description="Top-5 Q-Value Ensemble Backtest")
    parser.add_argument("--tc-bp", type=float, default=None,
                        help="Run backtest at specific BP level (e.g., 0.0020)")
    parser.add_argument("--all-bp", action="store_true",
                        help="Run backtest for all BP levels in TC_BP_LEVELS")
    args = parser.parse_args()
    
    print("="*100)
    print("Top-5 Q-Value Ensemble Backtest for Table 2 Metrics")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Seeds: [42, 45, 47, 48, 49, 51] (top-5 by validation)")
    print(f"  Ensemble method: Q-value averaging → argmax")
    print(f"  σ_tgt: {SIGMA_TGT_DEFAULT}")
    print(f"  port_vol_target: 0.97")
    
    if args.tc_bp is not None:
        bp_levels = [args.tc_bp]
        print(f"  BP Level: {args.tc_bp}")
    elif args.all_bp:
        bp_levels = TC_BP_LEVELS
        print(f"  BP Levels: {TC_BP_LEVELS}")
    else:
        bp_levels = [None]
        print(f"  BP Level: from manifest (default)")
    
    all_results = {}
    
    for bp_level in bp_levels:
        if bp_level is not None:
            print(f"\n{'='*70}")
            print(f"BP Level: {bp_level} (bp{int(bp_level*10000)})")
            print(f"{'='*70}")
        
        REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
        
        for asset_name in ["Commodity", "Equity Index", "Fixed Income", "Forex"]:
            try:
                result = run_asset_ensemble_backtest(asset_name, bp_level=bp_level)
                all_results[asset_name] = result
            except Exception as e:
                print(f"\n  ERROR: Failed to process {asset_name}: {e}")
                import traceback
                traceback.print_exc()

        # Compute "All" portfolio: equal-weighted across all contracts
        if len(all_results) > 0:
            try:
                all_result = compute_all_portfolio(all_results, port_vol_target=0.97)
                all_results["All"] = all_result
                print(f"\n  All portfolio: {all_result['n_contracts']:.0f} total contracts")
            except Exception as e:
                print(f"\n  WARNING: Could not compute 'All' portfolio: {e}")

        if len(bp_levels) > 1:
            metrics_path = REPORTS_ROOT / f"bp{int(bp_level*10000)}" / "table2_metrics.json"
            metrics_to_save = {
                asset: {
                    "metrics": result["metrics"],
                    "n_contracts": result["n_contracts"],
                }
                for asset, result in all_results.items()
            }
            
            with open(metrics_path, "w") as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"\n  Saved metrics: {metrics_path}")
            
            print(format_comparison_table(all_results))
            all_results = {}
    
    # Single BP: save to bp{XX}/table2_metrics.json (per-BP isolation)
    bp_label = int(bp_levels[0] * 10000) if bp_levels[0] is not None else None
    if bp_label:
        metrics_path = REPORTS_ROOT / f"bp{bp_label}" / "table2_metrics.json"
    else:
        metrics_path = REPORTS_ROOT / "table2_metrics.json"
    metrics_to_save = {
        asset: {
            "metrics": result["metrics"],
            "n_contracts": result["n_contracts"],
        }
        for asset, result in all_results.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\n  Saved metrics: {metrics_path}")

    print(format_comparison_table(all_results))

    print("\n" + "="*100)
    print("DONE")
    print("="*100)


if __name__ == "__main__":
    main()
