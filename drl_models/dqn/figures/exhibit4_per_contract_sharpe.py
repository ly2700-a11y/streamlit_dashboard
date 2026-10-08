#!/usr/bin/env python3
"""
Exhibit 4: Per-Contract Annualized Sharpe — DQN Seeds vs Long Only

Generates a boxplot comparing per-contract annualized Sharpe ratios across
4 asset classes. For each asset:
- Long Only baseline (single box)
- DQN 10 seeds (42-51), each a box showing distribution of per-contract Sharpes

Data pipeline uses same contracts as Long Only (baseline_run.load_contracts),
which applies default preset (structural_38) with source_overrides/excluded_contracts.

Usage:
    python3 drl/dqn/figures/exhibit4_per_contract_sharpe.py

Output:
    drl/dqn/figures/exhibit4_per_contract_sharpe.png (300 DPI)
"""
import sys
sys.path.insert(0, '.')

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from baseline_run import (
    load_contracts,
    compute_contract_returns,
    compute_contract_returns_from_positions,
)
from drl_models.dqn.model import DQNAgent
from drl_models.dqn.spec import contract_data_path, resolve_checkpoint_path
from drl_shared.state_space import action_id_to_position, get_feature_window
from drl_shared.spec import SEQ_LEN as FEATURE_SEQ_LEN, current_source_policy

# Configuration
ASSETS = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']
ASSET_SLUGS = {
    'Commodity': 'Commodity',
    'Equity Index': 'Equity_Index',
    'Fixed Income': 'Fixed_Income',
    'Forex': 'Forex',
}
SEEDS = list(range(42, 52))  # 42-51
SIGMA_TGT = 0.058
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
ROUND_NUM = 1  # Use r1-trained models per task
SOURCE_POLICY = current_source_policy()

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / 'drl' / 'dqn' / 'reports' / 'per_contract'
OUTPUT_PATH = REPO_ROOT / 'drl' / 'dqn' / 'figures' / 'exhibit4_per_contract_sharpe.png'
DPI = 300

def compute_annualized_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    mean_daily = np.mean(returns)
    std_daily = np.std(returns)
    if std_daily == 0:
        return 0.0
    # Annualized Sharpe = mean_daily * 252 / (std_daily * sqrt(252))
    return mean_daily * 252 / (std_daily * np.sqrt(252))


def get_cache_path(asset: str, ticker: str, seed: int) -> Path:
    """Get cache path for per-contract DQN returns."""
    asset_slug = ASSET_SLUGS[asset]
    return CACHE_DIR / asset_slug / f"{ticker}_s{seed}.npz"


def run_dqn_inference(
    asset: str,
    seed: int,
    rd: Dict,
    device: str = 'cpu'
) -> np.ndarray:
    """Run DQN inference for a single contract and return positions."""
    ticker = rd['tk']
    prices = np.asarray(rd['prices'], dtype=float)
    start, t1 = int(rd['start']), int(rd['t1'])
    eval_dates = pd.to_datetime(rd['dates'])
    eval_len = min(len(eval_dates), max(0, t1 - start + 1))
    
    positions = np.zeros(len(prices), dtype=float)
    if eval_len <= 0:
        return positions
    
    # Load checkpoint for this asset/seed
    ckpt_path = resolve_checkpoint_path(ROUND_NUM, asset, run_id='latest')
    # Find the specific seed bundle
    asset_slug = ASSET_SLUGS[asset]
    r1_dir = REPO_ROOT / 'drl' / 'dqn' / 'models' / asset_slug / 'r1'
    
    if not r1_dir.exists():
        raise FileNotFoundError(f"R1 directory not found: {r1_dir}")
    
    # Find bundle for this seed
    seed_bundle = None
    for bundle in sorted(r1_dir.iterdir()):
        if bundle.is_dir() and f'_s{seed}' in bundle.name:
            seed_bundle = bundle
            break
    
    if seed_bundle is None:
        raise FileNotFoundError(f"No r1 bundle found for {asset} seed {seed}")
    
    ckpt_path = seed_bundle / 'checkpoint.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load agent
    agent = DQNAgent(device=device)
    agent.load(ckpt_path)
    agent.q_net.eval()
    
    # Load precomputed features from npz
    npz_path = contract_data_path(ROUND_NUM, ticker)
    if not npz_path.exists():
        raise FileNotFoundError(f"Features not found: {npz_path}")
    
    npz_data = np.load(npz_path, allow_pickle=True)
    round_features = npz_data['features']
    npz_dates = pd.to_datetime(npz_data['dates'])
    
    # Build date mapping
    date_to_npz = {d: i for i, d in enumerate(npz_dates)}
    baseline_to_npz = {}
    eval_dates_ts = pd.to_datetime(eval_dates[:eval_len])
    for i, ed in enumerate(eval_dates_ts):
        npz_idx = date_to_npz.get(ed)
        if npz_idx is not None:
            baseline_to_npz[start + i] = npz_idx
    
    # Get valid indices (need warmup)
    from drl_models.dqn.spec import WARMUP
    full_indices = np.arange(start, start + eval_len)
    valid = full_indices[(full_indices >= WARMUP) & (full_indices < len(prices))]
    
    if len(valid) == 0:
        return positions
    
    # Resolve features
    npz_indices = np.array([baseline_to_npz.get(int(v), -1) for v in valid], dtype=int)
    ok = npz_indices >= FEATURE_SEQ_LEN
    if not ok.any():
        return positions
    
    # Build states
    states = np.stack([
        get_feature_window(round_features, int(npz_indices[j]))
        for j in range(len(valid)) if ok[j]
    ]).astype(np.float32)
    valid = valid[ok]
    
    # Run inference in batches
    batch_size = 2048
    action_ids = []
    import torch
    for i in range(0, len(states), batch_size):
        batch = states[i:i+batch_size]
        tensor = torch.from_numpy(batch).to(agent.device)
        with torch.no_grad():
            batch_actions = agent.q_net(tensor).argmax(dim=1).cpu().numpy()
        action_ids.extend(batch_actions)
    
    # Convert action IDs to positions
    positions[valid] = [action_id_to_position(int(aid)) for aid in action_ids]
    
    return positions


def get_dqn_per_contract_returns(
    asset: str,
    seed: int,
    raw_data: List[Dict],
    device: str = 'cpu',
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    returns_dict = {}
    
    for rd in raw_data:
        ticker = rd['tk']
        cache_path = get_cache_path(asset, ticker, seed)
        
        if use_cache and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            returns_dict[ticker] = cached['returns']
            continue
        
        npz_path = contract_data_path(ROUND_NUM, ticker)
        if not npz_path.exists():
            print(f"    Skipping {ticker}: features not found at {npz_path}")
            continue
        
        try:
            positions = run_dqn_inference(asset, seed, rd, device=device)
            Rt = compute_contract_returns_from_positions(rd, positions, SIGMA_TGT)
            start, t1 = rd['start'], rd['t1']
            returns_sliced = Rt[start:t1+1]
            
            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, returns=returns_sliced, ticker=ticker, seed=seed)
            
            returns_dict[ticker] = returns_sliced
        except Exception as e:
            print(f"    ERROR for {ticker}: {e}")
            continue
    
    return returns_dict


def get_long_only_per_contract_returns(
    raw_data: List[Dict]
) -> Dict[str, np.ndarray]:
    """Get per-contract returns for Long Only baseline."""
    returns_dict = {}
    
    for rd in raw_data:
        ticker = rd['tk']
        Rt = compute_contract_returns(rd, 'Long', SIGMA_TGT)
        start, t1 = rd['start'], rd['t1']
        returns_sliced = Rt[start:t1+1]
        returns_dict[ticker] = returns_sliced
    
    return returns_dict


def compute_per_contract_sharpes(returns_dict: Dict[str, np.ndarray]) -> List[float]:
    """Compute annualized Sharpe ratio for each contract."""
    sharpes = []
    for ticker, returns in returns_dict.items():
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = compute_annualized_sharpe(returns)
            sharpes.append(sharpe)
    return sharpes


def load_or_compute_all_sharpes(
    device: str = 'cpu',
    use_cache: bool = True
) -> Dict[str, Dict]:
    """
    Load or compute all per-contract Sharpe ratios.
    
    Returns:
        {
            asset: {
                'Long': [sharpe1, sharpe2, ...],
                42: [sharpe1, sharpe2, ...],
                43: [sharpe1, sharpe2, ...],
                ...
            }
        }
    """
    all_sharpes = {}
    
    for asset in ASSETS:
        print(f"\nProcessing {asset}...")
        
        # Load contracts for this asset
        raw_data = load_contracts(
            asset,
            test_start=TEST_START,
            test_end=TEST_END,
            excluded_contracts=SOURCE_POLICY['excluded_contracts'],
            source_overrides=SOURCE_POLICY['source_overrides'],
        )
        
        if not raw_data:
            print(f"  Warning: No contracts found for {asset}")
            continue
        
        print(f"  Loaded {len(raw_data)} contracts")
        
        asset_sharpes = {}
        
        # Long Only baseline
        print(f"  Computing Long Only returns...")
        long_returns = get_long_only_per_contract_returns(raw_data)
        asset_sharpes['Long'] = compute_per_contract_sharpes(long_returns)
        print(f"    Long Only: {len(asset_sharpes['Long'])} contracts, "
              f"mean Sharpe = {np.mean(asset_sharpes['Long']):.3f}")
        
        # DQN seeds
        for seed in SEEDS:
            print(f"  Computing DQN seed {seed}...")
            try:
                dqn_returns = get_dqn_per_contract_returns(
                    asset, seed, raw_data, device=device, use_cache=use_cache
                )
                asset_sharpes[seed] = compute_per_contract_sharpes(dqn_returns)
                print(f"    Seed {seed}: {len(asset_sharpes[seed])} contracts, "
                      f"mean Sharpe = {np.mean(asset_sharpes[seed]):.3f}")
            except Exception as e:
                print(f"    ERROR for seed {seed}: {e}")
                asset_sharpes[seed] = []
        
        all_sharpes[asset] = asset_sharpes
    
    return all_sharpes


def create_boxplot(all_sharpes: Dict[str, Dict]):
    """Create the boxplot figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
    fig.suptitle(
        'Exhibit 4: Per-Contract Annualized Sharpe — DQN Seeds vs Long Only',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Colors
    long_color = '#ff7f0e'  # Orange for Long Only
    dqn_color = '#1f77b4'   # Blue for DQN seeds
    
    for idx, asset in enumerate(ASSETS):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if asset not in all_sharpes:
            ax.set_title(asset, fontsize=11, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        asset_sharpes = all_sharpes[asset]
        
        # Prepare data for boxplot
        # Box 0: Long Only
        # Boxes 1-10: Seeds 42-51
        data_to_plot = []
        labels = []
        colors = []
        
        # Long Only
        if 'Long' in asset_sharpes and len(asset_sharpes['Long']) > 0:
            data_to_plot.append(asset_sharpes['Long'])
            labels.append('Long')
            colors.append(long_color)
        
        # DQN seeds
        for seed in SEEDS:
            if seed in asset_sharpes and len(asset_sharpes[seed]) > 0:
                data_to_plot.append(asset_sharpes[seed])
                labels.append(f's{seed}')
                colors.append(dqn_color)
        
        if not data_to_plot:
            ax.set_title(asset, fontsize=11, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create boxplot
        bp = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Styling
        ax.set_title(asset, fontsize=11, fontweight='bold')
        ax.set_ylabel('Annualized Sharpe', fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
        
        # Rotate x labels for seeds
        if len(labels) > 1:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    print("=" * 70)
    print("Exhibit 4: Per-Contract Annualized Sharpe — DQN Seeds vs Long Only")
    print("=" * 70)
    print(f"Assets: {ASSETS}")
    print(f"DQN Seeds: {SEEDS}")
    print(f"Sigma Target: {SIGMA_TGT}")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"Output Path: {OUTPUT_PATH}")
    print("=" * 70)
    
    # Load or compute all Sharpe ratios
    all_sharpes = load_or_compute_all_sharpes(device='cpu', use_cache=True)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    for asset in ASSETS:
        if asset not in all_sharpes:
            continue
        print(f"\n{asset}:")
        asset_sharpes = all_sharpes[asset]
        
        if 'Long' in asset_sharpes and len(asset_sharpes['Long']) > 0:
            long_sharpes = asset_sharpes['Long']
            print(f"  Long Only: n={len(long_sharpes)}, "
                  f"mean={np.mean(long_sharpes):.3f}, std={np.std(long_sharpes):.3f}, "
                  f"min={np.min(long_sharpes):.3f}, max={np.max(long_sharpes):.3f}")
        
        dqn_means = []
        for seed in SEEDS:
            if seed in asset_sharpes and len(asset_sharpes[seed]) > 0:
                seed_sharpes = asset_sharpes[seed]
                dqn_means.append(np.mean(seed_sharpes))
        
        if dqn_means:
            print(f"  DQN Seeds (mean of means): {np.mean(dqn_means):.3f} "
                  f"± {np.std(dqn_means):.3f}")
    
    # Create figure
    print("\nGenerating boxplot...")
    fig = create_boxplot(all_sharpes)
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {OUTPUT_PATH}")
    
    # Also save PDF
    pdf_path = OUTPUT_PATH.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close(fig)
    print("\nDone!")


if __name__ == '__main__':
    main()
