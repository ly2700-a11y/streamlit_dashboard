#!/usr/bin/env python3
"""Generate Table 3 comparison across Long, MACD, and Sign(R) strategies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import frontier modules from each strategy folder
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "baseline" / "macd"))
sys.path.insert(0, str(ROOT / "baseline" / "signr"))

import frontier_40plus_enumeration as fe_long  # noqa: E402
import frontier_40plus_enumeration_macd as fe_macd  # noqa: E402
import frontier_40plus_enumeration_signr as fe_signr  # noqa: E402
from config import METRIC_NAMES  # noqa: E402


STRATEGIES = [
    ("Long", fe_long.LEGACY_EXPERIMENTAL_OVERRIDES_LONG, fe_long.LEGACY_EXPERIMENTAL_EXCLUDED_LONG, fe_long),
    ("Sign(R)", fe_signr.LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR, fe_signr.LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR, fe_signr),
    ("MACD", fe_macd.LEGACY_EXPERIMENTAL_OVERRIDES_MACD, fe_macd.LEGACY_EXPERIMENTAL_EXCLUDED_MACD, fe_macd),
]

ASSETS = ["Commodity", "Equity Indexes", "Fixed Income", "FX", "All"]


def get_results_for_strategy(strategy_name, overrides, excluded, fe_module):
    """获取指定策略的所有资产结果"""
    # 根据各策略的最优参数选择 numerator_mode
    numerator_mode_map = {
        "Long": "annual_mean_sleeve",
        "Sign(R)": "annual_mean_sleeve",
        "MACD": "wealth_cagr",
    }
    
    summary = fe_module.evaluate_scenario(
        overrides_key=tuple(sorted(overrides.items())),
        excluded_key=tuple(sorted(excluded)),
        default_capital_mode="risk_price_source",
        asset_capital_overrides_key=(("Equity Index", "risk_price_non"),),
        numerator_mode=numerator_mode_map[strategy_name],
        asset_path_mode="contract_equal_path",
        all_mode="contract_equal_path",
    )
    
    results = {}
    # Map internal asset names to display names
    asset_mapping = {
        "Commodity": "Commodity",
        "Equity Index": "Equity Indexes",
        "Fixed Income": "Fixed Income",
        "Forex": "FX",
        "All": "All"
    }
    
    for internal_asset, display_asset in asset_mapping.items():
        if internal_asset in summary["results"]:
            metrics = summary["results"][internal_asset]["metrics"]
            results[display_asset] = {
                "E(R)": metrics.get("E(R)", np.nan),
                "std(R)": metrics.get("std(R)", np.nan),
                "DD": metrics.get("DD", np.nan),
                "Sharpe": metrics.get("Sharpe", np.nan),
                "Sortino": metrics.get("Sortino", np.nan),
                "MDD": metrics.get("MDD", np.nan),
                "Calmar": metrics.get("Calmar", np.nan),
                "% +ve": metrics.get("% +ve", np.nan),
                "Ave P/L": metrics.get("Ave P/L", np.nan),
            }
    
    return results


def format_metric(value: float, precision: int = 3) -> str:
    """格式化指标值"""
    if not np.isfinite(value):
        return "—"
    if abs(value) < 0.001:
        return f"{value:.{precision}f}"
    return f"{value:.{precision}f}"


def print_asset_table(asset_name: str, all_results: dict[str, dict]) -> None:
    """打印单个资产的表格"""
    print(f"{'':12} | ", end="")
    print(" | ".join(f"{col:>8}" for col in METRIC_NAMES))
    print("-" * (14 + len(METRIC_NAMES) * 11))
    
    for strategy in ["Long", "Sign(R)", "MACD"]:
        print(f"{strategy:12} | ", end="")
        metrics = all_results.get(strategy, {}).get(asset_name, {})
        values = [metrics.get(col, np.nan) for col in METRIC_NAMES]
        print(" | ".join(f"{format_metric(v):>8}" for v in values))


def main():
    # 获取各策略结果
    all_strategy_results = {}
    
    print("=" * 100)
    print("Generating Table 3: Experiment Results for the Raw Signal")
    print("=" * 100)
    print("\nFetching results from three strategies...")
    
    for strategy_name, overrides, excluded, fe_module in STRATEGIES:
        print(f"  Processing {strategy_name}...", end=" ", flush=True)
        results = get_results_for_strategy(strategy_name, overrides, excluded, fe_module)
        all_strategy_results[strategy_name] = results
        print("✓")
    
    # 打印表格
    print("\n")
    print("=" * 100)
    
    for asset in ASSETS:
        print(f"\n{asset.upper()}")
        print("-" * 100)
        print_asset_table(asset, all_strategy_results)
    
    print("\n" + "=" * 100)
    print("\nNote:")
    print("  • Long: Experimental upper bound with source violations (41/45 score)")
    print("  • Sign(R): Optimal for momentum strategy (16/36 score)")
    print("  • MACD: Optimal for mean-reversion strategy (17/36 score)")
    print("=" * 100)


if __name__ == "__main__":
    main()
