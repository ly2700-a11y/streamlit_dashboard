"""
Utils Module - 通用工具函数

包含内容:
    - metrics.py: 风险收益指标计算
    - indicators.py: 技术指标计算
    - visualization.py: 绘图和可视化
    - helpers.py: 通用助手函数

主要函数:
    - calculate_sharpe_ratio()
    - calculate_max_drawdown()
    - calculate_returns_metrics()
    - plot_strategy_comparison()
    - compute_sma(), compute_ema()
    - compute_macd()

使用示例:
    from src.core.utils import metrics, indicators
    
    sharpe = metrics.calculate_sharpe_ratio(returns)
    macd = indicators.compute_macd(prices)
"""

__all__ = [
    "metrics",
    "indicators",
    "visualization",
    "helpers",
]
