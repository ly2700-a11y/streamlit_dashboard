"""
DRL Trading Strategies - 模块化 Python 框架

一个完整的交易策略框架，包含：
- 传统策略: Long Only, Sign(R), MACD
- 深度强化学习: DQN (论文复现), A2C (待上传), Route B
- 完整的数据管理、指标计算、可视化工具

使用示例:
    from src.core.strategies import LongOnlyStrategy, SignRStrategy, MACDStrategy
    from src.core.models.dqn import DQNEvaluator
    
    long_strategy = LongOnlyStrategy(config)
    dqn_evaluator = DQNEvaluator(model_path)
"""

__version__ = "1.0.0"
__author__ = "IEOR 4733 Project Team"

from src.core import strategies, models, data, utils

__all__ = [
    "strategies",
    "models", 
    "data",
    "utils",
]
