"""
Strategies Module - 交易策略实现

包含的策略:
    - LongOnlyStrategy: 长仓基准
    - SignRStrategy: 基于sign(收益率)的信号策略
    - MACDStrategy: MACD 技术指标策略
    - RouteB: A2C + 制度检测策略
    - DQNPaperStrategy: 论文中的 DQN 策略
"""

__all__ = [
    "LongOnlyStrategy",
    "SignRStrategy", 
    "MACDStrategy",
    "RouteB",
    "DQNPaperStrategy",
]
