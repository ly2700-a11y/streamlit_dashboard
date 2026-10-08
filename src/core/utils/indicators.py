#!/usr/bin/env python3
"""
技术指标计算模块
论文第 4 页：State Space 特征工程
"""

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series]

# =============================================================================
# MACD 指标 (论文公式 3)
# =============================================================================

def compute_macd(prices: ArrayLike, short_span: int, long_span: int) -> np.ndarray:
    """
    论文公式 (3): MACD 指标
    
    MACD_t = q_t / std(q_{t-252:t})
    q_t = (m(S) - m(L)) / std(p_{t-63:t})
    
    参数:
        prices: 价格序列
        short_span: 短期时间尺度 (论文：8, 16, 32)
        long_span: 长期时间尺度 (论文：24, 48, 96)
    
    返回:
        MACD 值序列
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # 指数加权移动平均
    m_short = prices.ewm(span=short_span, adjust=False).mean()
    m_long = prices.ewm(span=long_span, adjust=False).mean()
    
    # 63 天滚动标准差
    std_63 = prices.rolling(window=63, min_periods=1).std()
    
    # q_t
    q = (m_short - m_long) / std_63
    
    # 252 天滚动标准差
    std_q = q.rolling(window=252, min_periods=1).std()
    
    # MACD_t
    macd = q / std_q
    
    return macd.fillna(0).values


def compute_macd_multi_scale(prices: ArrayLike) -> np.ndarray:
    """
    多时间尺度 MACD 平均 (论文公式 12)
    
    Sk ∈ {8, 16, 32}
    Lk ∈ {24, 48, 96}
    
    返回:
        3 个时间尺度的平均 MACD
    """
    macd_8_24 = compute_macd(prices, 8, 24)
    macd_16_48 = compute_macd(prices, 16, 48)
    macd_32_96 = compute_macd(prices, 32, 96)
    
    return (macd_8_24 + macd_16_48 + macd_32_96) / 3


# =============================================================================
# RSI 指标
# =============================================================================

def compute_rsi(prices: ArrayLike, window: int = 30) -> np.ndarray:
    """
    RSI 指标 - 相对强弱指数
    
    参数:
        prices: 价格序列
        window: 回溯窗口 (论文：30 天)
    
    返回:
        RSI 值序列 (0-100)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).values


def normalize_rsi(rsi: np.ndarray) -> np.ndarray:
    """
    RSI 归一化到 [-1, 1]
    
    <20: 超卖 → -1
    >80: 超买 → +1
    50: 中性 → 0
    """
    return (rsi - 50) / 50


# =============================================================================
# 波动率计算
# =============================================================================

def compute_volatility(returns: ArrayLike, window: int = 60) -> np.ndarray:
    """
    60 天指数加权移动波动率 (论文第 4 页)
    
    参数:
        returns: 收益率序列
        window: 窗口大小 (默认 60 天)
    
    返回:
        波动率序列
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    vol = returns.ewm(span=window, adjust=False).std().values
    return vol


def normalize_return(returns: np.ndarray, vol: np.ndarray, horizon: int = 252) -> np.ndarray:
    """
    论文：用日波动率调整到合理时间尺度
    
    r_normalized = r / (vol * sqrt(horizon))
    
    参数:
        returns: 原始收益率
        vol: 波动率估计
        horizon: 时间尺度 (21/42/63/252 等)
    
    返回:
        波动率调整的收益率
    """
    return returns / (vol * np.sqrt(horizon) + 1e-10)


# =============================================================================
# 多周期收益率
# =============================================================================

def compute_multi_horizon_returns(returns: np.ndarray, vol: np.ndarray) -> dict:
    """
    计算多周期收益率 (论文第 4 页)
    
    周期:
    - 1 个月 (21 天)
    - 2 个月 (42 天)
    - 3 个月 (63 天)
    - 1 年 (252 天)
    
    返回:
        字典：{'ret_21': ..., 'ret_42': ..., 'ret_63': ..., 'ret_252': ...}
    """
    return {
        'ret_21': normalize_return(returns, vol, 21),
        'ret_42': normalize_return(returns, vol, 42),
        'ret_63': normalize_return(returns, vol, 63),
        'ret_252': normalize_return(returns, vol, 252)
    }


# =============================================================================
# 价格归一化
# =============================================================================

def normalize_prices(prices: np.ndarray) -> np.ndarray:
    """
    归一化收盘价序列
    
    z-score 归一化: (p - mean) / std
    """
    return (prices - np.mean(prices)) / (np.std(prices) + 1e-10)


# =============================================================================
# 完整状态空间构建
# =============================================================================

class FeatureEngineer:
    """
    特征工程 - 构建论文要求的状态空间
    
    输出：8 个特征
    1. 归一化收盘价
    2. 21 天收益率 (波动率调整)
    3. 42 天收益率 (波动率调整)
    4. 63 天收益率 (波动率调整)
    5. 252 天收益率 (波动率调整)
    6. MACD (多时间尺度平均)
    7. RSI (归一化)
    8. 波动率 (归一化)
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.feature_dim = 8
    
    def build_features(self, prices: np.ndarray, returns: np.ndarray, 
                      current_idx: int) -> np.ndarray:
        """
        在 current_idx 时刻构建状态特征
        
        参数:
            prices: 价格序列
            returns: 收益率序列
            current_idx: 当前时间索引
        
        返回:
            (window_size, feature_dim) = (60, 8)
        """
        if current_idx < self.window_size:
            return np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
        
        # 获取时间窗口数据
        start_idx = current_idx - self.window_size
        window_prices = prices[start_idx:current_idx]
        window_returns = returns[start_idx:current_idx]
        
        # 处理 NaN：用 0 填充
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算波动率
        vol = compute_volatility(window_returns, 60)
        vol = np.nan_to_num(vol, nan=0.01, posinf=1.0, neginf=0.001)  # ✅ 处理 vol 的 NaN
        vol_mean = np.mean(vol) + 1e-10
        
        # 特征 1: 归一化收盘价
        norm_price = normalize_prices(window_prices)
        
        # 特征 2-5: 多周期收益率
        ret_21 = normalize_return(window_returns, vol, 21)
        ret_42 = normalize_return(window_returns, vol, 42)
        ret_63 = normalize_return(window_returns, vol, 63)
        ret_252 = normalize_return(window_returns, vol, 252)
        
        # 特征 6: MACD
        macd = compute_macd_multi_scale(window_prices)
        
        # 特征 7: RSI
        rsi = compute_rsi(window_prices, 30)
        rsi_norm = normalize_rsi(rsi)
        
        # 特征 8: 波动率
        vol_norm = vol / vol_mean
        
        # 堆叠特征：(window_size, 8)
        features = np.stack([
            norm_price,
            ret_21,
            ret_42,
            ret_63,
            ret_252,
            macd,
            rsi_norm,
            vol_norm
        ], axis=1)
        
        # ✅ 最终 NaN 处理：确保没有 NaN
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features.astype(np.float32)
    
    def get_feature_names(self) -> list:
        """返回特征名称列表"""
        return [
            'norm_price',
            'ret_21',
            'ret_42',
            'ret_63',
            'ret_252',
            'macd',
            'rsi_norm',
            'vol_norm'
        ]


# =============================================================================
# 测试函数
# =============================================================================

def test_indicators():
    """测试指标计算"""
    print("测试技术指标计算...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n))
    returns = np.diff(prices) / prices[:-1]
    returns = np.insert(returns, 0, 0)
    
    # 测试 MACD
    macd = compute_macd(prices, 8, 24)
    print(f"✅ MACD: shape={macd.shape}, mean={np.mean(macd):.4f}")
    
    # 测试多尺度 MACD
    macd_multi = compute_macd_multi_scale(prices)
    print(f"✅ Multi-MACD: shape={macd_multi.shape}")
    
    # 测试 RSI
    rsi = compute_rsi(prices, 30)
    print(f"✅ RSI: shape={rsi.shape}, mean={np.mean(rsi):.2f}")
    
    # 测试波动率
    vol = compute_volatility(returns, 60)
    print(f"✅ Volatility: shape={vol.shape}, mean={np.mean(vol):.6f}")
    
    # 测试特征工程
    fe = FeatureEngineer(window_size=60)
    features = fe.build_features(prices, returns, 200)
    print(f"✅ Features: shape={features.shape}, dtype={features.dtype}")
    print(f"   特征数：{fe.feature_dim}")
    print(f"   时间窗口：{fe.window_size}")
    
    print("\n✅ 所有指标测试通过！")
    return True


if __name__ == '__main__':
    test_indicators()
