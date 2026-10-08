"""
strategies.py — Trading signal functions
Reference: [4] Baz et al. 2015 for MACD; [37] Moskowitz et al. 2012 for Sign(R)

All strategies operate in the additive framework (Paper Section 3.2):
  r_t = p_t - p_{t-1}  (additive profits on raw prices)
  
Signal conventions:
- Long Only: A_t = 1
- Sign(R): A_t = sign(sum(r_{t-252:t}))  [Paper Eq 10]
- MACD: A_t = φ(MACD_tilde)  [Paper Eq 3, 11, 12]
"""
import numpy as np
import pandas as pd
from config import SIGN_LOOKBACK, MACD_PAIRS, MACD_VOL_WINDOW, MACD_STD_WINDOW


# =============================================================================
# Strategy 1: Long Only
# =============================================================================
def strategy_long_only(n):
    """Long Only: A_t = 1 for all t. Position ∈ {+1}."""
    return np.ones(n)


# =============================================================================
# Strategy 2: Sign(R) — [37] Moskowitz et al. 2012; [27] Lim et al. 2019
# =============================================================================
def strategy_sign_r(additive_returns, lookback=SIGN_LOOKBACK):
    """
    Sign(R): A_t = sign(r_{t-252:t})  [Paper Eq 10]
    
    In the additive framework:
      r_t = p_t - p_{t-1}
      r_{t-252:t} = sum of r_{t-251}, r_{t-250}, ..., r_t  (252 returns)
                 = p_t - p_{t-252}
      A_t = sign(p_t - p_{t-252})
    
    NOTE: additive_returns[i] must be r_i = p_i - p_{i-1} for i >= 1.
    Then sum(rt[t-251:t+1]) = p_t - p_{t-252}.
    
    Position ∈ {-1, 0, +1}.
    
    Reference: [37] Moskowitz, Ooi, Pedersen 2012; [27] Lim, Zohren, Roberts 2019
    """
    positions = np.zeros(len(additive_returns))
    for t in range(lookback, len(additive_returns)):
        # sum r_{t-251}...r_t = p_t - p_{t-252}
        cum_ret = np.sum(additive_returns[t - lookback + 1 : t + 1])
        positions[t] = np.sign(cum_ret)
    return positions


# =============================================================================
# Strategy 3: MACD — [4] Baz et al. 2015
# =============================================================================
def strategy_macd(prices, pairs=MACD_PAIRS,
                  vol_window=MACD_VOL_WINDOW,
                  std_window=MACD_STD_WINDOW):
    """
    MACD: multi-timeframe MACD with φ transformation.
    
    Paper Eq 3: MACD_t = q_t / std(q_{t-252:t})
    Paper Eq 11-12: A_t = φ(MACD_tilde), where MACD_tilde = avg across pairs
    
    For each (S, L) pair:
      q_t = (EMA(P, S) - EMA(P, L)) / std(P_{t-63:t})
      MACD_t = q_t / std(q_{t-252:t})
    
    Average MACD_t across pairs, then apply:
      φ(x) = x * exp(-x²/4) / 0.89
    
    Position ∈ [-1, +1].
    
    NOTE: Prices used directly (p0 normalization is a no-op for Eq 4).
    
    Reference: [4] Baz et al. 2015
    """
    positions = np.zeros(len(prices))
    macd_sum = np.zeros(len(prices))

    for S, L in pairs:
        ema_fast = pd.Series(prices).ewm(span=S, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=L, adjust=False).mean()
        std_vol = pd.Series(prices).rolling(vol_window, min_periods=vol_window).std()
        q = (ema_fast - ema_slow) / std_vol
        std_q = q.rolling(std_window, min_periods=std_window).std()
        macd_sum += (q / std_q).fillna(0).values

    macd_avg = macd_sum / len(pairs)

    for t in range(std_window, len(prices)):
        m = macd_avg[t]
        if not (np.isnan(m) or np.isinf(m)):
            positions[t] = m * np.exp(-m ** 2 / 4) / 0.89

    return positions
