# DQN Alignment Notes

## Current retained direction

- one model per contract
- shared RL state space across `DQN / PG / A2C`
- Table 3 only
- global baseline backtesting engine shared by baseline and RL strategies

## Shared state space

The retained 8-dimensional schema:

1. normalized close price
2. 21-day vol-adjusted return
3. 42-day vol-adjusted return
4. 63-day vol-adjusted return
5. 252-day vol-adjusted return
6. MACD feature (63-window volatility normalization)
7. RSI(30)
8. volatility ratio

State window:
- `seq_len = 60`

## Volatility convention

The RL stack intentionally uses a single volatility estimator:

- `sigma_t = EWMA(60)` on daily additive returns

Used in:
- Eq.4 reward scaling
- all horizon return-feature normalization
- volatility feature

MACD feature convention:
- 63-window volatility normalization (locked)

## Action adapters

- `DQN`: discrete `{-1,0,1}`
- `PG`: discrete `{-1,0,1}` (future consumer)
- `A2C`: continuous `[-1,1]` (future consumer)

## DQN architecture

Retained in the single-contract model:
- LSTM `[64, 32]`
- Leaky-ReLU
- fixed Q-targets
- Double DQN
- dueling DQN

## Retrain schedule

- `r1`: train `2005-2010`, test `2011-2015`
- `r2`: train `2005-2015`, test `2016-2019`

Applied per contract.

## Backtesting

Backtesting computes portfolio metrics from the simulated portfolio path:
- `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L, MDD, Calmar`

`MDD` and `Calmar` are computed on the portfolio path directly (no separate reporting bridge in this RL phase).
