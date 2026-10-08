# A2C Futures Trading Project

## Overview

This project reproduces and extends a deep reinforcement learning futures trading framework based on an Advantage Actor-Critic (A2C) architecture.

The A2C parts contains two main versions:

1. **Original Reward Version**
   R_orig = mu * [ (sigma_target / sigma_(t-1) * A_(t-1)) * r_t- bp * p_(t-1) * abs((sigma_target / sigma_(t-1) * A_(t-1)) - (sigma_target /sigma_(t-2) * A_(t-2))) ]
   - Uses the original reward function described in the paper.
   - Trains A2C directly on the paper-style trading reward.

2. **Extension Reward Version**
   R_new = R_orig - lambda_corr * (w^T * Sigma * w)
   where:

   - R_orig: original paper reward
   - Sigma: rolling covariance matrix
   - w: vector of volatility-scaled positions
   - lambda_corr: covariance penalty strength

   Adds a covariance-based portfolio penalty term.
   The objective is to reduce portfolio-level correlation risk while preserving trading performance.

The codebase is organized so that both versions share the same:

- feature engineering pipeline
- state construction
- LSTM-based actor/critic architecture
- training logic
- evaluation framework

The only major difference is the reward function used during training.

---

# Folder Structure

```text
A2C/
│
├── Original Reward/
│   ├── loaddata.py
│   ├── project_code.ipynb
│   ├── *.pt
│   ├── *.csv
│
├── Extention Reward/
│   ├── loaddata_new_reward.py
│   ├── project_code_new_reward.ipynb
│   ├── *.pt
│   ├── *.csv
```

---

# Original Reward Version

## Main Files

### `loaddata.py`

Core utility module.

Contains:

- data loading
- feature engineering
- state tensor construction
- reward calculation
- A2C architecture
- trading environment
- training utilities
- evaluation utilities

This file acts as the backend engine for the notebook.

---

### `project_code.ipynb`

Main notebook for the original reward version.

Responsible for:

- loading futures data
- generating features
- building training/testing datasets
- training A2C models
- generating A2C trading signals
- computing out-of-sample rewards
- plotting cumulative reward curves
- transaction cost robustness analysis
- Sharpe ratio analysis
- comparison with Table 3 in the paper

---

### `*.pt`

Saved PyTorch checkpoints.

Examples:

```text
a2c_Commodity_period1.pt
a2c_FX_period2.pt
```

Each file stores:

- actor network weights
- critic network weights
- optimizer states
- training log

---

### `*.csv`

Saved evaluation outputs.

Examples:

```text
a2c_tc_sharpe_by_asset_wide.csv
a2c_tc_final_cumulative_reward_by_asset_wide.csv
```

These store:

- cumulative reward summaries
- Sharpe ratios
- transaction cost robustness results
- Table 3 comparison tables

---

# Extension Reward Version

## Main Files

### `loaddata_new_reward.py`

Modified backend for the extended reward formulation.

Most of the code is identical to the original version.

The main additions are:

- covariance matrix estimation
- covariance lookup builder
- new reward rollout function
- covariance-penalized reward

---

### `project_code_new_reward.ipynb`

Main notebook for the extended reward version.

Responsible for:

- training the covariance-penalized A2C strategy
- evaluating out-of-sample performance
- transaction cost robustness analysis
- Sharpe ratio analysis
- comparison against the paper's Table 3 A2C results


---

# Assumptions

This section separates:

- assumptions explicitly stated in the paper
- implementation assumptions introduced in this project

---

# Assumptions Taken Directly From the Paper

## Futures Universe

The project uses the same 50 continuous futures contracts described in the paper.

Asset classes:

- Commodities
- Equity Indexes
- Fixed Income
- FX

---

## State Window

The state uses the previous 60 observations.

```text
window = 60
```

---

## Feature Set

The project follows the paper's feature design:

### Price normalization

Normalized close price series.

### Return features

- 1-month return
- 2-month return
- 3-month return
- 1-year return

Each normalized by:

```text
sigma_t * sqrt(h)
```

---

## MACD Features

The implementation follows the paper:

```text
q_t = (EMA(short) - EMA(long)) / std(price)
MACD_t = q_t / std(q_t)
```

---

## RSI Feature

RSI(30) using Wilder smoothing.

---

## A2C Hyperparameters

The following are directly taken from the paper:

```text
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.3
BATCH_SIZE = 128
BP = 0.0020
MU = 1.0
```

---

## LSTM Structure

The paper specifies:

- two-layer LSTM
- hidden sizes 64 and 32
- Leaky-ReLU activation

---

## Reward Function

Original reward function:

R_orig = mu * [ (sigma_target / sigma_(t-1) * A_(t-1)) * r_t
- bp * p_(t-1) * abs((sigma_target / sigma_(t-1) * A_(t-1)) - (sigma_target / sigma_(t-2) * A_(t-2))) ]

where:

- A_t: trading action / position
- r_t: additive daily return
- sigma_t: volatility estimate
- sigma_target: target volatility
- bp: transaction cost rate
- p_t: futures price
- mu: position scaling constant

The implementation computes:

```text
net_pnl = gross_pnl - trading_cost
```

with volatility scaling:

```text
sigma_target / sigma_t
```

This reward is recomputed during evaluation using compute_pnl().


The original reward follows the paper's Eq.(4):

```text
R_t = gross_pnl - trading_cost
```

with volatility scaling:

```text
sigma_target / sigma_t
```

---

# Implementation Assumptions Introduced in This Project

The paper does not fully specify all implementation details.

The following assumptions were added during implementation.

---

## Gaussian Policy Distribution

The paper states:

```text
continuous action space in [-1, 1]
```

but does not specify the policy distribution.

This implementation assumes:

```text
Gaussian policy
```

with:

```python
Normal(mu, std)
```

and:

```text
trainable global log_std
```

---

## Tanh Squashing

Actions are squashed into:

```text
[-1, 1]
```

using:

```python
action = tanh(z)
```

This is an implementation choice.

---

## LSTM Readout

The paper does not specify how to extract the final LSTM representation.

This implementation uses:

```python
last hidden state
```

from the second LSTM layer.

---

## Gradient Clipping

The paper does not specify gradient clipping.

This project uses:

```python
max_grad_norm = 1.0
```

for training stability.

---

## Entropy Regularization

The paper does not specify entropy regularization.

This implementation does NOT use entropy regularization.

The entropy term is intentionally disabled.

---

## Volatility Estimate

The paper references volatility scaling but does not explicitly define the estimator.

This implementation uses:

```python
EWMA volatility
span = 60
```

on additive daily returns.

---

## Normalized Price Feature

The paper discusses normalized prices but does not define the exact normalization.

This implementation uses:

```python
60-day rolling z-score
```

for the close price.

---

# Extension Reward Formulation

The extension version introduces a covariance-penalized reward:

R_new = R_orig - lambda_corr * (w^T * Sigma * w)

where:

- R_orig: original paper reward
- Sigma: rolling covariance matrix
- w: vector of volatility-scaled positions
- lambda_corr: covariance penalty strength

The term:

```text
w^T * Sigma * w
```

measures portfolio-level correlation risk.

The extension reward therefore attempts to:

```text
maximize trading reward while discouraging highly correlated portfolio exposures
```

The implementation computes:

```python
reward_new = reward_orig - lambda_corr * (w.T @ Sigma @ w)
```

inside collect_rollout_newreward().


The extension version introduces:

```text
R_corr = R_orig - lambda_corr * w^T Sigma w
```

where:

- `R_orig` = original paper reward
- `Sigma` = rolling covariance matrix
- `w` = vector of volatility-scaled positions
- `lambda_corr` = covariance penalty strength

---

## Covariance Matrix Assumptions

The paper does NOT include this extension.

The following assumptions were introduced by this project:

- rolling covariance estimation
- equal-weight portfolio aggregation
- covariance penalty term
- portfolio-level risk penalization

---

## Covariance Estimation

Covariance matrices are estimated using:

```python
hist.cov()
```

on rolling historical returns.

The covariance matrix at time t only uses past information:

```text
[t-lookback, ..., t-1]
```

so no future data leakage is introduced.

---

# Main Functions

This section summarizes the most important functions in the codebase.

---

# Data Loading Functions

## `find_clcdata(root_path)`

Searches recursively for the `CLCDATA` folder.

Returns:

```python
path_to_clcdata
```

---

## `read_rad_csv(file_path)`

Reads one Pinnacle continuous futures CSV file.

Returns:

```python
DataFrame
```

with:

- OHLC prices
- volume
- open interest
- parsed dates

---

## `load_paper_rad_data(root_path, start=None, end=None)`

Loads all futures contracts.

Returns:

```python
(data_dict, panel, missing)
```

where:

- `data_dict[ticker]` = individual futures DataFrame
- `panel` = stacked long-format panel
- `missing` = contracts not found

---

# Feature Engineering Functions

## `build_paper_features(data_dict, dropna=False)`

Builds all paper-style features.

Features include:

- returns
- EWMA volatility
- normalized prices
- MACD
- RSI

Returns:

```python
(feature_dict, feature_panel)
```

---

# State Tensor Functions

## `make_state_tensor_single()`

Converts one contract into rolling state tensors.

Each sample uses:

```text
previous 60 observations
```

Returns:

```python
X
```

with shape:

```text
(n_samples, 60, n_features)
```

---

## `make_state_tensors_all()`

Applies tensor construction to all contracts.

Returns:

```python
state_dict
```

---

# Reward / Evaluation Functions

## `compute_pnl()`

Recomputes the paper-style reward function.

Outputs:

- gross_pnl
- trading_cost
- net_pnl
- cumulative_net_pnl

This function is used for:

- final evaluation
- plotting
- transaction cost robustness
- Sharpe calculation

---

## `build_portfolio_pnl()`

Aggregates individual contract rewards into portfolio-level rewards.

---

# Environment Functions

## `PaperTradingEnv`

Trading environment used by A2C.

Responsibilities:

- state transitions
- reward calculation
- volatility scaling
- trading cost modeling

---

# A2C Functions

## `StackedLSTMBackbone`

Shared two-layer LSTM backbone.

Used by:

- actor
- critic

---

## `ActorContinuous`

Continuous-action policy network.

Outputs:

```python
(mu, std)
```

for Gaussian action sampling.

---

## `CriticValue`

State-value network.

Outputs:

```python
V(s)
```

---

## `collect_rollout()`

Collects synchronized rollouts from all environments.

Used during training.

---

## `compute_returns_and_advantages()`

Computes:

- discounted returns
- advantages

for actor-critic updates.

---

## `PaperA2CTrainer`

Main A2C training class.

Responsibilities:

- actor updates
- critic updates
- checkpoint saving
- checkpoint loading
- rollout collection

Main method:

```python
trainer.fit(...)
```

---

# Extension Reward Functions

## `build_cov_lookup_newreward()`

Builds rolling covariance matrices.

Returns:

```python
cov_lookup[date]
```

---

## `collect_rollout_newreward()`

Modified rollout collector for the covariance-penalized reward.

Implements:

```text
R_corr = R_orig - lambda_corr * w^T Sigma w
```

---

# Transaction Cost Robustness

The notebooks evaluate robustness across:

```text
1bp
5bp
10bp
15bp
20bp
25bp
30bp
35bp
40bp
45bp
```

The model is NOT retrained.

The same learned signal is re-evaluated using different trading cost assumptions.

---

# Train/Test Splits

Two train/test periods are used.

## Period 1

```text
Train: 2005–2010
Test: 2011–2015
```

## Period 2

```text
Train: 2005–2015
Test: 2016–2019
```

---

# How To Run The Project

## Step 1

Place Pinnacle continuous futures data inside:

```text
CLCDATA/
```

---

## Step 2

Open the notebook:

```text
project_code.ipynb
```

or:

```text
project_code_new_reward.ipynb
```

---

## Step 3

Run all notebook cells sequentially.

The notebook will:

- build features
- build state tensors
- train A2C
- save checkpoints
- generate signals
- compute rewards
- generate plots
- export CSV summaries

---

# Outputs

The project generates:

- cumulative reward curves
- Sharpe ratio summaries
- transaction cost robustness tables
- Table 3 comparison tables
- saved model checkpoints


