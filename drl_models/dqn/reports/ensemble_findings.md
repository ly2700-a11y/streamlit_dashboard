# DQN Ensemble Results — Validation-Selected Top-k

**Date**: 2026-05-06
**Method**: 10 seeds per retrain window → validation ranking by best_val_reward → top-k Q-value ensemble → OOS test
**Gamma**: 0.6 (fixed, pre-selected from Forex tuning)
**Features**: structural_38, 9-dim (price norm + 4 returns + 3 MACD + RSI)
**Model**: DuelingDQNLSTM (2-layer LSTM 64→32, dueling heads, Double DQN, dropout 0.2)

---

## Validation Rankings (used for seed selection)

### Commodity
| r1 Rank | Seed | Best Val | r2 Rank | Seed | Best Val |
|---------|------|----------|---------|------|----------|
| 1 | s42 | +0.649 | 1 | s49 | +1.076 |
| 2 | s48 | +0.578 | 2 | s46 | +1.052 |
| 3 | s45 | +0.551 | 3 | s43 | +1.047 |
| 4 | s50 | +0.408 | 4 | s45 | +1.029 |
| 5 | s44 | +0.279 | 5 | s48 | +0.901 |

### Forex
| r1 Rank | Seed | Best Val | r2 Rank | Seed | Best Val |
|---------|------|----------|---------|------|----------|
| 1 | s46 | +1.010 | 1 | s47 | +0.305 |
| 2 | s47 | +1.003 | 2 | s45 | +0.219 |
| 3 | s44 | +0.943 | 3 | s48 | +0.167 |
| 4 | s48 | +0.889 | 4 | s49 | +0.144 |
| 5 | s50 | +0.844 | 5 | s50 | +0.111 |

### Equity Index
| r1 Rank | Seed | Best Val | r2 Rank | Seed | Best Val |
|---------|------|----------|---------|------|----------|
| 1 | s42 | +2.040 | 1 | s42 | +1.441 |
| 2 | s45 | +1.972 | 2 | s47 | +1.032 |
| 3 | s50 | +1.826 | 3 | s50 | +0.888 |
| 4 | s48 | +1.802 | 4 | s43 | +0.800 |
| 5 | s43 | +1.742 | 5 | s46 | +0.666 |

### Fixed Income
| r1 Rank | Seed | Best Val | r2 Rank | Seed | Best Val |
|---------|------|----------|---------|------|----------|
| 1 | s47 | +0.450 | 1 | s42 | +0.153 |
| 2 | s44 | +0.421 | 2 | s47 | +0.112 |
| 3 | s51 | +0.229 | 3 | s49 | +0.111 |
| 4 | s48 | +0.210 | 4 | s43 | +0.106 |
| 5 | s50 | +0.189 | 5 | s44 | +0.093 |

---

## OOS Performance (Q-value Ensemble)

### Commodity
| Method | Period | E(R) | Sharpe | Sortino | MDD | %+ve |
|--------|--------|------|--------|---------|-----|------|
| Paper DQN | 2011-2019 | +0.187 | +0.623 | +1.085 | 0.039 | 0.498 |
| Paper Long | 2011-2019 | -0.298 | -0.723 | -1.152 | 0.248 | 0.473 |
| Best Single (s51) | r1 | **+0.020** | **+0.204** | +0.235 | 0.008 | 0.267 |
| Top-3 Ensemble | r1 | -0.177 | -1.217 | -1.366 | 0.070 | 0.263 |
| Top-5 Ensemble | r1 | -0.161 | -1.100 | -1.225 | 0.068 | 0.267 |
| Top-3 Ensemble | r2 | -0.080 | -0.436 | -0.480 | 0.034 | 0.215 |
| Top-5 Ensemble | r2 | -0.067 | -0.352 | -0.385 | 0.028 | 0.213 |

> **Finding**: Individual best seed (s51) outperforms both top-3 and top-5 ensembles on r1. Validation ranking does NOT correlate with OOS performance — s51 ranked 10th in validation but 1st in OOS.

### Forex
| Method | Period | E(R) | Sharpe | Sortino | MDD | %+ve |
|--------|--------|------|--------|---------|-----|------|
| Paper DQN | 2011-2019 | +0.272 | +0.560 | +0.972 | 0.085 | 0.510 |
| Paper Long | 2011-2019 | -0.198 | -0.420 | -0.696 | 0.219 | 0.491 |
| Top-3 Ensemble | r2 | -0.028 | -0.417 | -0.265 | 0.048 | 0.108 |

> **Finding**: Top-3 r2 ensemble achieves E(R) = -0.028 — nearly breakeven after costs. However, r1 ensemble is deeply negative, suggesting the model degrades over time.

### Equity Index & Fixed Income
> **Finding**: All ensemble models voted "hold" (action=0) for every time step. Q-value averaging results in hold being the consensus action. Individual models DO trade (as shown in per-seed backtests), but in ensemble they disagree and cancel out to hold.

---

## Key Findings

1. **Paper unreproducible** — All ensembles are far below paper DQN claims
2. **Validation selection fails** — Best validation seed ≠ best OOS seed (s51 ranked 10th in validation, 1st in OOS for Commodity)
3. **Equity/FI ensemble paralysis** — Models disagree too much, resulting in all-hold 
4. **r1 vs r2 asymmetry** — r2 ensembles are consistently better than r1, suggesting the extra training data helps
5. **Single best seed outranks ensemble** — Ensemble averaging degrades performance when seeds have divergent strategies

## Limitations

- Validation split is only 10% of training data → validation performance is noisy
- 10 seeds may be insufficient for stable ensemble
- Q-value scale differences between models can bias averaging
- No transaction cost consideration in validation ranking (only raw val_reward)
