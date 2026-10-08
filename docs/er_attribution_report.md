# E(R) Attribution Report

- Focus: continue Table 3 work toward the `40/45` target by proving where `E(R)` comes from and which contracts move the gap.
- Metric definition: `additive_subset`
- Active excluded set: `LB, ZO, CC, FB`
- Sigma: `0.0627`

## Math Identity

For variable-N aggregation,

```
R_port,t = (1 / N_t) * Σ_i R_i,t
E(R_port) = 252 * mean_t[(1 / N_t) * Σ_i R_i,t]
```

So each contract has realized annualized contribution

```
contrib_i = 252 * mean_t[I_i,t * R_i,t / N_t]
```

and because `R_i,t = signal_i,t - tc_i,t`, the same identity holds for the signal and transaction-cost pieces.

## Current 45-Comparison Context

- Current full 9-metric baseline score: `n10=24/45`, `n15=29/45`
- This confirms we are **not** at the `40/45` target yet and should keep pushing Table 3.

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 22 | 3 | 6 | -0.268 | -0.298 | 0.030 | -0.631 | -0.723 | 0.092 |
| Equity Index | 11 | 5 | 6 | +0.578 | +0.504 | 0.074 | +0.637 | +0.543 | 0.094 |
| Fixed Income | 4 | 7 | 7 | +0.602 | +0.605 | 0.003 | +0.649 | +0.645 | 0.004 |
| Forex | 9 | 6 | 7 | -0.215 | -0.198 | 0.017 | -0.469 | -0.420 | 0.049 |
| All | 46 | 3 | 3 | +0.043 | -0.013 | 0.056 | +0.114 | -0.036 | 0.150 |

## Add-Back Candidates From Current Excluded Set

| Ticker | Asset | Asset E(R) after add-back | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| |
|---|---|---|---|---|---|---|
| LB | Commodity | -0.249 | +0.019 | +0.028 | +0.003 | +0.011 |
| ZO | Commodity | -0.254 | +0.014 | +0.030 | +0.001 | +0.003 |
| CC | Commodity | -0.258 | +0.010 | +0.011 | -0.001 | -0.002 |
| FB | Fixed Income | +0.517 | +0.085 | +0.081 | +0.002 | +0.008 |

## Commodity

### Realized E(R) Contributors

| Ticker | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|
| KW | -0.026 | -0.025 | +0.001 | 2266 |
| SB | -0.026 | -0.025 | +0.001 | 2265 |
| ZL | -0.026 | -0.024 | +0.002 | 2266 |
| ZA | +0.018 | +0.020 | +0.002 | 2266 |
| DA | +0.013 | +0.015 | +0.002 | 2262 |
| ZT | +0.004 | +0.007 | +0.003 | 2266 |

### Best Leave-One-Out Diagnostics

| Ticker | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| |
|---|---|---|---|---|
| ZA | -0.029 | -0.078 | -0.009 | -0.022 |
| DA | -0.026 | -0.038 | -0.006 | -0.017 |
| ZT | -0.017 | -0.030 | -0.002 | -0.006 |
| ZL | +0.014 | +0.030 | +0.012 | +0.033 |
| KW | +0.015 | +0.035 | +0.012 | +0.033 |
| SB | +0.015 | +0.043 | +0.012 | +0.032 |

## Equity Index

### Realized E(R) Contributors

| Ticker | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|
| LX | +0.024 | +0.030 | +0.007 | 2278 |
| XU | +0.031 | +0.036 | +0.005 | 2283 |
| ER | +0.032 | +0.037 | +0.005 | 2301 |
| EN | +0.084 | +0.090 | +0.006 | 2268 |
| YM | +0.080 | +0.088 | +0.008 | 2268 |
| SP | +0.070 | +0.077 | +0.007 | 2268 |

### Best Leave-One-Out Diagnostics

| Ticker | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| |
|---|---|---|---|---|
| EN | -0.036 | -0.041 | -0.020 | -0.051 |
| YM | -0.032 | -0.033 | -0.019 | -0.049 |
| SP | -0.021 | -0.017 | -0.017 | -0.042 |
| ER | +0.017 | +0.016 | -0.006 | -0.013 |
| LX | +0.028 | +0.016 | -0.006 | -0.015 |
| XU | +0.032 | +0.034 | -0.004 | -0.008 |

## Fixed Income

### Realized E(R) Contributors

| Ticker | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|
| TY | +0.086 | +0.133 | +0.047 | 2267 |
| US | +0.112 | +0.137 | +0.024 | 2267 |
| UB | +0.151 | +0.253 | +0.102 | 2283 |
| DT | +0.253 | +0.298 | +0.045 | 2283 |
| UB | +0.151 | +0.253 | +0.102 | 2283 |
| US | +0.112 | +0.137 | +0.024 | 2267 |

### Best Leave-One-Out Diagnostics

| Ticker | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| |
|---|---|---|---|---|
| UB | +0.017 | +0.010 | -0.016 | -0.044 |
| US | +0.024 | +0.024 | -0.011 | -0.030 |
| TY | +0.051 | +0.052 | -0.008 | -0.023 |
| DT | +0.109 | +0.110 | -0.023 | -0.063 |

## Forex

### Realized E(R) Contributors

| Ticker | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|
| JN | -0.071 | -0.060 | +0.012 | 2268 |
| FN | -0.060 | -0.048 | +0.012 | 2267 |
| CN | -0.056 | -0.042 | +0.013 | 2268 |
| NK | +0.058 | +0.064 | +0.006 | 2325 |
| DX | +0.034 | +0.050 | +0.016 | 2312 |
| SN | -0.009 | +0.003 | +0.012 | 2268 |

### Best Leave-One-Out Diagnostics

| Ticker | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| |
|---|---|---|---|---|
| CN | +0.003 | -0.039 | +0.011 | +0.032 |
| FN | +0.008 | -0.022 | +0.012 | +0.032 |
| SN | +0.016 | +0.055 | +0.002 | +0.004 |
| JN | +0.021 | +0.042 | +0.014 | +0.034 |
| DX | +0.082 | +0.006 | -0.004 | -0.015 |
| NK | +0.092 | +0.153 | -0.003 | -0.007 |

## Interpretation

- `E(R)` remains the bottleneck; std / %+ve / Ave P/L are still the stable metrics.
- The add-back table tells us which excluded contracts are promising candidates under the current metric/scaling understanding.
- The leave-one-out tables identify where the current included universe is still structurally fighting the paper, especially in Equity and the All row.
- Next work should use these generated deltas to justify any future contract add-back or data-path investigation.
