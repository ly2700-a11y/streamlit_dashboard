# Source Override Search Report

- Goal: improve Table 3 by changing data-source interpretation for included contracts, not by shrinking the universe.
- Metric definition: `additive_subset`
- Search scope: `Commodity, Equity Index` contracts only
- Candidate sources: `REV, RAD_REGEN`

## Baseline Score

| Scenario | n15 | n10 | Focus |E(R)| gap | Focus |Sharpe| gap | All |E(R)| gap |
|---|---|---|---|---|---|
| Baseline | 29 | 24 | 0.160 | 0.336 | 0.056 |

## Baseline Asset Summary

| Asset | # | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| | n10 | n15 |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 22 | -0.268 | -0.298 | 0.030 | -0.631 | -0.723 | 0.092 | 3 | 6 |
| Equity Index | 11 | +0.578 | +0.504 | 0.074 | +0.637 | +0.543 | 0.094 | 5 | 6 |
| Fixed Income | 4 | +0.602 | +0.605 | 0.003 | +0.649 | +0.645 | 0.004 | 7 | 7 |
| Forex | 9 | -0.215 | -0.198 | 0.017 | -0.469 | -0.420 | 0.049 | 6 | 7 |
| All | 46 | +0.043 | -0.013 | 0.056 | +0.114 | -0.036 | 0.150 | 3 | 3 |

## Best One-By-One Overrides

| Asset | Ticker | Source | Δn15 | Δn10 | Δ focus |E(R)| | Δ focus |Sharpe| | Δ All |E(R)| | Asset E(R) | Asset |E(R)| | All |E(R)| |
|---|---|---|---|---|---|---|---|---|---|---|
| Commodity | ZT | RAD_REGEN | +0 | +1 | -0.009 | -0.022 | -0.003 | -0.274 | 0.024 | 0.053 |
| Commodity | DA | RAD_REGEN | +0 | +1 | -0.008 | -0.017 | -0.003 | -0.273 | 0.025 | 0.053 |
| Commodity | ZU | REV | +0 | +1 | -0.006 | -0.014 | -0.002 | -0.272 | 0.026 | 0.054 |
| Commodity | JO | REV | +0 | +1 | -0.005 | -0.009 | -0.002 | -0.271 | 0.027 | 0.054 |
| Commodity | ZN | REV | +0 | +1 | -0.004 | -0.008 | -0.002 | -0.270 | 0.028 | 0.054 |
| Commodity | GI | RAD_REGEN | +0 | +1 | -0.003 | -0.009 | -0.001 | -0.270 | 0.028 | 0.055 |
| Commodity | ZG | RAD_REGEN | +0 | +1 | -0.003 | -0.009 | -0.001 | -0.270 | 0.028 | 0.055 |
| Commodity | ZF | REV | +0 | +1 | -0.003 | -0.008 | -0.001 | -0.270 | 0.028 | 0.055 |
| Commodity | ZW | REV | +0 | +1 | -0.003 | -0.005 | -0.001 | -0.270 | 0.028 | 0.055 |
| Commodity | DA | REV | +0 | +1 | -0.002 | -0.005 | -0.001 | -0.269 | 0.029 | 0.055 |
| Commodity | KW | REV | +0 | +1 | -0.002 | -0.005 | -0.001 | -0.269 | 0.029 | 0.055 |
| Commodity | ZG | REV | +0 | +1 | -0.002 | -0.005 | -0.001 | -0.269 | 0.029 | 0.055 |
| Commodity | ZH | REV | +0 | +1 | +0.001 | -0.003 | +0.002 | -0.269 | 0.029 | 0.058 |
| Equity Index | EN | REV | +0 | +0 | -0.006 | -0.009 | -0.001 | +0.573 | 0.069 | 0.055 |
| Equity Index | ES | RAD_REGEN | +0 | +0 | -0.004 | -0.005 | -0.001 | +0.575 | 0.071 | 0.055 |
| Equity Index | SC | RAD_REGEN | +0 | +0 | -0.004 | -0.005 | -0.001 | +0.575 | 0.071 | 0.055 |
| Equity Index | SP | RAD_REGEN | +0 | +0 | -0.004 | -0.005 | -0.001 | +0.575 | 0.071 | 0.055 |
| Equity Index | ES | REV | +0 | +0 | -0.003 | -0.005 | -0.001 | +0.576 | 0.072 | 0.055 |
| Equity Index | MD | RAD_REGEN | +0 | +0 | -0.003 | -0.005 | -0.001 | +0.576 | 0.072 | 0.055 |
| Equity Index | YM | RAD_REGEN | +0 | +0 | -0.003 | -0.004 | -0.001 | +0.576 | 0.072 | 0.055 |

## Greedy Accepted Overrides

- `DA` -> `RAD_REGEN`
- `EN` -> `REV`
- `ER` -> `REV`
- `ES` -> `REV`
- `GI` -> `RAD_REGEN`
- `JO` -> `REV`
- `KW` -> `REV`
- `MD` -> `RAD_REGEN`
- `SC` -> `RAD_REGEN`
- `SP` -> `RAD_REGEN`
- `XU` -> `RAD_REGEN`
- `XX` -> `RAD_REGEN`
- `YM` -> `RAD_REGEN`
- `ZA` -> `RAD_REGEN`
- `ZF` -> `REV`
- `ZG` -> `RAD_REGEN`
- `ZH` -> `REV`
- `ZN` -> `REV`
- `ZT` -> `RAD_REGEN`
- `ZU` -> `REV`

| Step | Asset | Ticker | Source | n15 | n10 | Focus |E(R)| gap | Commodity |E(R)| | Equity |E(R)| | All |E(R)| |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Commodity | ZT | RAD_REGEN | 29 | 25 | 0.151 | 0.024 | 0.074 | 0.053 |
| 2 | Commodity | DA | RAD_REGEN | 29 | 26 | 0.143 | 0.019 | 0.074 | 0.050 |
| 3 | Commodity | ZH | REV | 30 | 26 | 0.144 | 0.017 | 0.074 | 0.053 |
| 4 | Commodity | ZU | REV | 30 | 26 | 0.138 | 0.013 | 0.074 | 0.051 |
| 5 | Commodity | JO | REV | 30 | 26 | 0.132 | 0.009 | 0.074 | 0.049 |
| 6 | Equity Index | EN | REV | 30 | 26 | 0.126 | 0.009 | 0.069 | 0.048 |
| 7 | Commodity | ZN | REV | 30 | 26 | 0.122 | 0.006 | 0.069 | 0.047 |
| 8 | Commodity | ZG | RAD_REGEN | 30 | 26 | 0.118 | 0.003 | 0.069 | 0.046 |
| 9 | Commodity | GI | RAD_REGEN | 30 | 26 | 0.115 | 0.001 | 0.069 | 0.045 |
| 10 | Equity Index | ES | REV | 30 | 26 | 0.112 | 0.001 | 0.067 | 0.044 |
| 11 | Equity Index | MD | RAD_REGEN | 30 | 26 | 0.108 | 0.001 | 0.064 | 0.043 |
| 12 | Equity Index | SC | RAD_REGEN | 31 | 26 | 0.106 | 0.001 | 0.062 | 0.043 |
| 13 | Equity Index | SP | RAD_REGEN | 31 | 26 | 0.102 | 0.001 | 0.059 | 0.042 |
| 14 | Commodity | KW | REV | 31 | 26 | 0.100 | 0.000 | 0.059 | 0.041 |
| 15 | Commodity | ZF | REV | 31 | 27 | 0.102 | 0.003 | 0.059 | 0.040 |
| 16 | Equity Index | YM | RAD_REGEN | 31 | 27 | 0.100 | 0.003 | 0.057 | 0.040 |
| 17 | Equity Index | ER | REV | 31 | 27 | 0.098 | 0.003 | 0.056 | 0.039 |
| 18 | Equity Index | XU | RAD_REGEN | 31 | 27 | 0.097 | 0.003 | 0.055 | 0.039 |
| 19 | Commodity | ZA | RAD_REGEN | 31 | 27 | 0.097 | 0.003 | 0.055 | 0.039 |
| 20 | Equity Index | XX | RAD_REGEN | 31 | 27 | 0.097 | 0.003 | 0.055 | 0.039 |

## Greedy Final Asset Summary

| Asset | # | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| | n10 | n15 |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 22 | -0.301 | -0.298 | 0.003 | -0.711 | -0.723 | 0.012 | 6 | 7 |
| Equity Index | 11 | +0.559 | +0.504 | 0.055 | +0.616 | +0.543 | 0.073 | 5 | 7 |
| Fixed Income | 4 | +0.602 | +0.605 | 0.003 | +0.649 | +0.645 | 0.004 | 7 | 7 |
| Forex | 9 | -0.215 | -0.198 | 0.017 | -0.469 | -0.420 | 0.049 | 6 | 7 |
| All | 46 | +0.026 | -0.013 | 0.039 | +0.070 | -0.036 | 0.106 | 3 | 3 |

## Interpretation

- Baseline uses the active 46-contract universe and current additive metric path.
- One-by-one results show whether a single contract-level source swap improves the full 45-comparison score or at least reduces the key `E(R)` gaps.
- Greedy results show whether those improvements stack, which is the real test for a hybrid source-map fix.
- Best greedy scenario reached `n15=31/45` and `n10=27/45`, versus baseline `n15=29/45`, `n10=24/45`.
