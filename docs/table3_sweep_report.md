# Table 3 Sweep Report

- Metric definition used: `additive_subset`
- Sigma grid: `0.0618, 0.0621, 0.0624, 0.0627, 0.0630, 0.0633, 0.0636`
- Presets: `current_config, memory_5, memory_5_plus_us, memory_5_plus_us_zh`
- Aggregation modes: `variable_n, dropna`

## Top Scenarios

| # | Preset | Agg | Sigma | Lane A <10 | Lane A <15 | Lane B <10 | Lane B <15 | Lane A MAE | All |ER gap| | All |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | memory_5 | variable_n | 0.0618 | 9/16 | 13/16 | 12/12 | 12/12 | 9.30% | 0.056 | 0.152 |
| 2 | current_config | variable_n | 0.0618 | 8/16 | 13/16 | 12/12 | 12/12 | 9.24% | 0.052 | 0.137 |
| 3 | memory_5_plus_us_zh | variable_n | 0.0618 | 8/16 | 13/16 | 12/12 | 12/12 | 9.24% | 0.052 | 0.137 |
| 4 | current_config | variable_n | 0.0621 | 8/16 | 13/16 | 12/12 | 12/12 | 9.42% | 0.052 | 0.137 |
| 5 | memory_5_plus_us_zh | variable_n | 0.0621 | 8/16 | 13/16 | 12/12 | 12/12 | 9.42% | 0.052 | 0.137 |
| 6 | memory_5_plus_us | variable_n | 0.0618 | 8/16 | 13/16 | 12/12 | 12/12 | 9.58% | 0.046 | 0.122 |
| 7 | current_config | variable_n | 0.0624 | 8/16 | 13/16 | 12/12 | 12/12 | 9.62% | 0.052 | 0.137 |
| 8 | memory_5_plus_us_zh | variable_n | 0.0624 | 8/16 | 13/16 | 12/12 | 12/12 | 9.62% | 0.052 | 0.137 |
| 9 | current_config | variable_n | 0.0627 | 8/16 | 13/16 | 12/12 | 12/12 | 9.83% | 0.052 | 0.137 |
| 10 | memory_5_plus_us_zh | variable_n | 0.0627 | 8/16 | 13/16 | 12/12 | 12/12 | 9.83% | 0.052 | 0.137 |
| 11 | memory_5 | variable_n | 0.0621 | 9/16 | 12/16 | 12/12 | 12/12 | 9.43% | 0.057 | 0.152 |
| 12 | memory_5 | variable_n | 0.0624 | 9/16 | 12/16 | 12/12 | 12/12 | 9.59% | 0.057 | 0.152 |

## Best Scenario Detail

| Asset | # | E(R) err | Sharpe err | DD err | Sortino err | std err | %+ve err | P/L err | |ER gap| | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|---|
| Commodity | 21 | 8.1% | 12.0% | 14.7% | 19.6% | 4.6% | 2.5% | 2.9% | 0.024 | 0.087 |
| Equity Index | 11 | 13.1% | 17.3% | 15.8% | 2.3% | 3.6% | 1.3% | 0.8% | 0.066 | 0.094 |
| Fixed Income | 4 | 1.8% | 0.6% | 8.4% | 9.6% | 2.6% | 3.5% | 7.0% | 0.011 | 0.004 |
| Forex | 9 | 7.1% | 11.7% | 2.8% | 3.9% | 4.2% | 0.0% | 0.8% | 0.014 | 0.049 |
| All | 45 | 430.8% | 422.2% | 15.7% | 386.0% | 3.0% | 1.7% | 0.9% | 0.056 | 0.152 |

## Interpretation

- Best scenario: `memory_5`, `variable_n`, `sigma=0.0618`.
- `variable_n` still wins on Lane A coverage overall; `dropna` improves Equity but gives back more on Forex.
- The main remaining misses are Equity `E(R)` / `Sharpe` and the `All` row near-zero target problem.
