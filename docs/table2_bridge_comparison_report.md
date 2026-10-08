# Table 2 Bridge Comparison Report

- Frozen Table 3 scenario: `memory_5`, `variable_n`, `sigma=0.0618`
- Metric definition: `additive_subset`

## Bridge Scoreboard

| Bridge | Lane A <10 | Lane A <15 | Lane B <10 | Lane B <15 | Lane A MAE | Worst std err | All |ER gap| | All |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|
| rolling252_lagged | 2/16 | 4/16 | 8/12 | 8/12 | 51.86% | 89.99% | 0.045 | 0.043 |
| constant_posthoc | 2/16 | 3/16 | 8/12 | 8/12 | 53.65% | 89.79% | 0.043 | 0.058 |
| ewma60_lagged | 2/16 | 2/16 | 7/12 | 8/12 | 58.23% | 89.58% | 0.044 | 0.049 |

## Best Bridge Detail

| Asset | E(R) ours | E(R) paper | E(R) err | Sharpe ours | Sharpe paper | Sharpe err | std ours | std paper | std err | Realized ann std | Yearly sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Commodity | -0.063 | -0.71 | 91.1% | -0.649 | -0.726 | 10.6% | 0.098 | 0.979 | 90.0% | 0.098 | 2011:-0.05, 2012:-0.05, 2013:-0.16 |
| Equity Index | 0.065 | 0.668 | 90.3% | 0.649 | 0.688 | 5.7% | 0.1 | 0.97 | 89.7% | 0.100 | 2011:-0.05, 2012:+0.07, 2013:+0.21 |
| Fixed Income | 0.066 | 0.68 | 90.3% | 0.652 | 0.698 | 6.6% | 0.102 | 0.975 | 89.5% | 0.102 | 2011:+0.21, 2012:+0.07, 2013:-0.12 |
| Forex | -0.044 | -0.344 | 87.2% | -0.433 | -0.353 | 22.7% | 0.101 | 0.973 | 89.6% | 0.101 | 2011:+0.00, 2012:+0.01, 2013:-0.03 |
| All | 0.01 | 0.055 | 81.8% | 0.101 | 0.058 | 74.1% | 0.099 | 0.975 | 89.8% | 0.099 | 2011:-0.02, 2012:+0.04, 2013:+0.02 |

## Interpretation

- Best tested Table 2 bridge: `rolling252_lagged`.
- Selection prioritized Lane A coverage, then std alignment, then `All` absolute gaps.
- This report compares every bridge against the paper and against the current constant baseline implicitly through the bridge table.
