# Final Table 3 Comparison Report

- Final metric definition: `additive_subset`
- Final preset: `memory_5`
- Final aggregation: `variable_n`
- Final sigma: `0.0618`
- Final exclusion set: `LB, JO, ZO, CC, FB`

| Asset | # | E(R) ours | E(R) paper | E(R) err | Sharpe ours | Sharpe paper | Sharpe err | DD ours | DD paper | DD err | std ours | std paper | std err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Commodity | 21 | -0.274 | -0.298 | 8.1% | -0.636 | -0.723 | 12.0% | 0.296 | 0.258 | 14.7% | 0.431 | 0.412 | 4.6% |
| Equity Index | 11 | 0.57 | 0.504 | 13.1% | 0.637 | 0.543 | 17.3% | 0.702 | 0.606 | 15.8% | 0.895 | 0.928 | 3.6% |
| Fixed Income | 4 | 0.594 | 0.605 | 1.8% | 0.649 | 0.645 | 0.6% | 0.608 | 0.561 | 8.4% | 0.915 | 0.939 | 2.6% |
| Forex | 9 | -0.212 | -0.198 | 7.1% | -0.469 | -0.42 | 11.7% | 0.293 | 0.285 | 2.8% | 0.452 | 0.472 | 4.2% |
| All | 45 | 0.043 | -0.013 | 430.8% | 0.116 | -0.036 | 422.2% | 0.266 | 0.23 | 15.7% | 0.374 | 0.363 | 3.0% |

## Interpretation

- Reproducibility rerun identical: `True`.
- Lane A score: `13/16` within 15%; Lane B score: `12/12` within 15%.
- Table 3 is frozen on the current best upstream scenario; `MDD/Calmar` remain report-only.
