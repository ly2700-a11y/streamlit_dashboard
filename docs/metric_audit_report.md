# Metric Audit Report

- Scenario: `current_config`, `variable_n`, `sigma=0.0627`
- Compared definitions: `additive_subset, nav_subset, nav_full`
- Selected reporting DD: `additive_subset`
- MDD policy: `diagnostic_only`

## Summary Table

| Definition | Avg DD err | Avg Sortino err | Avg MDD err | Avg Calmar err | Avg ours |Calmar-ER/MDD| |
|---|---|---|---|---|---|
| additive_subset | 13.8% | 76.0% | 949.9% | 179.8% | 0.000 |
| nav_subset | 1131.8% | 70.1% | 689.9% | 1723.2% | 0.067 |
| nav_full | 1110.4% | 71.5% | 689.9% | 1723.2% | 0.067 |

## Conclusion

- `DD_subset` on the additive portfolio remains the most plausible reporting definition because it matches the paper text most literally and stays competitive on DD/Sortino errors.
- NAV-based DD/MDD are useful diagnostics, but they do not fully resolve the paper’s Calmar inconsistency.
- `MDD` and `Calmar` remain diagnostic-only for scenario selection; the paper stays internally inconsistent even after the best tested bridge.

## additive_subset

| Asset | # | DD ours | DD paper | DD err | MDD ours | MDD paper | MDD err | Ours |Calmar-ER/MDD| | Paper |Calmar-ER/MDD| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 20 | 0.295 | 0.258 | 14.3% | 2.63 | 0.248 | 960.5% | 0.000 | 1.072 |
| Equity Index | 11 | 0.712 | 0.606 | 17.5% | 0.893 | 0.127 | 603.1% | 0.000 | 3.503 |
| Fixed Income | 3 | 0.629 | 0.561 | 12.1% | 0.494 | 0.108 | 357.4% | 0.001 | 5.147 |
| Forex | 9 | 0.297 | 0.285 | 4.2% | 1.838 | 0.219 | 739.3% | 0.000 | 0.803 |
| All | 43 | 0.278 | 0.23 | 20.9% | 0.81 | 0.037 | 2089.2% | 0.001 | 0.342 |

## nav_subset

| Asset | # | DD ours | DD paper | DD err | MDD ours | MDD paper | MDD err | Ours |Calmar-ER/MDD| | Paper |Calmar-ER/MDD| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 20 | 8.475 | 0.258 | 3184.9% | 1.0 | 0.248 | 303.2% | 0.000 | 1.072 |
| Equity Index | 11 | 0.0 | 0.606 | 100.0% | 0.001 | 0.127 | 99.2% | 0.307 | 3.503 |
| Fixed Income | 3 | 0.005 | 0.561 | 99.1% | 0.013 | 0.108 | 88.0% | 0.028 | 5.147 |
| Forex | 9 | 2.114 | 0.285 | 641.8% | 1.0 | 0.219 | 356.6% | 0.000 | 0.803 |
| All | 43 | 3.986 | 0.23 | 1633.0% | 1.0 | 0.037 | 2602.7% | 0.000 | 0.342 |

## nav_full

| Asset | # | DD ours | DD paper | DD err | MDD ours | MDD paper | MDD err | Ours |Calmar-ER/MDD| | Paper |Calmar-ER/MDD| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 20 | 8.274 | 0.258 | 3107.0% | 1.0 | 0.248 | 303.2% | 0.000 | 1.072 |
| Equity Index | 11 | 0.0 | 0.606 | 100.0% | 0.001 | 0.127 | 99.2% | 0.307 | 3.503 |
| Fixed Income | 3 | 0.005 | 0.561 | 99.1% | 0.013 | 0.108 | 88.0% | 0.028 | 5.147 |
| Forex | 9 | 2.186 | 0.285 | 667.0% | 1.0 | 0.219 | 356.6% | 0.000 | 0.803 |
| All | 43 | 3.862 | 0.23 | 1579.1% | 1.0 | 0.037 | 2602.7% | 0.000 | 0.342 |
