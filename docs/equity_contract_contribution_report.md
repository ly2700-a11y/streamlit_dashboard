# Equity Contract Contribution Report

- Scenario: `memory_5`, `variable_n`, `sigma=0.0618`
- Metric definition: `additive_subset`

| Ticker | Δ Equity |ER gap| | Δ Equity |Sharpe gap| | Δ All |ER gap| | Δ All |Sharpe gap| | Equity |ER gap| after drop | Equity |Sharpe gap| after drop |
|---|---|---|---|---|---|---|
| EN | -0.036 | -0.041 | -0.019 | -0.052 | 0.030 | 0.053 |
| YM | -0.032 | -0.033 | -0.019 | -0.049 | 0.034 | 0.061 |
| ES | -0.021 | -0.017 | -0.016 | -0.041 | 0.045 | 0.077 |
| SC | -0.021 | -0.017 | -0.016 | -0.042 | 0.045 | 0.077 |
| SP | -0.021 | -0.017 | -0.016 | -0.042 | 0.045 | 0.077 |
| MD | +0.014 | +0.017 | -0.008 | -0.022 | 0.080 | 0.111 |
| ER | +0.017 | +0.016 | -0.005 | -0.013 | 0.083 | 0.110 |
| XX | +0.018 | +0.013 | -0.008 | -0.020 | 0.084 | 0.107 |
| CA | +0.019 | +0.013 | -0.006 | -0.017 | 0.085 | 0.107 |
| LX | +0.027 | +0.016 | -0.005 | -0.014 | 0.093 | 0.110 |
| XU | +0.031 | +0.034 | -0.003 | -0.009 | 0.097 | 0.128 |

## Interpretation

- Most helpful leave-one-out candidates for Equity under the frozen Table 3 setup are: `EN, YM, ES`.
- A negative delta means removing that contract reduces the paper gap; these are diagnosis candidates, not automatic exclusions.
