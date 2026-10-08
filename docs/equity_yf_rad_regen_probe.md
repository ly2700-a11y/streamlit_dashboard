# ES/EN Yahoo RAD_REGEN Probe

## Mapping

- `ES` uses Yahoo `ES=F`
- `EN` uses Yahoo `NQ=F`

## Score Summary

| Scenario | <=10 | <=15 | Equity misses | All misses |
| --- | ---: | ---: | --- | --- |
| legacy upper bound (EN/ES excluded) | 36 | 41 | Calmar | MDD, Calmar |
| put back EN+ES with current CLC sources | 34 | 38 | Calmar | E(R), Sharpe, Sortino, MDD, Calmar |
| put back EN+ES with YF_NON on both | 34 | 38 | Calmar | E(R), Sharpe, Sortino, MDD, Calmar |
| put back EN+ES with YF_RAD_REGEN on both | 34 | 37 | Sharpe, Calmar | E(R), Sharpe, Sortino, MDD, Calmar |
| put back EN only via YF_RAD_REGEN | 34 | 37 | Sharpe, Calmar | E(R), Sharpe, Sortino, MDD, Calmar |
| put back ES only via YF_RAD_REGEN | 34 | 38 | Calmar | E(R), Sharpe, Sortino, MDD, Calmar |

## Key Metrics

| Scenario | Equity E(R) | Equity MDD | Equity Calmar | All E(R) | All MDD | All Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy upper bound (EN/ES excluded) | +0.470 | 0.126 | +0.331 | -0.013 | 0.125 | +0.300 |
| put back EN+ES with current CLC sources | +0.523 | 0.121 | +0.383 | +0.021 | 0.121 | +0.321 |
| put back EN+ES with YF_NON on both | +0.521 | 0.124 | +0.381 | +0.021 | 0.122 | +0.320 |
| put back EN+ES with YF_RAD_REGEN on both | +0.524 | 0.121 | +0.386 | +0.022 | 0.122 | +0.321 |
| put back EN only via YF_RAD_REGEN | +0.525 | 0.120 | +0.385 | +0.021 | 0.121 | +0.321 |
| put back ES only via YF_RAD_REGEN | +0.522 | 0.122 | +0.384 | +0.021 | 0.121 | +0.321 |
