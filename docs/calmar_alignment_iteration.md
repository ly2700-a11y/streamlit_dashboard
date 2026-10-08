# Calmar Alignment Iteration

Frozen strategy:

- Table 3 Long only
- ignore `All` until the 4 asset rows stabilize
- fixed trade world
- frozen reporting bridge starts from `RISK_PRICE_SIGMA0`
- frozen rebuilt historical strong baseline:
  - `EN -> REV`
  - `DT -> REV`
  - `CC -> RAD_REGEN`
  - `LB -> RAD`
  - `JO -> RAD_REGEN`
  - `ZH -> RAD_REGEN`
  - exclusions: `FB, ZA, ZO`

## Iteration 0 — Frozen Baseline

| Asset | Trade E(R) | Reporting Annual Return | Paper-Implied Annual Return | MDD | Paper MDD | Calmar | Paper Calmar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Commodity | -0.268 | -0.095 | -0.032 | 0.700 | 0.248 | -0.135 | -0.130 |
| Equity Index | +0.523 | +0.047 | +0.059 | 0.146 | 0.127 | +0.324 | +0.466 |
| Fixed Income | +0.555 | +0.046 | +0.049 | 0.111 | 0.108 | +0.414 | +0.455 |
| Forex | -0.173 | -0.020 | -0.022 | 0.220 | 0.219 | -0.090 | -0.101 |

## Iteration 1 — Commodity Suspect Contract Audit

Suspects: `SB, KC, ZL, NR, ZC`

### SB

- current source: `RAD`
- best source candidate: `NON`
- best source score: `+0.0580`
- drop score: `+0.1239`
- classification: `exclusion candidate`
- recommendation: `drop`

| Source | Rows | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | 2266 | 0.0623 | 0.4524 | 0.0050 | 10.1% | 3.2% | 5.5% | +0.0000 |
| REV | 2266 | 0.0628 | 0.4551 | 0.0051 | 10.1% | 3.2% | 5.6% | -0.0034 |
| RAD_REGEN | 2266 | 0.0602 | 0.4431 | 0.0037 | 10.7% | 3.4% | 5.9% | +0.0126 |
| NON | 2266 | 0.0497 | 0.4070 | 0.0048 | 13.4% | 3.2% | 5.1% | +0.0580 |
| NON_FWD_ANCHORED | 2266 | 0.0625 | 0.4537 | 0.0050 | 10.1% | 3.2% | 5.6% | -0.0015 |

| Drop Effect | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drop | 0.0391 | 0.3466 | 0.0100 | 14.1% | 3.6% | 5.5% | +0.1239 |

### KC

- current source: `RAD`
- best source candidate: `NON`
- best source score: `+0.0494`
- drop score: `+0.0528`
- classification: `exclusion candidate`
- recommendation: `drop`

| Source | Rows | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | 2266 | 0.0623 | 0.4524 | 0.0050 | 10.1% | 3.2% | 5.5% | +0.0000 |
| REV | 2266 | 0.0589 | 0.4359 | 0.0033 | 10.1% | 3.2% | 5.5% | +0.0215 |
| RAD_REGEN | 2266 | 0.0607 | 0.4485 | 0.0035 | 10.7% | 3.2% | 5.4% | +0.0069 |
| NON | 2266 | 0.0525 | 0.4155 | 0.0022 | 15.1% | 3.2% | 5.1% | +0.0494 |
| NON_FWD_ANCHORED | 1958 | 0.0706 | 0.3732 | 0.0356 | 10.1% | 3.4% | 6.0% | +0.0402 |

| Drop Effect | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drop | 0.0541 | 0.4117 | 0.0009 | 13.1% | 4.4% | 7.3% | +0.0528 |

### ZL

- current source: `RAD`
- best source candidate: `NON`
- best source score: `+0.0268`
- drop score: `+0.0384`
- classification: `exclusion candidate`
- recommendation: `drop`

| Source | Rows | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | 2267 | 0.0623 | 0.4524 | 0.0050 | 10.1% | 3.2% | 5.5% | +0.0000 |
| REV | 2267 | 0.0638 | 0.4598 | 0.0057 | 10.4% | 3.4% | 5.8% | -0.0098 |
| RAD_REGEN | 2267 | 0.0596 | 0.4426 | 0.0030 | 12.1% | 3.6% | 6.1% | +0.0143 |
| NON | 2267 | 0.0574 | 0.4341 | 0.0014 | 13.4% | 3.6% | 5.7% | +0.0268 |
| NON_FWD_ANCHORED | 2267 | 0.0620 | 0.4512 | 0.0047 | 10.4% | 3.4% | 5.8% | +0.0017 |

| Drop Effect | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drop | 0.0560 | 0.4239 | 0.0013 | 14.4% | 3.0% | 4.9% | +0.0384 |

### NR

- current source: `RAD`
- best source candidate: `NON`
- best source score: `+0.0473`
- drop score: `+0.0352`
- classification: `source distortion`
- recommendation: `NON`

| Source | Rows | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | 2265 | 0.0623 | 0.4524 | 0.0050 | 10.1% | 3.2% | 5.5% | +0.0000 |
| REV | 2265 | 0.0587 | 0.4355 | 0.0030 | 10.1% | 3.2% | 5.5% | +0.0225 |
| RAD_REGEN | 2265 | 0.0605 | 0.4467 | 0.0036 | 11.1% | 3.2% | 5.6% | +0.0088 |
| NON | 2265 | 0.0521 | 0.4171 | 0.0032 | 15.4% | 4.0% | 6.3% | +0.0473 |
| NON_FWD_ANCHORED | 2265 | 0.0616 | 0.4499 | 0.0045 | 10.4% | 3.2% | 5.6% | +0.0036 |

| Drop Effect | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drop | 0.0567 | 0.4257 | 0.0020 | 12.1% | 3.8% | 6.3% | +0.0352 |

### ZC

- current source: `RAD`
- best source candidate: `NON`
- best source score: `+0.0422`
- drop score: `+0.0257`
- classification: `source distortion`
- recommendation: `NON`

| Source | Rows | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | 2267 | 0.0623 | 0.4524 | 0.0050 | 10.1% | 3.2% | 5.5% | +0.0000 |
| REV | 2267 | 0.0610 | 0.4473 | 0.0041 | 10.1% | 3.2% | 5.5% | +0.0072 |
| RAD_REGEN | 2267 | 0.0607 | 0.4469 | 0.0037 | 11.1% | 3.2% | 5.4% | +0.0083 |
| NON | 2267 | 0.0541 | 0.4222 | 0.0011 | 14.4% | 3.6% | 5.8% | +0.0422 |
| NON_FWD_ANCHORED | 2267 | 0.0621 | 0.4517 | 0.0048 | 10.1% | 3.2% | 5.5% | +0.0010 |

| Drop Effect | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drop | 0.0573 | 0.4356 | 0.0010 | 12.1% | 3.4% | 5.6% | +0.0257 |

Commodity candidate set carried into Iteration 2:

| Ticker | Current | Best Source | Classification | Recommendation |
| --- | --- | --- | --- | --- |
| SB | RAD | NON | exclusion candidate | drop |
| KC | RAD | NON | exclusion candidate | drop |
| ZL | RAD | NON | exclusion candidate | drop |
| NR | RAD | NON | source distortion | NON |
| ZC | RAD | NON | source distortion | NON |

## Iteration 2 — Commodity Local Combination Search

| Candidate | Ann Gap | MDD Gap | Calmar Gap | E(R) Err | % +ve Err | Ave P/L Err |
| --- | --- | --- | --- | --- | --- | --- |
| drop SB, drop KC, drop ZL, NR:RAD->NON, ZC:RAD->NON | 0.0097 | 0.1834 | 0.0327 | 33.6% | 4.4% | 4.8% |
| drop SB, drop KC, drop ZL, NR:RAD->NON | 0.0161 | 0.2151 | 0.0257 | 28.5% | 4.0% | 4.8% |
| drop SB, drop KC, NR:RAD->NON, ZC:RAD->NON | 0.0162 | 0.2231 | 0.0272 | 28.2% | 4.7% | 5.7% |
| drop SB, drop ZL, NR:RAD->NON, ZC:RAD->NON | 0.0176 | 0.2410 | 0.0281 | 29.2% | 3.8% | 4.1% |
| drop SB, drop KC, drop ZL, ZC:RAD->NON | 0.0178 | 0.2217 | 0.0235 | 27.5% | 4.2% | 5.1% |
| drop SB, drop KC, NR:RAD->NON | 0.0226 | 0.2561 | 0.0212 | 23.5% | 4.2% | 5.5% |
| drop SB, NR:RAD->NON, ZC:RAD->NON | 0.0238 | 0.2783 | 0.0235 | 24.2% | 4.7% | 6.4% |
| drop SB, drop ZL, NR:RAD->NON | 0.0240 | 0.2736 | 0.0221 | 24.5% | 3.0% | 3.4% |
| drop SB, drop KC, ZC:RAD->NON | 0.0243 | 0.2625 | 0.0192 | 22.1% | 4.4% | 6.1% |
| drop SB, drop KC, drop ZL | 0.0246 | 0.2563 | 0.0173 | 22.5% | 3.8% | 4.8% |
| drop SB, drop ZL, ZC:RAD->NON | 0.0258 | 0.2795 | 0.0200 | 23.5% | 3.0% | 3.4% |
| drop SB, NR:RAD->NON | 0.0303 | 0.3095 | 0.0178 | 19.8% | 4.0% | 5.7% |

Commodity-cleaned baseline: `drop SB, drop KC, drop ZL, NR:RAD->NON, ZC:RAD->NON`

## Iteration 3 — Global Numerator Re-Audit

| Candidate | Mean Annual Return Gap | Mean Calmar Gap | Worst Annual Return Gap | Worst Calmar Gap |
| --- | --- | --- | --- | --- |
| annual_mean_simple | 14.8% | 21.3% | 19.3% | 31.8% |
| wealth_cagr | 17.0% | 18.9% | 30.2% | 30.4% |
| annual_mean_log | 18.4% | 19.2% | 33.0% | 31.9% |
| annual_mean_sleeve | 65.5% | 54.4% | 234.3% | 177.2% |

Global numerator winner: `annual_mean_simple`

## Iteration 4 — Reporting Extraction / Aggregation Audit

Notes:

- On the 4 asset rows, `asset-equal` / `asset-count-weighted` aggregation is not yet meaningful; those are deferred to the future `All` audit.
- This stage therefore audits only same-path extraction variants inside the frozen per-asset reporting path family.

| Path Mode | Return Extraction | Mean Annual Return Gap | Mean Calmar Gap | Worst Annual Return Gap | Worst Calmar Gap |
| --- | --- | --- | --- | --- | --- |
| contract_equal_path | annual_mean_simple | 14.8% | 21.3% | 19.3% | 31.8% |
| contract_equal_path | wealth_cagr | 17.0% | 18.9% | 30.2% | 30.4% |
| contract_equal_path | annual_mean_log | 18.4% | 19.2% | 33.0% | 31.9% |
| sleeve_first_simple_path | wealth_cagr | 36.3% | 35.6% | 111.3% | 104.2% |
| sleeve_first_simple_path | annual_mean_log | 37.7% | 36.2% | 111.3% | 104.1% |
| sleeve_first_simple_path | annual_mean_simple | 65.5% | 47.6% | 234.3% | 149.2% |

## Iteration 5 — `All` Consistency Audit

- status: `ready to open next`

## Final Diagnosis

- final classification: `MDD aligned, numerator wrong`
- next single recommended action: `change default numerator to global winner on the cleaned baseline`

Current cleaned baseline after Commodity iteration:

| Asset | Trade E(R) | Reporting Annual Return | Paper-Implied Annual Return | MDD | Paper MDD | Calmar | Paper Calmar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Commodity | -0.198 | -0.042 | -0.032 | 0.431 | 0.248 | -0.097 | -0.130 |
| Equity Index | +0.523 | +0.047 | +0.059 | 0.146 | 0.127 | +0.324 | +0.466 |
| Fixed Income | +0.555 | +0.046 | +0.049 | 0.111 | 0.108 | +0.414 | +0.455 |
| Forex | -0.173 | -0.020 | -0.022 | 0.220 | 0.219 | -0.090 | -0.101 |

Assumptions used in this iteration:

- candidate source changes in Iteration 1 are only promoted when the alternative has the same test-window row count as the current source
- commodity local search only touches `SB, KC, ZL, NR, ZC`
- `REV` remains reference-only for the known negative-price-sensitive problem contracts
- `All` remains deferred until the 4 asset rows stabilize

