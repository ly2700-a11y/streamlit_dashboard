# Historical 36/40 Rebuild Search

Goal:

- start from the historical `36/40` strong baseline
- rebuild it under the newer source understanding
- enforce `% +ve` and `Ave P/L` <= 2% on each of the 4 asset rows
- only if that fails, trigger a separate `Calmar` / reporting-return formula line

## Step A: Historical Strong Baseline vs Rebuilt Baseline

| Scenario | Hard 2% OK? | <=15 /40 | <=10 /40 | Avg E(R) Err | Avg Helper Gap | Avg MDD Err |
| --- | --- | --- | --- | --- | --- | --- |
| Historical 36/40 skeleton | no | 36/40 | 31/40 | 6.17 | 0.0088 | 7.37 |
| Rebuilt baseline | no | 35/40 | 30/40 | 8.68 | 0.0200 | 50.11 |

Historical 36/40 skeleton detail:

- overrides: `EN:REV`, `DT:REV`, `CC:RAD_REGEN`, `LB:REV`, `JO:REV`, `ZH:REV`
- excluded: `FB, ZA, ZO`

Rebuilt baseline detail:

- overrides: `EN:REV`, `DT:REV`, `CC:RAD_REGEN`, `LB:RAD`, `JO:RAD_REGEN`, `ZH:RAD_REGEN`
- excluded: `FB, ZA, ZO`

Rebuilt baseline 2% violations:

- Commodity % +ve 3.17%; Commodity Ave P/L 5.47%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54%

## Step A.1: Problem-Contract Four-Table Audit

Allowed sources locked for the search:

- `CC`: `RAD_REGEN`, `NON_FWD_ANCHORED`
- `LB`: `RAD`, `NON_FWD_ANCHORED`, `RAD_REGEN`
- `JO`: `RAD_REGEN`
- `ZH`: `RAD_REGEN`
- `ZO`: `RAD`, `RAD_REGEN`, `NON_FWD_ANCHORED`

### CC in Commodity

Single-contract audit:

| Source | Allowed? | Rows | Trade E(R) | Helper | MDD | Calmar |
| --- | --- | --- | --- | --- | --- | --- |
| RAD | reference | 2266 | -0.036 | -0.007 | 0.588 | -0.013 |
| REV | reference | 12 | +nan | N/A | N/A | N/A |
| RAD_REGEN | yes | 2266 | -0.039 | -0.009 | 0.570 | -0.015 |
| NON_FWD_ANCHORED | yes | 2266 | -0.029 | -0.005 | 0.581 | -0.009 |

Asset-row impact:

| Source | Allowed? | Asset E(R) | Asset Helper | |Helper Gap| | Asset MDD | Asset Calmar | Score <=15 /9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | reference | -0.267 | -0.094 | 0.0622 | 0.700 | -0.135 | 7 |
| REV | reference | -0.278 | -0.099 | 0.0672 | 0.717 | -0.139 | 7 |
| RAD_REGEN | yes | -0.268 | -0.095 | 0.0623 | 0.700 | -0.135 | 7 |
| NON_FWD_ANCHORED | yes | -0.267 | -0.094 | 0.0620 | 0.700 | -0.135 | 7 |

### LB in Commodity

Single-contract audit:

| Source | Allowed? | Rows | Trade E(R) | Helper | MDD | Calmar |
| --- | --- | --- | --- | --- | --- | --- |
| RAD | yes | 2264 | +0.153 | +0.039 | 0.796 | +0.049 |
| REV | reference | 599 | +0.401 | +0.409 | 0.853 | +0.479 |
| RAD_REGEN | yes | 2264 | +0.087 | +0.026 | 0.878 | +0.029 |
| NON_FWD_ANCHORED | yes | 2264 | +0.151 | +0.036 | 0.749 | +0.048 |

Asset-row impact:

| Source | Allowed? | Asset E(R) | Asset Helper | |Helper Gap| | Asset MDD | Asset Calmar | Score <=15 /9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | yes | -0.268 | -0.095 | 0.0623 | 0.700 | -0.135 | 7 |
| REV | reference | -0.282 | -0.045 | 0.0130 | 0.200 | -0.227 | 7 |
| RAD_REGEN | yes | -0.271 | -0.096 | 0.0642 | 0.707 | -0.136 | 7 |
| NON_FWD_ANCHORED | yes | -0.268 | -0.095 | 0.0627 | 0.702 | -0.135 | 7 |

### JO in Commodity

Single-contract audit:

| Source | Allowed? | Rows | Trade E(R) | Helper | MDD | Calmar |
| --- | --- | --- | --- | --- | --- | --- |
| RAD | reference | 2267 | -0.043 | -0.022 | 0.542 | -0.041 |
| REV | reference | 2129 | -0.140 | +nan | 1.022 | +nan |
| RAD_REGEN | yes | 2267 | +0.013 | -0.001 | 0.505 | -0.002 |
| NON_FWD_ANCHORED | reference | 2267 | -0.059 | -0.030 | 0.571 | -0.052 |

Asset-row impact:

| Source | Allowed? | Asset E(R) | Asset Helper | |Helper Gap| | Asset MDD | Asset Calmar | Score <=15 /9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | reference | -0.270 | -0.096 | 0.0638 | 0.707 | -0.136 | 7 |
| REV | reference | -0.274 | -0.123 | 0.0910 | 0.670 | -0.184 | 7 |
| RAD_REGEN | yes | -0.268 | -0.095 | 0.0623 | 0.700 | -0.135 | 7 |
| NON_FWD_ANCHORED | reference | -0.271 | -0.097 | 0.0644 | 0.710 | -0.136 | 7 |

### ZH in Commodity

Single-contract audit:

| Source | Allowed? | Rows | Trade E(R) | Helper | MDD | Calmar |
| --- | --- | --- | --- | --- | --- | --- |
| RAD | reference | 2267 | -0.247 | -0.068 | 0.864 | -0.079 |
| REV | reference | 986 | -0.374 | +nan | 1.358 | +nan |
| RAD_REGEN | yes | 2267 | -0.113 | -0.027 | 0.727 | -0.037 |
| NON_FWD_ANCHORED | reference | 2267 | -0.249 | -0.068 | 0.865 | -0.079 |

Asset-row impact:

| Source | Allowed? | Asset E(R) | Asset Helper | |Helper Gap| | Asset MDD | Asset Calmar | Score <=15 /9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | reference | -0.274 | -0.097 | 0.0651 | 0.710 | -0.137 | 7 |
| REV | reference | -0.277 | -0.112 | 0.0793 | 0.427 | -0.261 | 7 |
| RAD_REGEN | yes | -0.268 | -0.095 | 0.0623 | 0.700 | -0.135 | 7 |
| NON_FWD_ANCHORED | reference | -0.274 | -0.097 | 0.0650 | 0.710 | -0.137 | 7 |

### ZO in Commodity

Single-contract audit:

| Source | Allowed? | Rows | Trade E(R) | Helper | MDD | Calmar |
| --- | --- | --- | --- | --- | --- | --- |
| RAD | yes | 2267 | +0.045 | +0.010 | 0.659 | +0.015 |
| REV | reference | 2211 | +0.071 | +0.044 | 1.216 | +0.037 |
| RAD_REGEN | yes | 2267 | +0.057 | +0.013 | 0.692 | +0.019 |
| NON_FWD_ANCHORED | yes | 2267 | +0.019 | +0.004 | 0.677 | +0.005 |

Asset-row impact:

| Source | Allowed? | Asset E(R) | Asset Helper | |Helper Gap| | Asset MDD | Asset Calmar | Score <=15 /9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAD | yes | -0.268 | -0.095 | 0.0623 | 0.700 | -0.135 | 7 |
| REV | reference | -0.268 | -0.095 | 0.0623 | 0.700 | -0.135 | 7 |
| RAD_REGEN | yes | -0.268 | -0.095 | 0.0623 | 0.700 | -0.135 | 7 |
| NON_FWD_ANCHORED | yes | -0.268 | -0.095 | 0.0623 | 0.700 | -0.135 | 7 |

## Step B: Source-Only Frontier

Accepted candidates (all 4 asset rows satisfy `% +ve` and `Ave P/L` <= 2%):

| Candidate | Hard 2% OK? | <=15 /40 | <=10 /40 | Avg E(R) Err | Avg Helper Gap | Avg MDD Err |
| --- | --- | --- | --- | --- | --- | --- |
| none | - | - | - | - | - | - |

Rejected top candidates (failed the 2% rule):

| Candidate | Hard 2% OK? | <=15 /40 | <=10 /40 | Avg E(R) Err | Avg Helper Gap | Avg MDD Err | Violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LB:RAD->RAD_REGEN | no | 35/40 | 31/40 | 8.43 | 0.0205 | 50.82 | Commodity % +ve 3.38%; Commodity Ave P/L 5.88%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| CC:RAD_REGEN->NON_FWD_ANCHORED, LB:RAD->RAD_REGEN | no | 35/40 | 31/40 | 8.51 | 0.0204 | 50.82 | Commodity % +ve 3.17%; Commodity Ave P/L 5.67%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| DT:REV->RAD, LB:RAD->RAD_REGEN | no | 35/40 | 30/40 | 8.68 | 0.0221 | 51.74 | Commodity % +ve 3.38%; Commodity Ave P/L 5.88%; Fixed Income % +ve 3.50%; Fixed Income Ave P/L 7.06% |
| base | no | 35/40 | 30/40 | 8.68 | 0.0200 | 50.11 | Commodity % +ve 3.17%; Commodity Ave P/L 5.47%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| LB:RAD->NON_FWD_ANCHORED | no | 35/40 | 30/40 | 8.68 | 0.0201 | 50.31 | Commodity % +ve 3.17%; Commodity Ave P/L 5.47%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| DT:REV->RAD, CC:RAD_REGEN->NON_FWD_ANCHORED, LB:RAD->RAD_REGEN | no | 35/40 | 30/40 | 8.76 | 0.0220 | 51.74 | Commodity % +ve 3.17%; Commodity Ave P/L 5.67%; Fixed Income % +ve 3.50%; Fixed Income Ave P/L 7.06% |
| CC:RAD_REGEN->NON_FWD_ANCHORED | no | 35/40 | 30/40 | 8.77 | 0.0199 | 50.11 | Commodity % +ve 2.96%; Commodity Ave P/L 5.27%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| CC:RAD_REGEN->NON_FWD_ANCHORED, LB:RAD->NON_FWD_ANCHORED | no | 35/40 | 30/40 | 8.77 | 0.0200 | 50.21 | Commodity % +ve 2.96%; Commodity Ave P/L 5.27%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| DT:REV->RAD | no | 35/40 | 29/40 | 8.93 | 0.0216 | 51.04 | Commodity % +ve 3.17%; Commodity Ave P/L 5.47%; Fixed Income % +ve 3.50%; Fixed Income Ave P/L 7.06% |
| DT:REV->RAD, LB:RAD->NON_FWD_ANCHORED | no | 35/40 | 29/40 | 8.93 | 0.0217 | 51.24 | Commodity % +ve 3.17%; Commodity Ave P/L 5.47%; Fixed Income % +ve 3.50%; Fixed Income Ave P/L 7.06% |

Best source-only base used for the next step:

- `LB:RAD->RAD_REGEN`
- hard 2% ok: `no`

## Step C: Source + Exclusion Frontier

Accepted candidates (all 4 asset rows satisfy `% +ve` and `Ave P/L` <= 2%):

| Excluded | Hard 2% OK? | <=15 /40 | <=10 /40 | Avg E(R) Err | Avg Helper Gap | Avg MDD Err |
| --- | --- | --- | --- | --- | --- | --- |
| none | - | - | - | - | - | - |

Rejected top candidates (failed the 2% rule):

| Excluded | Hard 2% OK? | <=15 /40 | <=10 /40 | Avg E(R) Err | Avg Helper Gap | Avg MDD Err | Violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CC,FB,JO,LB,ZA | no | 35/40 | 31/40 | 6.25 | 0.0244 | 57.07 | Commodity Ave P/L 2.74%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| CC,FB,LB,ZA,ZO | no | 35/40 | 31/40 | 6.25 | 0.0246 | 57.17 | Commodity Ave P/L 3.14%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| CC,FB,JO,ZA,ZO | no | 35/40 | 31/40 | 6.42 | 0.0238 | 55.96 | Commodity % +ve 3.59%; Commodity Ave P/L 6.69%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| FB,LB,ZA,ZH,ZO | no | 35/40 | 31/40 | 6.42 | 0.0245 | 56.87 | Commodity Ave P/L 4.26%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| FB,JO,LB,ZA,ZO | no | 35/40 | 31/40 | 6.42 | 0.0253 | 58.48 | Commodity Ave P/L 3.65%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| FB,JO,LB,ZA,ZH | no | 35/40 | 31/40 | 6.50 | 0.0243 | 56.77 | Commodity Ave P/L 2.84%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| FB,JO,ZA,ZH,ZO | no | 35/40 | 31/40 | 6.67 | 0.0237 | 55.86 | Commodity % +ve 2.11%; Commodity Ave P/L 4.46%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| CC,FB,JO,ZA,ZH,ZO | no | 35/40 | 31/40 | 6.67 | 0.0254 | 57.88 | Commodity % +ve 2.33%; Commodity Ave P/L 5.17%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| CC,FB,JO,LB,ZH,ZO | no | 35/40 | 31/40 | 6.75 | 0.0229 | 55.96 | Commodity Ave P/L 3.04%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |
| CC,FB,LB,ZA,ZH | no | 35/40 | 31/40 | 6.75 | 0.0236 | 55.36 | Commodity Ave P/L 3.65%; Fixed Income % +ve 3.69%; Fixed Income Ave P/L 7.54% |

Best source+exclusion candidate:

- source base: `LB:RAD->RAD_REGEN`
- excluded: `CC,FB,JO,LB,ZA`
- hard 2% ok: `no`
- score: `<=15 35/40`, `<=10 31/40`

Asset-row payload of the best source+exclusion candidate:

| Asset | <=15 /8 | <=10 /8 | E(R) | MDD | Helper | Paper Helper | % +ve | Ave P/L | % +ve Err | Ave P/L Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Commodity | 7 | 7 | -0.297 | 0.769 | -0.112 | -0.032 | 0.480 | 0.960 | 1.48% | 2.74% |
| Equity Index | 8 | 6 | +0.523 | 0.146 | +0.047 | +0.059 | 0.547 | 0.920 | 1.11% | 0.86% |
| Fixed Income | 8 | 8 | +0.555 | 0.111 | +0.046 | +0.049 | 0.534 | 0.969 | 3.69% | 7.54% |
| Forex | 8 | 6 | -0.173 | 0.220 | -0.020 | -0.022 | 0.490 | 0.972 | 0.20% | 0.62% |

## Step D: Calmar Trigger Status

- trigger Calmar / reporting-return formula line: `yes`

Reason:

- under the rebuilt, source-clean doctrine, no candidate satisfied the per-asset `% +ve` and `Ave P/L` <= 2% rule, or the frontier remained clearly capped after satisfying the trade-side source constraints
- this means the bottleneck is no longer just trade-side data cleaning; the reporting-return / Calmar line needs a separate audit

