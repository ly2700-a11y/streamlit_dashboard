# Frontier 40+ Enumeration

Unified workflow:

1. push the clean same-rule line as far as it goes
2. if still below `40/45`, enumerate coherent `40+ first` frontiers
3. summarize interpretation cost vs score under one scorecard

## Family Winners

| Family | <=10 | <=15 | 4-asset <=15 | Mean Ann Gap | Mean Cal Gap | All Blocker Removed? | Label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean same-rule | 29/45 | 34/45 | 30/36 | 17.0% | 18.9% | Partly | clean / wealth_cagr / contract_equal_path / contract_equal_path |
| coherent override | 31/45 | 38/45 | 34/36 | 61.3% | 14.5% | Partly | structural_history / Equity Index:risk_price_non / wealth_cagr / contract_equal_path |
| structural-heavy | 31/45 | 38/45 | 34/36 | 61.3% | 14.5% | Partly | history_seed / Equity:risk_price_non / wealth_cagr / contract_equal_path |
| legacy experimental upper bound | 36/45 | 41/45 | 34/36 | 66.8% | 71.1% | Partly | legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / contract_equal_path |

## Clean Same-Rule Final Push

| Label | <=10 | <=15 | 4-asset <=15 | Mean Ann Gap | Mean Cal Gap | Capital | Numerator | All Mode | 4-asset Remaining Misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean / wealth_cagr / contract_equal_path / contract_equal_path | 29/45 | 34/45 | 30/36 | 17.0% | 18.9% | risk_price_source | wealth_cagr | contract_equal_path | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar |
| clean / wealth_cagr / contract_equal_path / asset_equal_path | 29/45 | 34/45 | 30/36 | 17.0% | 18.9% | risk_price_source | wealth_cagr | asset_equal_path | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar |
| clean / wealth_cagr / contract_equal_path / asset_count_weighted_path | 29/45 | 34/45 | 30/36 | 17.0% | 18.9% | risk_price_source | wealth_cagr | asset_count_weighted_path | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar |
| clean / wealth_cagr / contract_equal_path / asset_equal_simple | 29/45 | 34/45 | 30/36 | 17.0% | 18.9% | risk_price_source | wealth_cagr | asset_equal_simple | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar |
| clean / wealth_cagr / contract_equal_path / asset_count_weighted_simple | 29/45 | 34/45 | 30/36 | 17.0% | 18.9% | risk_price_source | wealth_cagr | asset_count_weighted_simple | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar |
| clean / annual_mean_log / contract_equal_path / contract_equal_path | 29/45 | 34/45 | 30/36 | 18.4% | 19.2% | risk_price_source | annual_mean_log | contract_equal_path | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar |
| clean / annual_mean_log / contract_equal_path / asset_equal_path | 29/45 | 34/45 | 30/36 | 18.4% | 19.2% | risk_price_source | annual_mean_log | asset_equal_path | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar |
| clean / annual_mean_log / contract_equal_path / asset_count_weighted_path | 29/45 | 34/45 | 30/36 | 18.4% | 19.2% | risk_price_source | annual_mean_log | asset_count_weighted_path | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar |

## Coherent Override Frontiers

| Label | <=10 | <=15 | 4-asset <=15 | Mean Ann Gap | Mean Cal Gap | Capital | Numerator | All Mode | 4-asset Remaining Misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| structural_history / Equity Index:risk_price_non / wealth_cagr / contract_equal_path | 31/45 | 38/45 | 34/36 | 61.3% | 14.5% | Equity Index->risk_price_non | wealth_cagr | contract_equal_path | Commodity: MDD ; Equity Index: Calmar |
| structural_history / Equity Index:risk_price_non / wealth_cagr / asset_equal_path | 31/45 | 38/45 | 34/36 | 61.3% | 14.5% | Equity Index->risk_price_non | wealth_cagr | asset_equal_path | Commodity: MDD ; Equity Index: Calmar |
| structural_history / Equity Index:risk_price_non / wealth_cagr / asset_count_weighted_path | 31/45 | 38/45 | 34/36 | 61.3% | 14.5% | Equity Index->risk_price_non | wealth_cagr | asset_count_weighted_path | Commodity: MDD ; Equity Index: Calmar |
| structural_history / Equity Index:risk_price_non / annual_mean_log / contract_equal_path | 31/45 | 38/45 | 34/36 | 65.6% | 16.4% | Equity Index->risk_price_non | annual_mean_log | contract_equal_path | Commodity: MDD ; Equity Index: Calmar |
| structural_history / Equity Index:risk_price_non / annual_mean_log / asset_equal_path | 31/45 | 38/45 | 34/36 | 65.6% | 16.4% | Equity Index->risk_price_non | annual_mean_log | asset_equal_path | Commodity: MDD ; Equity Index: Calmar |
| structural_history / Equity Index:risk_price_non / annual_mean_log / asset_count_weighted_path | 31/45 | 38/45 | 34/36 | 65.6% | 16.4% | Equity Index->risk_price_non | annual_mean_log | asset_count_weighted_path | Commodity: MDD ; Equity Index: Calmar |
| structural_history / Equity Index:risk_price_non / annual_mean_simple / contract_equal_path | 31/45 | 37/45 | 33/36 | 59.0% | 14.5% | Equity Index->risk_price_non | annual_mean_simple | contract_equal_path | Commodity: MDD ; Equity Index: Calmar ; Forex: Calmar |
| structural_history / Equity Index:risk_price_non / annual_mean_simple / asset_equal_path | 31/45 | 37/45 | 33/36 | 59.0% | 14.5% | Equity Index->risk_price_non | annual_mean_simple | asset_equal_path | Commodity: MDD ; Equity Index: Calmar ; Forex: Calmar |
| structural_history / Equity Index:risk_price_non / annual_mean_simple / asset_count_weighted_path | 31/45 | 37/45 | 33/36 | 59.0% | 14.5% | Equity Index->risk_price_non | annual_mean_simple | asset_count_weighted_path | Commodity: MDD ; Equity Index: Calmar ; Forex: Calmar |
| structural_history / Commodity:risk_price_non / wealth_cagr / contract_equal_path | 29/45 | 36/45 | 32/36 | 154.3% | 31.0% | Commodity->risk_price_non | wealth_cagr | contract_equal_path | Commodity: MDD, Calmar ; Equity Index: MDD, Calmar |

## Structural-Heavy Frontiers

| Label | <=10 | <=15 | 4-asset <=15 | Mean Ann Gap | Mean Cal Gap | Capital | Numerator | All Mode | 4-asset Remaining Misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| history_seed / Equity:risk_price_non / wealth_cagr / contract_equal_path | 31/45 | 38/45 | 34/36 | 61.3% | 14.5% | Equity Index->risk_price_non | wealth_cagr | contract_equal_path | Commodity: MDD ; Equity Index: Calmar |
| history_seed / wealth_cagr / contract_equal_path / contract_equal_path | 30/45 | 37/45 | 33/36 | 58.0% | 15.6% | risk_price_source | wealth_cagr | contract_equal_path | Commodity: MDD ; Equity Index: MDD, Calmar |
| history_seed / wealth_cagr / contract_equal_path / asset_equal_path | 30/45 | 37/45 | 33/36 | 58.0% | 15.6% | risk_price_source | wealth_cagr | asset_equal_path | Commodity: MDD ; Equity Index: MDD, Calmar |
| history_seed / wealth_cagr / contract_equal_path / asset_count_weighted_path | 30/45 | 37/45 | 33/36 | 58.0% | 15.6% | risk_price_source | wealth_cagr | asset_count_weighted_path | Commodity: MDD ; Equity Index: MDD, Calmar |
| history_seed / annual_mean_log / contract_equal_path / contract_equal_path | 30/45 | 37/45 | 33/36 | 62.4% | 17.5% | risk_price_source | annual_mean_log | contract_equal_path | Commodity: MDD ; Equity Index: MDD, Calmar |
| history_seed / annual_mean_log / contract_equal_path / asset_equal_path | 30/45 | 37/45 | 33/36 | 62.4% | 17.5% | risk_price_source | annual_mean_log | asset_equal_path | Commodity: MDD ; Equity Index: MDD, Calmar |
| history_seed / annual_mean_log / contract_equal_path / asset_count_weighted_path | 30/45 | 37/45 | 33/36 | 62.4% | 17.5% | risk_price_source | annual_mean_log | asset_count_weighted_path | Commodity: MDD ; Equity Index: MDD, Calmar |
| history_seed / annual_mean_simple / contract_equal_path / contract_equal_path | 30/45 | 36/45 | 32/36 | 55.6% | 15.5% | risk_price_source | annual_mean_simple | contract_equal_path | Commodity: MDD ; Equity Index: MDD, Calmar ; Forex: Calmar |
| history_seed / annual_mean_simple / contract_equal_path / asset_equal_path | 30/45 | 36/45 | 32/36 | 55.6% | 15.5% | risk_price_source | annual_mean_simple | asset_equal_path | Commodity: MDD ; Equity Index: MDD, Calmar ; Forex: Calmar |
| history_seed / annual_mean_simple / contract_equal_path / asset_count_weighted_path | 30/45 | 36/45 | 32/36 | 55.6% | 15.5% | risk_price_source | annual_mean_simple | asset_count_weighted_path | Commodity: MDD ; Equity Index: MDD, Calmar ; Forex: Calmar |

## Legacy Experimental Upper Bound

| Label | <=10 | <=15 | 4-asset <=15 | Mean Ann Gap | Mean Cal Gap | Capital | Numerator | All Mode | 4-asset Remaining Misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / contract_equal_path | 36/45 | 41/45 | 34/36 | 66.8% | 71.1% | Equity Index->risk_price_non | annual_mean_sleeve | contract_equal_path | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_equal_path | 36/45 | 41/45 | 34/36 | 66.8% | 71.1% | Equity Index->risk_price_non | annual_mean_sleeve | asset_equal_path | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_count_weighted_path | 36/45 | 41/45 | 34/36 | 66.8% | 71.1% | Equity Index->risk_price_non | annual_mean_sleeve | asset_count_weighted_path | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_equal_simple | 36/45 | 41/45 | 34/36 | 66.8% | 71.1% | Equity Index->risk_price_non | annual_mean_sleeve | asset_equal_simple | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_count_weighted_simple | 36/45 | 41/45 | 34/36 | 66.8% | 71.1% | Equity Index->risk_price_non | annual_mean_sleeve | asset_count_weighted_simple | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / contract_equal_path | 35/45 | 41/45 | 34/36 | 26.7% | 32.2% | Equity Index->risk_price_non | wealth_cagr | contract_equal_path | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / asset_equal_path | 35/45 | 41/45 | 34/36 | 26.7% | 32.2% | Equity Index->risk_price_non | wealth_cagr | asset_equal_path | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / asset_count_weighted_path | 35/45 | 41/45 | 34/36 | 26.7% | 32.2% | Equity Index->risk_price_non | wealth_cagr | asset_count_weighted_path | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / asset_equal_simple | 35/45 | 41/45 | 34/36 | 26.7% | 32.2% | Equity Index->risk_price_non | wealth_cagr | asset_equal_simple | Commodity: Calmar ; Equity Index: Calmar |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / asset_count_weighted_simple | 35/45 | 41/45 | 34/36 | 26.7% | 32.2% | Equity Index->risk_price_non | wealth_cagr | asset_count_weighted_simple | Commodity: Calmar ; Equity Index: Calmar |

## 40+ First Cases

| Frontier | Score <=10 /45 | Score <=15 /45 | Same-rule? | Asset-specific override? | Structural-heavy? | Main blocker removed? | Explanation cost | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / contract_equal_path | 36/45 | 41/45 | No | Yes | Yes | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_equal_path | 36/45 | 41/45 | No | Yes | Yes | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_count_weighted_path | 36/45 | 41/45 | No | Yes | Yes | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_equal_simple | 36/45 | 41/45 | No | Yes | Yes | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_count_weighted_simple | 36/45 | 41/45 | No | Yes | Yes | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / contract_equal_path | 35/45 | 41/45 | No | Yes | Yes | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / asset_equal_path | 35/45 | 41/45 | No | Yes | Yes | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / asset_count_weighted_path | 35/45 | 41/45 | No | Yes | Yes | Partly | High | Experimental upper bound only |

## Unified Candidate Payload

| Name | <=10 /45 | <=15 /45 | Trade-lane overrides | Exclusions | Reporting bridge / numerator / path / all | Same-rule? | Asset-specific? | Historical experimental? | 4-asset Remaining Misses | All still main blocker? | Explanation cost | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / contract_equal_path | 36/45 | 41/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:REV, LB:REV, ZH:REV | EN,ES,FB,ZA,ZO | risk_price_source | annual_mean_sleeve | contract_equal_path | contract_equal_path | No | Yes | Yes | Commodity: Calmar ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_equal_path | 36/45 | 41/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:REV, LB:REV, ZH:REV | EN,ES,FB,ZA,ZO | risk_price_source | annual_mean_sleeve | contract_equal_path | asset_equal_path | No | Yes | Yes | Commodity: Calmar ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_count_weighted_path | 36/45 | 41/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:REV, LB:REV, ZH:REV | EN,ES,FB,ZA,ZO | risk_price_source | annual_mean_sleeve | contract_equal_path | asset_count_weighted_path | No | Yes | Yes | Commodity: Calmar ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_equal_simple | 36/45 | 41/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:REV, LB:REV, ZH:REV | EN,ES,FB,ZA,ZO | risk_price_source | annual_mean_sleeve | contract_equal_path | asset_equal_simple | No | Yes | Yes | Commodity: Calmar ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / asset_count_weighted_simple | 36/45 | 41/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:REV, LB:REV, ZH:REV | EN,ES,FB,ZA,ZO | risk_price_source | annual_mean_sleeve | contract_equal_path | asset_count_weighted_simple | No | Yes | Yes | Commodity: Calmar ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / contract_equal_path | 35/45 | 41/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:REV, LB:REV, ZH:REV | EN,ES,FB,ZA,ZO | risk_price_source | wealth_cagr | contract_equal_path | contract_equal_path | No | Yes | Yes | Commodity: Calmar ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / asset_equal_path | 35/45 | 41/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:REV, LB:REV, ZH:REV | EN,ES,FB,ZA,ZO | risk_price_source | wealth_cagr | contract_equal_path | asset_equal_path | No | Yes | Yes | Commodity: Calmar ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| legacy_experimental / Equity:risk_price_non / wealth_cagr / asset_count_weighted_path | 35/45 | 41/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:REV, LB:REV, ZH:REV | EN,ES,FB,ZA,ZO | risk_price_source | wealth_cagr | contract_equal_path | asset_count_weighted_path | No | Yes | Yes | Commodity: Calmar ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| structural_history / Equity Index:risk_price_non / wealth_cagr / contract_equal_path | 31/45 | 38/45 | CC:RAD_REGEN, DT:REV, EN:RAD_REGEN, JO:RAD_REGEN, LB:RAD, ZH:RAD_REGEN | EN,ES,FB,ZA,ZO | risk_price_source | wealth_cagr | contract_equal_path | contract_equal_path | No | Yes | Yes | Commodity: MDD ; Equity Index: Calmar | Partly | High | Alternative candidate |
| history_seed / Equity:risk_price_non / wealth_cagr / contract_equal_path | 31/45 | 38/45 | CC:RAD_REGEN, DT:REV, EN:RAD_REGEN, JO:RAD_REGEN, LB:RAD, ZH:RAD_REGEN | EN,ES,FB,ZA,ZO | risk_price_source | wealth_cagr | contract_equal_path | contract_equal_path | No | Yes | Yes | Commodity: MDD ; Equity Index: Calmar | Partly | High | Experimental upper bound only |
| clean / wealth_cagr / contract_equal_path / contract_equal_path | 29/45 | 34/45 | CC:RAD_REGEN, DT:REV, EN:REV, JO:RAD_REGEN, LB:RAD, NR:NON, ZC:NON, ZH:RAD_REGEN | FB,KC,SB,ZA,ZL,ZO | risk_price_source | wealth_cagr | contract_equal_path | contract_equal_path | Yes | No | No | Commodity: E(R), Sharpe, Sortino, MDD, Calmar ; Equity Index: Calmar | Partly | Low | Primary clean solution |

## Overall Read

- clean same-rule max frontier: `clean / wealth_cagr / contract_equal_path / contract_equal_path` → `<=15 34/45`, `<=10 29/45`
- coherent override max frontier: `structural_history / Equity Index:risk_price_non / wealth_cagr / contract_equal_path` → `<=15 38/45`, `<=10 31/45`
- structural-heavy max frontier: `history_seed / Equity:risk_price_non / wealth_cagr / contract_equal_path` → `<=15 38/45`, `<=10 31/45`
- legacy experimental upper bound: `legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / contract_equal_path` → `<=15 41/45`, `<=10 36/45`

Final read:

- clean same-rule still does **not** reach `40+/45` under the current clean doctrine.
- the next review should therefore compare it against the `40+ first` cases above, not wait for more blind local search.
- under the current cleaner source doctrine, the only `40+` cases found here come from the legacy experimental upper-bound family.
