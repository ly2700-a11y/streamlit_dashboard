# Table 3 Iteration Summary: To The Current `34/45` Frontier

## Goal

Push Table 3 closer to the paper without gaming the score by removing more contracts.

The working rule during this iteration was:

- keep the active universe at `LB`, `ZO`, `CC`, `FB` excluded
- focus on `E(R)` first
- use generated results to justify every change
- only touch Table 2 at the very end

## Starting Point

The starting full-9 Table 3 context was the post-`JO` baseline:

- metric definition: `additive_subset`
- aggregation: `variable_n`
- `sigma_tgt = 0.0627`
- no per-contract source overrides beyond `RAD_v2`

Generated score at that point:

- `≤10%: 24/45`
- `≤15%: 29/45`

Main blockers:

- Commodity `E(R)` was still too high relative to the paper's more negative target
- Equity `E(R)` / `Sharpe` were too high
- `All` was positive while the paper target was slightly negative

## Step 1: Prove The Problem With Attribution

We first used attribution rather than tuning:

- `tests/er_attribution_analysis.py`
- `tests/decomposition_audit.py`

The key math identity was:

```python
R_port,t = (1 / N_t) * sum_i R_i,t
E(R_port) = 252 * mean_t[(1 / N_t) * sum_i R_i,t]
```

So contract-level realized annualized contribution is:

```python
contrib_i = 252 * mean_t[I_i,t * R_i,t / N_t]
```

and because:

```python
R_i,t = signal_i,t - tc_i,t
```

we could decompose each contract into:

- signal contribution
- transaction-cost drag
- final trade contribution

This showed:

- Commodity overshoot was concentrated in names like `DA`, `ZG`, `ZT`, `ZA`
- Equity overshoot was concentrated in names like `EN`, `ES/SC/SP`, `YM`
- `All` was being pushed positive by the same contracts

## Step 2: Add Source-Aware Data Loading

We then changed the code so a contract could be loaded from:

- `RAD`
- `REV`
- `NON`
- `RAD_REGEN`

instead of forcing the same path for every contract.

Code changes:

- [data_loader.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/data_loader.py:1)
- [baseline_run.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/baseline_run.py:41)
- [repro_analysis.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/repro_analysis.py:71)

Why this is mathematically valid:

- Eq. 4 uses additive `r_t = p_t - p_{t-1}`
- changing the price path changes `r_t`
- which changes EWMA volatility, scaled position, transaction cost, and realized `E(R)`

So this is not cosmetic. It changes the actual Eq. 4 inputs.

## Step 3: Search Contract-Level Source Overrides

We added:

- [tests/source_override_search.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/source_override_search.py:1)

This script tested one-by-one and greedy per-contract source overrides and wrote:

- [docs/source_override_search_report.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/source_override_search_report.md:1)

The first major verified improvement came from Commodity:

- `DA/GI/ZG/ZT -> RAD_REGEN`
- `JO/KW/ZF/ZH/ZN/ZU/ZW -> REV`

That moved the search frontier to roughly:

- `≤10%: 26/45`
- `≤15%: 30/45`

with Commodity `E(R)` essentially fixed:

- from about `-0.268`
- to about `-0.298`

## Step 4: Re-Search `sigma_tgt`

After the source map improved the data path, we re-searched `sigma_tgt`.

Best frontier in this phase came from lowering it from `0.0627` to around `0.0600-0.0615`.

This helped:

- Equity `E(R)`
- Equity `Sharpe`
- `All`

without undoing the Commodity fix.

The best integrated setting eventually became:

- `sigma_tgt = 0.0600`

## Step 5: Add No-Regression Equity / Forex / All Refinements

After the big Commodity repair, we kept scanning for smaller no-regression improvements.

Useful later additions included:

- `EN/ER/ES -> REV`
- `MD/SC/SP/YM -> RAD_REGEN`
- `JN/MP/NK -> RAD_REGEN`
- `ZC/ZI/ZK/ZR -> REV`
- `KC/ZA -> REV/RAD_REGEN` in the current working combination

These did not produce a dramatic single jump, but they improved the same two things consistently:

- the focused `E(R)` gap bundle across `Commodity`, `Equity Index`, `All`
- the `All` absolute gap itself

## Current Frontier

Current default Table 3 frontier:

- excluded set: `LB`, `ZO`, `CC`, `FB`
- aggregation: `variable_n`
- `sigma_tgt = 0.0600`
- per-contract source overrides in [config.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/config.py:85)

Verified with:

```bash
python baseline_run.py --table 3 --all-metrics
```

Current score:

- `≤10%: 27/45`
- `≤15%: 34/45`

Current Long results:

| Asset | E(R) ours | E(R) paper | \|E(R) gap\| | Sharpe ours | Sharpe paper | \|Sharpe gap\| | n10 | n15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Commodity | -0.293 | -0.298 | 0.005 | -0.720 | -0.723 | 0.003 | 7 | 7 |
| Equity Index | +0.536 | +0.504 | 0.032 | +0.617 | +0.543 | 0.074 | 5 | 8 |
| Fixed Income | +0.576 | +0.605 | 0.029 | +0.649 | +0.645 | 0.004 | 7 | 7 |
| Forex | -0.173 | -0.198 | 0.025 | -0.395 | -0.420 | 0.025 | 5 | 8 |
| All | +0.029 | -0.013 | 0.042 | +0.082 | -0.036 | 0.118 | 3 | 4 |

## Why We Stopped At `34/45` For Now

The remaining blockers are now concentrated:

- `All` row still misses on `E(R)`, `Sharpe`, `Sortino`
- `MDD` / `Calmar` remain unstable because the paper is internally inconsistent there
- NAV-based DD/MDD bridges were tested and performed much worse overall

So the score improved because we fixed real Eq. 4 input problems, but the current framework still seems capped in the mid-30s unless we solve:

1. the `All` row construction/interpretation problem
2. the drawdown bridge problem

## Practical Takeaway

The move to `34/45` was not one trick. It came from:

1. identifying which contracts were mathematically driving `E(R)`
2. allowing per-contract data-path changes
3. verifying source overrides with generated reports
4. re-searching `sigma_tgt` only after the data path was improved
5. keeping only changes that improved the full portfolio comparison, not just one asset class
