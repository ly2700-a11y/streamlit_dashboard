# DQN Validation (Asset-Class Shared + 9-Feature State Space)

This note records the minimum non-training validation flow.

## 1) Verify state schema and backtest long metrics

```bash
python drl/dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN
```

Checks:
- state feature shape is `(N, 9)`
- one sampled window shape is `(60, 9)`
- features are finite
- Table 3 long-only backtest returns include `MDD` and `Calmar`

## 2) Verify prepared contract data

```bash
python drl_shared/prepare_features.py --ticker AN --round 1
python drl/dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN --require-prepared
```

Checks:
- prepared `.npz` exists at `drl/features/AN/r1.npz`
- `prices/returns/sigma/features/source` match recomputed values

## 3) Verify checkpoint existence (after training)

```bash
python drl/dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN --require-prepared --require-checkpoint
```

Checks:
- checkpoint exists at `drl/dqn/models/<asset_class>/r<round>/<run_id>/checkpoint.pt`

## 4) Manual smoke for universal backtester

```bash
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy Long --asset Forex
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy DQN --asset Forex
```

Notes:
- `Long` should run without checkpoints.
- `DQN` requires one asset-class checkpoint per round for the selected asset class.
