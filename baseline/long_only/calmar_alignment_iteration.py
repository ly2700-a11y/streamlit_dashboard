#!/usr/bin/env python3
"""
Iterative alignment audit for Table 3 Long reporting-world annual return / Calmar.

Stages:
  0. frozen baseline snapshot
  1. commodity suspect contract audit
  2. commodity local source/exclusion combination search
  3. global numerator re-audit on the cleaned baseline
  4. same-path reporting extraction audit if needed
  5. All consistency audit (deferred unless prerequisites are met)
"""
from __future__ import annotations

from functools import lru_cache
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import (  # noqa: E402
    build_reporting_portfolio_risk_price_sigma0,
    compute_portfolio_returns,
    load_contracts,
)
from config import ASSET_CLASSES, METRIC_NAMES, PAPER_TABLE3, SOURCE_OVERRIDES  # noqa: E402
from data_loader import load_clc_full  # noqa: E402
from metrics import compute_metrics, max_drawdown_from_path  # noqa: E402


SIGMA = 0.058
ASSETS = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
DOC_PATH = ROOT / "docs" / "calmar_alignment_iteration.md"
NUMERATOR_CANDIDATES = ["wealth_cagr", "annual_mean_simple", "annual_mean_log", "annual_mean_sleeve"]
EXTRACTION_CANDIDATES = [
    ("contract_equal_path", "wealth_cagr"),
    ("contract_equal_path", "annual_mean_simple"),
    ("contract_equal_path", "annual_mean_log"),
    ("sleeve_first_simple_path", "wealth_cagr"),
    ("sleeve_first_simple_path", "annual_mean_simple"),
    ("sleeve_first_simple_path", "annual_mean_log"),
]
BASE_EXCLUDED = {"FB", "ZA", "ZO"}
BASE_OVERRIDES = dict(SOURCE_OVERRIDES)
BASE_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
    }
)
BASE_OVERRIDES.pop("ZO", None)
COMMODITY_SUSPECTS = ["SB", "KC", "ZL", "NR", "ZC"]
PROBLEM_TICKERS = ["CC", "LB", "JO", "ZH", "ZO"]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def paper_helper(asset: str) -> float:
    paper = PAPER_TABLE3[asset]["Long"]
    return paper["Calmar"] * paper["MDD"]


def pct_err(value: float, paper: float) -> float:
    if not np.isfinite(value) or not np.isfinite(paper):
        return float("inf")
    if paper == 0:
        return abs(value - paper)
    return abs((value - paper) / abs(paper)) * 100.0


def annual_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean() * 252.0)


def path_mode_from_reporting(reporting: dict, path_mode: str):
    if path_mode == "contract_equal_path":
        return reporting
    if path_mode == "sleeve_first_simple_path":
        sleeves = reporting["sleeve_paths"]
        simple = np.full_like(sleeves, np.nan, dtype=float)
        simple[1:, :] = sleeves[1:, :] / sleeves[:-1, :] - 1.0
        daily = np.nanmean(simple[1:, :], axis=1)
        path = np.empty(len(daily) + 1, dtype=float)
        path[0] = 1.0
        path[1:] = np.cumprod(1.0 + daily)
        port_simple = np.full(len(path), np.nan, dtype=float)
        port_log = np.full(len(path), np.nan, dtype=float)
        if len(path) > 1:
            port_simple[1:] = path[1:] / path[:-1] - 1.0
            port_log[1:] = np.log(path[1:] / path[:-1])
        return {
            "portfolio_path": path,
            "portfolio_simple_returns": port_simple,
            "portfolio_log_returns": port_log,
            "sleeve_simple_returns": simple,
            "sleeve_paths": sleeves,
        }
    raise ValueError(path_mode)


def annual_return_from_reporting(reporting: dict, numerator_mode: str):
    port = reporting["portfolio_path"]
    if numerator_mode == "wealth_cagr":
        return float((port[-1] / port[0]) ** (252.0 / len(port)) - 1.0)
    if numerator_mode == "annual_mean_simple":
        return annual_mean(reporting["portfolio_simple_returns"][1:])
    if numerator_mode == "annual_mean_log":
        return annual_mean(reporting["portfolio_log_returns"][1:])
    if numerator_mode == "annual_mean_sleeve":
        return annual_mean(reporting["sleeve_simple_returns"][1:, :])
    raise ValueError(numerator_mode)


def load_asset_raw(asset: str, overrides: dict[str, str], excluded: set[str]):
    return load_contracts(asset, excluded_contracts=list(sorted(excluded)), source_overrides=overrides)


@lru_cache(maxsize=None)
def asset_state(asset: str, overrides_key: tuple[tuple[str, str], ...], excluded_key: tuple[str, ...], path_mode: str):
    overrides = dict(overrides_key)
    excluded = set(excluded_key)
    raw = load_asset_raw(asset, overrides, excluded)
    trade_returns = compute_portfolio_returns(raw, "Long", SIGMA, aggregation_mode="variable_n")
    trade = dict(zip(METRIC_NAMES, compute_metrics(trade_returns, n_contracts=len(raw))))
    reporting_base = build_reporting_portfolio_risk_price_sigma0(raw, SIGMA, strat="Long")
    reporting = path_mode_from_reporting(reporting_base, path_mode)
    mdd = float(max_drawdown_from_path(reporting["portfolio_path"]))
    paper = PAPER_TABLE3[asset]["Long"]
    cands = {}
    for name in NUMERATOR_CANDIDATES:
        ann = annual_return_from_reporting(reporting, name)
        cal = ann / mdd if mdd > 0 and np.isfinite(ann) else float("nan")
        cands[name] = {
            "annual_return": ann,
            "calmar": cal,
            "annual_return_err": pct_err(ann, paper_helper(asset)),
            "calmar_err": pct_err(cal, paper["Calmar"]),
            "annual_return_gap": abs(ann - paper_helper(asset)) if np.isfinite(ann) else float("inf"),
            "calmar_gap": abs(cal - paper["Calmar"]) if np.isfinite(cal) else float("inf"),
        }
    default = cands["wealth_cagr"]
    mdd_err = pct_err(mdd, paper["MDD"])
    hit_count = int(mdd_err < 15.0) + int(default["calmar_err"] < 15.0)
    return {
        "raw": raw,
        "trade": trade,
        "mdd": mdd,
        "mdd_err": mdd_err,
        "paper": paper,
        "paper_helper": paper_helper(asset),
        "candidates": cands,
        "hit_count": hit_count,
    }


def asset_summary_line(asset: str, overrides: dict[str, str], excluded: set[str], path_mode: str = "contract_equal_path"):
    state = asset_state(asset, tuple(sorted(overrides.items())), tuple(sorted(excluded)), path_mode)
    cur = state["candidates"]["wealth_cagr"]
    return [
        asset,
        f"{state['trade']['E(R)']:+.3f}",
        f"{cur['annual_return']:+.3f}",
        f"{state['paper_helper']:+.3f}",
        f"{state['mdd']:.3f}",
        f"{state['paper']['MDD']:.3f}",
        f"{cur['calmar']:+.3f}",
        f"{state['paper']['Calmar']:+.3f}",
    ]


def available_sources(ticker: str):
    out = []
    for source in ["RAD", "REV", "RAD_REGEN", "NON", "NON_FWD_ANCHORED"]:
        try:
            df = load_clc_full(ticker, source=source, anchor_date="2011-01-01")
        except Exception:
            df = None
        if df is not None:
            mask = (df["Date"] >= "2011-01-01") & (df["Date"] <= "2019-12-31")
            rows = int(mask.sum())
            if rows > 0:
                out.append((source, rows))
    return out


def classify_commodity_contract(source_score: float, drop_score: float, best_source: str | None, current_source: str):
    if max(source_score, drop_score) <= 0.002:
        return "noise", current_source
    if source_score > drop_score + 0.003 and best_source and best_source != current_source:
        return "source distortion", best_source
    if drop_score > source_score + 0.003:
        return "exclusion candidate", "drop"
    return "path-sensitive", best_source if best_source and best_source != current_source else "drop"


def commodity_objective(state: dict):
    cur = state["candidates"]["wealth_cagr"]
    paper = state["paper"]
    return (
        cur["annual_return_gap"],
        abs(state["mdd"] - paper["MDD"]),
        cur["calmar_gap"],
        pct_err(state["trade"]["E(R)"], paper["E(R)"]),
        pct_err(state["trade"]["% +ve"], paper["% +ve"]),
        pct_err(state["trade"]["Ave P/L"], paper["Ave P/L"]),
    )


def commodity_suspect_audit():
    rows = []
    candidate_set = []
    base_state = asset_state("Commodity", tuple(sorted(BASE_OVERRIDES.items())), tuple(sorted(BASE_EXCLUDED)), "contract_equal_path")
    base_obj = commodity_objective(base_state)
    for ticker in COMMODITY_SUSPECTS:
        current = BASE_OVERRIDES.get(ticker, "RAD")
        asset = "Commodity"
        current_rows = dict(available_sources(ticker)).get(current, 0)
        source_rows = []
        best_source = current
        best_source_score = float("-inf")
        for source, rows_count in available_sources(ticker):
            trial_overrides = dict(BASE_OVERRIDES)
            if source == "RAD":
                trial_overrides.pop(ticker, None)
            else:
                trial_overrides[ticker] = source
            trial = asset_state(asset, tuple(sorted(trial_overrides.items())), tuple(sorted(BASE_EXCLUDED)), "contract_equal_path")
            trial_obj = commodity_objective(trial)
            score = (base_obj[0] - trial_obj[0]) + (base_obj[1] - trial_obj[1]) + (base_obj[2] - trial_obj[2])
            source_rows.append((source, rows_count, trial_obj, score))
            if rows_count == current_rows and score > best_source_score:
                best_source_score = score
                best_source = source

        trial_excluded = set(BASE_EXCLUDED)
        trial_excluded.add(ticker)
        drop_state = asset_state(asset, tuple(sorted(BASE_OVERRIDES.items())), tuple(sorted(trial_excluded)), "contract_equal_path")
        drop_obj = commodity_objective(drop_state)
        drop_score = (base_obj[0] - drop_obj[0]) + (base_obj[1] - drop_obj[1]) + (base_obj[2] - drop_obj[2])
        cls, recommendation = classify_commodity_contract(best_source_score, drop_score, best_source, current)
        if cls in {"source distortion", "exclusion candidate", "path-sensitive"}:
            candidate_set.append(
                {
                    "ticker": ticker,
                    "current": current,
                    "best_source": best_source,
                    "classification": cls,
                    "recommendation": recommendation,
                }
            )
        rows.append(
            {
                "ticker": ticker,
                "current": current,
                "source_rows": source_rows,
                "drop_obj": drop_obj,
                "drop_score": drop_score,
                "best_source": best_source,
                "best_source_score": best_source_score,
                "classification": cls,
                "recommendation": recommendation,
            }
        )
    return base_state, rows, candidate_set


def commodity_local_search(candidate_set):
    option_map = {}
    for item in candidate_set:
        opts = [("keep", None)]
        if item["classification"] in {"source distortion", "path-sensitive"} and item["best_source"] != item["current"]:
            opts.append(("source", item["best_source"]))
        if item["classification"] in {"exclusion candidate", "path-sensitive"}:
            opts.append(("exclude", None))
        option_map[item["ticker"]] = opts

    if not option_map:
        state = asset_state("Commodity", tuple(sorted(BASE_OVERRIDES.items())), tuple(sorted(BASE_EXCLUDED)), "contract_equal_path")
        return [], {"overrides": dict(BASE_OVERRIDES), "excluded": set(BASE_EXCLUDED), "state": state, "label": "base"}

    tickers = list(option_map.keys())
    rows = []
    for combo in itertools.product(*[option_map[tk] for tk in tickers]):
        overrides = dict(BASE_OVERRIDES)
        excluded = set(BASE_EXCLUDED)
        labels = []
        for tk, (kind, value) in zip(tickers, combo):
            if kind == "keep":
                continue
            if kind == "source":
                if value == "RAD":
                    overrides.pop(tk, None)
                else:
                    overrides[tk] = value
                labels.append(f"{tk}:{BASE_OVERRIDES.get(tk,'RAD')}->{value}")
            elif kind == "exclude":
                excluded.add(tk)
                labels.append(f"drop {tk}")
        state = asset_state("Commodity", tuple(sorted(overrides.items())), tuple(sorted(excluded)), "contract_equal_path")
        obj = commodity_objective(state)
        rows.append(
            {
                "label": ", ".join(labels) if labels else "base",
                "overrides": overrides,
                "excluded": excluded,
                "state": state,
                "objective": obj,
            }
        )
    rows.sort(key=lambda r: r["objective"])
    return rows, rows[0]


def global_numerator_reaudit(clean_overrides: dict[str, str], clean_excluded: set[str]):
    rows = []
    default_row = None
    winner = None
    for candidate in NUMERATOR_CANDIDATES:
        per_asset = []
        ann_errs = []
        cal_errs = []
        for asset in ASSETS:
            state = asset_state(asset, tuple(sorted(clean_overrides.items())), tuple(sorted(clean_excluded)), "contract_equal_path")
            cand = state["candidates"][candidate]
            per_asset.append((asset, cand["annual_return_err"], cand["calmar_err"]))
            ann_errs.append(cand["annual_return_err"])
            cal_errs.append(cand["calmar_err"])
        row = {
            "candidate": candidate,
            "mean_ann_err": float(np.mean(ann_errs)),
            "mean_cal_err": float(np.mean(cal_errs)),
            "worst_ann_err": float(np.max(ann_errs)),
            "worst_cal_err": float(np.max(cal_errs)),
            "per_asset": per_asset,
        }
        rows.append(row)
        if candidate == "wealth_cagr":
            default_row = row
    rows.sort(key=lambda r: (r["mean_ann_err"], r["mean_cal_err"], r["worst_ann_err"], r["worst_cal_err"]))
    default_mean_ann = default_row["mean_ann_err"]
    for row in rows:
        if row["candidate"] == "wealth_cagr":
            continue
        if row["mean_ann_err"] >= default_mean_ann:
            continue
        # strict per-asset no-worse rule against the frozen default
        default_map = {asset: (ann, cal) for asset, ann, cal in default_row["per_asset"]}
        if all(
            ann <= default_map[asset][0] + 5.0 and cal <= default_map[asset][1] + 10.0
            for asset, ann, cal in row["per_asset"]
        ):
            winner = row
            break
    return rows, winner


def extraction_audit(clean_overrides: dict[str, str], clean_excluded: set[str]):
    rows = []
    for path_mode, numerator_mode in EXTRACTION_CANDIDATES:
        ann_errs = []
        cal_errs = []
        for asset in ASSETS:
            state = asset_state(asset, tuple(sorted(clean_overrides.items())), tuple(sorted(clean_excluded)), path_mode)
            cand = state["candidates"][numerator_mode]
            ann_errs.append(cand["annual_return_err"])
            cal_errs.append(cand["calmar_err"])
        rows.append(
            {
                "path_mode": path_mode,
                "numerator_mode": numerator_mode,
                "mean_ann_err": float(np.mean(ann_errs)),
                "mean_cal_err": float(np.mean(cal_errs)),
                "worst_ann_err": float(np.max(ann_errs)),
                "worst_cal_err": float(np.max(cal_errs)),
            }
        )
    rows.sort(key=lambda r: (r["mean_ann_err"], r["mean_cal_err"], r["worst_ann_err"], r["worst_cal_err"]))
    return rows


def final_classification(commodity_best, numerator_winner, extraction_rows):
    base_obj = commodity_best["objective"]
    if commodity_best["label"] != "base" and base_obj[0] < 0.05:
        if numerator_winner:
            return "MDD aligned, numerator wrong", "在 cleaned baseline 上把 default numerator 改到全局 winner"
        best_extract = extraction_rows[0]
        if best_extract["mean_ann_err"] + best_extract["mean_cal_err"] < 80:
            return "both off", "先保留 current numerator；下一步优先审 same-path reporting extraction / aggregation"
        return "contract-driven distortion", "先固定 Commodity cleaned baseline，再重审 numerator"
    if numerator_winner:
        return "MDD aligned, numerator wrong", "先采用全局 numerator winner，再复核 Commodity"
    if extraction_rows and extraction_rows[0]["path_mode"] != "contract_equal_path":
        return "both off", "优先审 reporting extraction / aggregation，而不是继续做广义 source 搜索"
    return "contract-driven distortion", "先把 Commodity distortion 解释清楚，再谈全局 Calmar 对齐"


def main():
    report = [
        "# Calmar Alignment Iteration",
        "",
        "Frozen strategy:",
        "",
        "- Table 3 Long only",
        "- ignore `All` until the 4 asset rows stabilize",
        "- fixed trade world",
        "- frozen reporting bridge starts from `RISK_PRICE_SIGMA0`",
        "- frozen rebuilt historical strong baseline:",
        "  - `EN -> REV`",
        "  - `DT -> REV`",
        "  - `CC -> RAD_REGEN`",
        "  - `LB -> RAD`",
        "  - `JO -> RAD_REGEN`",
        "  - `ZH -> RAD_REGEN`",
        "  - exclusions: `FB, ZA, ZO`",
        "",
    ]

    # Iteration 0
    report.extend(
        [
            "## Iteration 0 — Frozen Baseline",
            "",
            md_table(
                ["Asset", "Trade E(R)", "Reporting Annual Return", "Paper-Implied Annual Return", "MDD", "Paper MDD", "Calmar", "Paper Calmar"],
                [asset_summary_line(asset, BASE_OVERRIDES, BASE_EXCLUDED) for asset in ASSETS],
            ),
            "",
        ]
    )

    # Iteration 1
    commodity_base, suspect_rows, candidate_set = commodity_suspect_audit()
    report.extend(
        [
            "## Iteration 1 — Commodity Suspect Contract Audit",
            "",
            "Suspects: `SB, KC, ZL, NR, ZC`",
            "",
        ]
    )
    for row in suspect_rows:
        source_table = [
            [
                source,
                rows_count,
                f"{trial_obj[0]:.4f}",
                f"{trial_obj[1]:.4f}",
                f"{trial_obj[2]:.4f}",
                f"{trial_obj[3]:.1f}%",
                f"{trial_obj[4]:.1f}%",
                f"{trial_obj[5]:.1f}%",
                f"{score:+.4f}",
            ]
            for source, rows_count, trial_obj, score in row["source_rows"]
        ]
        report.extend(
            [
                f"### {row['ticker']}",
                "",
                f"- current source: `{row['current']}`",
                f"- best source candidate: `{row['best_source']}`",
                f"- best source score: `{row['best_source_score']:+.4f}`",
                f"- drop score: `{row['drop_score']:+.4f}`",
                f"- classification: `{row['classification']}`",
                f"- recommendation: `{row['recommendation']}`",
                "",
                md_table(
                    ["Source", "Rows", "Ann Gap", "MDD Gap", "Calmar Gap", "E(R) Err", "% +ve Err", "Ave P/L Err", "Score"],
                    source_table,
                ),
                "",
                md_table(
                    ["Drop Effect", "Ann Gap", "MDD Gap", "Calmar Gap", "E(R) Err", "% +ve Err", "Ave P/L Err", "Score"],
                    [[
                        "drop",
                        f"{row['drop_obj'][0]:.4f}",
                        f"{row['drop_obj'][1]:.4f}",
                        f"{row['drop_obj'][2]:.4f}",
                        f"{row['drop_obj'][3]:.1f}%",
                        f"{row['drop_obj'][4]:.1f}%",
                        f"{row['drop_obj'][5]:.1f}%",
                        f"{row['drop_score']:+.4f}",
                    ]],
                ),
                "",
            ]
        )

    report.extend(
        [
            "Commodity candidate set carried into Iteration 2:",
            "",
            md_table(
                ["Ticker", "Current", "Best Source", "Classification", "Recommendation"],
                [[x["ticker"], x["current"], x["best_source"], x["classification"], x["recommendation"]] for x in candidate_set]
                or [["none", "-", "-", "-", "-"]],
            ),
            "",
        ]
    )

    # Iteration 2
    combo_rows, commodity_best = commodity_local_search(candidate_set)
    report.extend(
        [
            "## Iteration 2 — Commodity Local Combination Search",
            "",
            md_table(
                ["Candidate", "Ann Gap", "MDD Gap", "Calmar Gap", "E(R) Err", "% +ve Err", "Ave P/L Err"],
                [
                    [
                        row["label"],
                        f"{row['objective'][0]:.4f}",
                        f"{row['objective'][1]:.4f}",
                        f"{row['objective'][2]:.4f}",
                        f"{row['objective'][3]:.1f}%",
                        f"{row['objective'][4]:.1f}%",
                        f"{row['objective'][5]:.1f}%",
                    ]
                    for row in combo_rows[:12]
                ],
            ),
            "",
            f"Commodity-cleaned baseline: `{commodity_best['label']}`",
            "",
        ]
    )

    clean_overrides = commodity_best["overrides"]
    clean_excluded = commodity_best["excluded"]

    # Iteration 3
    numerator_rows, numerator_winner = global_numerator_reaudit(clean_overrides, clean_excluded)
    report.extend(
        [
            "## Iteration 3 — Global Numerator Re-Audit",
            "",
            md_table(
                ["Candidate", "Mean Annual Return Gap", "Mean Calmar Gap", "Worst Annual Return Gap", "Worst Calmar Gap"],
                [
                    [
                        row["candidate"],
                        f"{row['mean_ann_err']:.1f}%",
                        f"{row['mean_cal_err']:.1f}%",
                        f"{row['worst_ann_err']:.1f}%",
                        f"{row['worst_cal_err']:.1f}%",
                    ]
                    for row in numerator_rows
                ],
            ),
            "",
            f"Global numerator winner: `{numerator_winner['candidate'] if numerator_winner else 'none'}`",
            "",
        ]
    )

    # Iteration 4
    extraction_rows = extraction_audit(clean_overrides, clean_excluded)
    report.extend(
        [
            "## Iteration 4 — Reporting Extraction / Aggregation Audit",
            "",
            "Notes:",
            "",
            "- On the 4 asset rows, `asset-equal` / `asset-count-weighted` aggregation is not yet meaningful; those are deferred to the future `All` audit.",
            "- This stage therefore audits only same-path extraction variants inside the frozen per-asset reporting path family.",
            "",
            md_table(
                ["Path Mode", "Return Extraction", "Mean Annual Return Gap", "Mean Calmar Gap", "Worst Annual Return Gap", "Worst Calmar Gap"],
                [
                    [
                        row["path_mode"],
                        row["numerator_mode"],
                        f"{row['mean_ann_err']:.1f}%",
                        f"{row['mean_cal_err']:.1f}%",
                        f"{row['worst_ann_err']:.1f}%",
                        f"{row['worst_cal_err']:.1f}%",
                    ]
                    for row in extraction_rows
                ],
            ),
            "",
        ]
    )

    # Iteration 5 (deferred unless stable)
    stable = bool(numerator_winner) or commodity_best["label"] != "base"
    report.extend(
        [
            "## Iteration 5 — `All` Consistency Audit",
            "",
            f"- status: `{'deferred' if not stable else 'ready to open next'}`",
            "",
        ]
    )
    if not stable:
        report.extend(
            [
                "Deferred reason:",
                "",
                "- the 4 asset rows do not yet have a single stable explanation, so reopening `All` now would mix near-zero artifact analysis with unresolved asset-level distortion",
                "",
            ]
        )

    # Final diagnosis
    final_cls, next_action = final_classification(commodity_best, numerator_winner, extraction_rows)
    report.extend(
        [
            "## Final Diagnosis",
            "",
            f"- final classification: `{final_cls}`",
            f"- next single recommended action: `{next_action}`",
            "",
            "Current cleaned baseline after Commodity iteration:",
            "",
            md_table(
                ["Asset", "Trade E(R)", "Reporting Annual Return", "Paper-Implied Annual Return", "MDD", "Paper MDD", "Calmar", "Paper Calmar"],
                [asset_summary_line(asset, clean_overrides, clean_excluded) for asset in ASSETS],
            ),
            "",
            "Assumptions used in this iteration:",
            "",
            "- candidate source changes in Iteration 1 are only promoted when the alternative has the same test-window row count as the current source",
            "- commodity local search only touches `SB, KC, ZL, NR, ZC`",
            "- `REV` remains reference-only for the known negative-price-sensitive problem contracts",
            "- `All` remains deferred until the 4 asset rows stabilize",
            "",
        ]
    )

    DOC_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {DOC_PATH}")
    print(f"Commodity cleaned baseline: {commodity_best['label']}")
    print(f"Global numerator winner: {numerator_winner['candidate'] if numerator_winner else 'none'}")
    print(f"Final classification: {final_cls}")
    print(f"Next action: {next_action}")


if __name__ == "__main__":
    main()
