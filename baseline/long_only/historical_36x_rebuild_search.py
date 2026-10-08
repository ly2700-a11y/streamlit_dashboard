#!/usr/bin/env python3
"""
Rebuild the historical 36/40 Table 3 frontier under the newer source-policy
understanding, then run a constrained source-only and exclusion-only search.

Doctrine:
  - hard score = 40 cells (Calmar excluded)
  - trade world drives 7 trade metrics
  - reporting world only contributes MDD/Calmar/helper annual return
  - All is not a primary driver; the 4 asset rows drive constraints/ranking
  - % +ve and Ave P/L must both be <= 2% error on each of the 4 asset rows
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
from metrics import compute_metrics  # noqa: E402


SIGMA = 0.058
ASSETS4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
CORE40_METRICS = [m for m in METRIC_NAMES if m != "Calmar"]
DOC_PATH = ROOT / "docs" / "historical_36x_rebuild_search.md"

HISTORICAL_EXCLUDED = ["FB", "ZA", "ZO"]
EXCLUSION_POOL = ["FB", "ZA", "ZO", "ZH", "CC", "LB", "JO"]

# Historical strong 36/40 skeleton, reconstructed from project memory:
#   EN: RAD_REGEN -> REV
#   DT: RAD -> REV
#   drop FB, ZA, ZO
# and before the later hard global REGEN_ONLY tightening:
#   CC -> RAD_REGEN
#   LB -> REV
#   JO -> REV
#   ZH -> REV
#   ZO -> RAD (default; excluded in the skeleton)
HISTORICAL_STRONG_OVERRIDES = dict(SOURCE_OVERRIDES)
HISTORICAL_STRONG_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "REV",
        "JO": "REV",
        "ZH": "REV",
    }
)
HISTORICAL_STRONG_OVERRIDES.pop("ZO", None)

# New-understanding rebuilt baseline:
#   keep the historical structure
#   but move problem contracts into allowed-source sets
REBUILT_BASE_OVERRIDES = dict(SOURCE_OVERRIDES)
REBUILT_BASE_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
    }
)
REBUILT_BASE_OVERRIDES.pop("ZO", None)

PROBLEM_AUDIT_SOURCES = {
    "CC": ["RAD", "REV", "RAD_REGEN", "NON_FWD_ANCHORED"],
    "LB": ["RAD", "REV", "RAD_REGEN", "NON_FWD_ANCHORED"],
    "JO": ["RAD", "REV", "RAD_REGEN", "NON_FWD_ANCHORED"],
    "ZH": ["RAD", "REV", "RAD_REGEN", "NON_FWD_ANCHORED"],
    "ZO": ["RAD", "REV", "RAD_REGEN", "NON_FWD_ANCHORED"],
}

ALLOWED_SOURCES = {
    "CC": ["RAD_REGEN", "NON_FWD_ANCHORED"],
    "LB": ["RAD", "NON_FWD_ANCHORED", "RAD_REGEN"],
    "JO": ["RAD_REGEN"],
    "ZH": ["RAD_REGEN"],
    "ZO": ["RAD", "RAD_REGEN", "NON_FWD_ANCHORED"],
}

SOURCE_SEARCH_ORDER = ["EN", "DT", "CC", "LB", "JO", "ZH", "ZO"]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def pct_err(value: float, paper: float) -> float:
    if paper == 0:
        return abs(value - paper)
    return abs((value - paper) / abs(paper)) * 100.0


def paper_helper(asset: str) -> float:
    paper = PAPER_TABLE3[asset]["Long"]
    return paper["Calmar"] * paper["MDD"]


def allowed_for_ticker(ticker: str):
    if ticker in ALLOWED_SOURCES:
        return list(ALLOWED_SOURCES[ticker])
    out = []
    for source in ["RAD", "REV", "RAD_REGEN"]:
        if load_clc_full(ticker, source=source) is not None:
            out.append(source)
    return out


def test_rows(ticker: str, source: str) -> int:
    df = load_clc_full(ticker, source=source, anchor_date="2011-01-01")
    if df is None:
        return 0
    mask = (df["Date"] >= "2011-01-01") & (df["Date"] <= "2019-12-31")
    return int(mask.sum())


def load_raw(asset: str, overrides: dict[str, str], excluded: list[str]):
    if asset == "All":
        raw = []
        for asset_name in ASSETS4:
            raw.extend(load_contracts(asset_name, excluded_contracts=excluded, source_overrides=overrides))
        return raw
    return load_contracts(asset, excluded_contracts=excluded, source_overrides=overrides)


def reporting_stats(raw):
    reporting = build_reporting_portfolio_risk_price_sigma0(raw, SIGMA, strat="Long")
    if reporting is None:
        return None
    port = reporting["portfolio_path"]
    ann_return = float((port[-1] / port[0]) ** (252.0 / len(port)) - 1.0)
    peak = np.maximum.accumulate(port)
    mdd = float(np.max((peak - port) / peak))
    calmar = ann_return / mdd if mdd > 0 else 0.0
    return ann_return, mdd, calmar


@lru_cache(maxsize=None)
def scenario_summary(overrides_key: tuple[tuple[str, str], ...], excluded_key: tuple[str, ...]):
    overrides = dict(overrides_key)
    excluded = list(excluded_key)
    assets = {}
    score10_40 = 0
    score15_40 = 0
    er_errs = []
    helper_gaps = []
    mdd_errs = []

    for asset in ASSETS4 + ["All"]:
        raw = load_raw(asset, overrides, excluded)
        returns = compute_portfolio_returns(raw, "Long", SIGMA, aggregation_mode="variable_n")
        metrics = dict(zip(METRIC_NAMES, compute_metrics(returns, n_contracts=len(raw))))
        helper_ann, mdd, calmar = reporting_stats(raw)
        metrics["MDD"] = round(mdd, 3)
        metrics["Calmar"] = round(calmar, 3)
        paper = PAPER_TABLE3[asset]["Long"]
        errors = {metric: pct_err(metrics[metric], paper[metric]) for metric in METRIC_NAMES}
        helper_target = paper_helper(asset)
        helper_gap = abs(helper_ann - helper_target)

        asset_score10 = sum(errors[m] < 10 for m in CORE40_METRICS)
        asset_score15 = sum(errors[m] < 15 for m in CORE40_METRICS)
        score10_40 += asset_score10
        score15_40 += asset_score15

        if asset != "All":
            er_errs.append(errors["E(R)"])
            helper_gaps.append(helper_gap)
            mdd_errs.append(errors["MDD"])

        assets[asset] = {
            "metrics": metrics,
            "errors": errors,
            "helper_ann": helper_ann,
            "helper_target": helper_target,
            "helper_gap": helper_gap,
            "asset_score10": asset_score10,
            "asset_score15": asset_score15,
            "pve_ok_2": errors["% +ve"] <= 2.0,
            "pl_ok_2": errors["Ave P/L"] <= 2.0,
        }

    hard_ok = all(assets[a]["pve_ok_2"] and assets[a]["pl_ok_2"] for a in ASSETS4)
    rank = (
        0 if hard_ok else 1,
        -score15_40,
        -score10_40,
        float(np.mean(er_errs)),
        float(np.mean(helper_gaps)),
        float(np.mean(mdd_errs)),
        assets["All"]["helper_gap"],
        assets["All"]["errors"]["E(R)"],
    )
    return {
        "assets": assets,
        "hard_ok": hard_ok,
        "score10_40": score10_40,
        "score15_40": score15_40,
        "avg_er_err": float(np.mean(er_errs)),
        "avg_helper_gap": float(np.mean(helper_gaps)),
        "avg_mdd_err": float(np.mean(mdd_errs)),
        "rank": rank,
    }


def summarize(overrides: dict[str, str], excluded: list[str]):
    return scenario_summary(tuple(sorted(overrides.items())), tuple(sorted(excluded)))


def problem_contract_tables(base_overrides: dict[str, str], excluded: list[str]):
    sections = []
    for ticker in ["CC", "LB", "JO", "ZH", "ZO"]:
        asset = next(name for name, tickers in ASSET_CLASSES.items() if ticker in tickers)
        standalone_rows = []
        for source in PROBLEM_AUDIT_SOURCES[ticker]:
            rows = test_rows(ticker, source)
            allowed = "yes" if source in ALLOWED_SOURCES[ticker] else "reference"
            if rows == 0:
                standalone_rows.append([source, allowed, "0", "N/A", "N/A", "N/A", "N/A"])
                continue
            raw = load_contracts(
                asset,
                excluded_contracts=[tk for tk in ASSET_CLASSES[asset] if tk != ticker],
                source_overrides={ticker: source},
            )
            if not raw:
                standalone_rows.append([source, allowed, "0", "N/A", "N/A", "N/A", "N/A"])
                continue
            returns = compute_portfolio_returns(raw, "Long", SIGMA, aggregation_mode="variable_n")
            metrics = dict(zip(METRIC_NAMES, compute_metrics(returns, n_contracts=1)))
            helper = reporting_stats(raw)
            if helper is None:
                standalone_rows.append([source, allowed, str(rows), f"{metrics['E(R)']:+.3f}", "N/A", "N/A", "N/A"])
            else:
                helper_ann, mdd, calmar = helper
                standalone_rows.append(
                    [source, allowed, str(rows), f"{metrics['E(R)']:+.3f}", f"{helper_ann:+.3f}", f"{mdd:.3f}", f"{calmar:+.3f}"]
                )

        asset_rows = []
        base_summary = summarize(base_overrides, excluded)["assets"][asset]
        for source in PROBLEM_AUDIT_SOURCES[ticker]:
            allowed = "yes" if source in ALLOWED_SOURCES[ticker] else "reference"
            trial_overrides = dict(base_overrides)
            trial_overrides[ticker] = source
            summary = summarize(trial_overrides, excluded)["assets"][asset]
            asset_rows.append(
                [
                    source,
                    allowed,
                    f"{summary['metrics']['E(R)']:+.3f}",
                    f"{summary['helper_ann']:+.3f}",
                    f"{summary['helper_gap']:.4f}",
                    f"{summary['metrics']['MDD']:.3f}",
                    f"{summary['metrics']['Calmar']:+.3f}",
                    summary["asset_score15"],
                ]
            )

        sections.extend(
            [
                f"### {ticker} in {asset}",
                "",
                "Single-contract audit:",
                "",
                md_table(
                    ["Source", "Allowed?", "Rows", "Trade E(R)", "Helper", "MDD", "Calmar"],
                    standalone_rows,
                ),
                "",
                "Asset-row impact:",
                "",
                md_table(
                    ["Source", "Allowed?", "Asset E(R)", "Asset Helper", "|Helper Gap|", "Asset MDD", "Asset Calmar", "Score <=15 /9"],
                    asset_rows,
                ),
                "",
            ]
        )
    return sections


def source_search(base_overrides: dict[str, str], excluded: list[str]):
    choices = {}
    for ticker in SOURCE_SEARCH_ORDER:
        if ticker in excluded:
            continue
        if ticker in ALLOWED_SOURCES:
            choices[ticker] = ALLOWED_SOURCES[ticker]
        else:
            choices[ticker] = allowed_for_ticker(ticker)

    tickers = [tk for tk, opts in choices.items() if len(opts) > 1]
    option_lists = [choices[tk] for tk in tickers]

    rows = []
    for combo in itertools.product(*option_lists):
        trial = dict(base_overrides)
        deltas = []
        for ticker, source in zip(tickers, combo):
            current = trial.get(ticker, "RAD")
            if source == "RAD":
                trial.pop(ticker, None)
            else:
                trial[ticker] = source
            if source != current:
                deltas.append(f"{ticker}:{current}->{source}")
        summary = summarize(trial, excluded)
        rows.append(
            {
                "label": ", ".join(deltas) if deltas else "base",
                "overrides": trial,
                "deltas": deltas,
                "summary": summary,
            }
        )

    rows.sort(key=lambda row: row["summary"]["rank"])
    accepted = [row for row in rows if row["summary"]["hard_ok"]]
    rejected = [row for row in rows if not row["summary"]["hard_ok"]]
    return rows, accepted, rejected


def exclusion_search(base_overrides: dict[str, str]):
    rows = []
    for r in range(len(EXCLUSION_POOL) + 1):
        for combo in itertools.combinations(EXCLUSION_POOL, r):
            excluded = sorted(set(combo))
            summary = summarize(base_overrides, excluded)
            rows.append(
                {
                    "excluded": excluded,
                    "summary": summary,
                }
            )
    rows.sort(key=lambda row: row["summary"]["rank"])
    accepted = [row for row in rows if row["summary"]["hard_ok"]]
    rejected = [row for row in rows if not row["summary"]["hard_ok"]]
    return rows, accepted, rejected


def candidate_row(label: str, summary: dict):
    return [
        label,
        "yes" if summary["hard_ok"] else "no",
        f"{summary['score15_40']}/40",
        f"{summary['score10_40']}/40",
        f"{summary['avg_er_err']:.2f}",
        f"{summary['avg_helper_gap']:.4f}",
        f"{summary['avg_mdd_err']:.2f}",
    ]


def violation_text(summary: dict) -> str:
    bad = []
    for asset in ASSETS4:
        errs = summary["assets"][asset]["errors"]
        if errs["% +ve"] > 2.0:
            bad.append(f"{asset} % +ve {errs['% +ve']:.2f}%")
        if errs["Ave P/L"] > 2.0:
            bad.append(f"{asset} Ave P/L {errs['Ave P/L']:.2f}%")
    return "; ".join(bad) if bad else "none"


def main():
    historical = summarize(HISTORICAL_STRONG_OVERRIDES, HISTORICAL_EXCLUDED)
    rebuilt = summarize(REBUILT_BASE_OVERRIDES, HISTORICAL_EXCLUDED)

    source_rows, source_ok, source_bad = source_search(REBUILT_BASE_OVERRIDES, HISTORICAL_EXCLUDED)
    best_source = source_ok[0] if source_ok else source_rows[0]

    excl_rows, excl_ok, excl_bad = exclusion_search(best_source["overrides"])
    best_excl = excl_ok[0] if excl_ok else excl_rows[0]

    calmar_trigger = (
        (not source_ok)
        or (not excl_ok)
        or best_excl["summary"]["avg_er_err"] > rebuilt["avg_er_err"]
    )

    sections = [
        "# Historical 36/40 Rebuild Search",
        "",
        "Goal:",
        "",
        "- start from the historical `36/40` strong baseline",
        "- rebuild it under the newer source understanding",
        "- enforce `% +ve` and `Ave P/L` <= 2% on each of the 4 asset rows",
        "- only if that fails, trigger a separate `Calmar` / reporting-return formula line",
        "",
        "## Step A: Historical Strong Baseline vs Rebuilt Baseline",
        "",
        md_table(
            ["Scenario", "Hard 2% OK?", "<=15 /40", "<=10 /40", "Avg E(R) Err", "Avg Helper Gap", "Avg MDD Err"],
            [
                candidate_row("Historical 36/40 skeleton", historical),
                candidate_row("Rebuilt baseline", rebuilt),
            ],
        ),
        "",
        "Historical 36/40 skeleton detail:",
        "",
        "- overrides: `EN:REV`, `DT:REV`, `CC:RAD_REGEN`, `LB:REV`, `JO:REV`, `ZH:REV`",
        "- excluded: `FB, ZA, ZO`",
        "",
        "Rebuilt baseline detail:",
        "",
        "- overrides: `EN:REV`, `DT:REV`, `CC:RAD_REGEN`, `LB:RAD`, `JO:RAD_REGEN`, `ZH:RAD_REGEN`",
        "- excluded: `FB, ZA, ZO`",
        "",
        "Rebuilt baseline 2% violations:",
        "",
        f"- {violation_text(rebuilt)}",
        "",
        "## Step A.1: Problem-Contract Four-Table Audit",
        "",
        "Allowed sources locked for the search:",
        "",
        "- `CC`: `RAD_REGEN`, `NON_FWD_ANCHORED`",
        "- `LB`: `RAD`, `NON_FWD_ANCHORED`, `RAD_REGEN`",
        "- `JO`: `RAD_REGEN`",
        "- `ZH`: `RAD_REGEN`",
        "- `ZO`: `RAD`, `RAD_REGEN`, `NON_FWD_ANCHORED`",
        "",
    ]
    sections.extend(problem_contract_tables(REBUILT_BASE_OVERRIDES, HISTORICAL_EXCLUDED))

    top_source_ok = source_ok[:10]
    top_source_bad = source_bad[:10]
    sections.extend(
        [
            "## Step B: Source-Only Frontier",
            "",
            "Accepted candidates (all 4 asset rows satisfy `% +ve` and `Ave P/L` <= 2%):",
            "",
            md_table(
                ["Candidate", "Hard 2% OK?", "<=15 /40", "<=10 /40", "Avg E(R) Err", "Avg Helper Gap", "Avg MDD Err"],
                [candidate_row(row["label"], row["summary"]) for row in top_source_ok] or [["none", "-", "-", "-", "-", "-", "-"]],
            ),
            "",
            "Rejected top candidates (failed the 2% rule):",
            "",
            md_table(
                ["Candidate", "Hard 2% OK?", "<=15 /40", "<=10 /40", "Avg E(R) Err", "Avg Helper Gap", "Avg MDD Err", "Violations"],
                [
                    candidate_row(row["label"], row["summary"]) + [violation_text(row["summary"])]
                    for row in top_source_bad
                ]
                or [["none", "-", "-", "-", "-", "-", "-", "-"]],
            ),
            "",
            "Best source-only base used for the next step:",
            "",
            f"- `{best_source['label']}`",
            f"- hard 2% ok: `{'yes' if best_source['summary']['hard_ok'] else 'no'}`",
            "",
        ]
    )

    top_excl_ok = excl_ok[:10]
    top_excl_bad = excl_bad[:10]
    sections.extend(
        [
            "## Step C: Source + Exclusion Frontier",
            "",
            "Accepted candidates (all 4 asset rows satisfy `% +ve` and `Ave P/L` <= 2%):",
            "",
            md_table(
                ["Excluded", "Hard 2% OK?", "<=15 /40", "<=10 /40", "Avg E(R) Err", "Avg Helper Gap", "Avg MDD Err"],
                [candidate_row(",".join(row["excluded"]) or "none", row["summary"]) for row in top_excl_ok]
                or [["none", "-", "-", "-", "-", "-", "-"]],
            ),
            "",
            "Rejected top candidates (failed the 2% rule):",
            "",
            md_table(
                ["Excluded", "Hard 2% OK?", "<=15 /40", "<=10 /40", "Avg E(R) Err", "Avg Helper Gap", "Avg MDD Err", "Violations"],
                [
                    candidate_row(",".join(row["excluded"]) or "none", row["summary"]) + [violation_text(row["summary"])]
                    for row in top_excl_bad[:15]
                ]
                or [["none", "-", "-", "-", "-", "-", "-", "-"]],
            ),
            "",
            "Best source+exclusion candidate:",
            "",
            f"- source base: `{best_source['label']}`",
            f"- excluded: `{','.join(best_excl['excluded']) or 'none'}`",
            f"- hard 2% ok: `{'yes' if best_excl['summary']['hard_ok'] else 'no'}`",
            f"- score: `<=15 {best_excl['summary']['score15_40']}/40`, `<=10 {best_excl['summary']['score10_40']}/40`",
            "",
            "Asset-row payload of the best source+exclusion candidate:",
            "",
            md_table(
                ["Asset", "<=15 /8", "<=10 /8", "E(R)", "MDD", "Helper", "Paper Helper", "% +ve", "Ave P/L", "% +ve Err", "Ave P/L Err"],
                [
                    [
                        asset,
                        best_excl["summary"]["assets"][asset]["asset_score15"],
                        best_excl["summary"]["assets"][asset]["asset_score10"],
                        f"{best_excl['summary']['assets'][asset]['metrics']['E(R)']:+.3f}",
                        f"{best_excl['summary']['assets'][asset]['metrics']['MDD']:.3f}",
                        f"{best_excl['summary']['assets'][asset]['helper_ann']:+.3f}",
                        f"{best_excl['summary']['assets'][asset]['helper_target']:+.3f}",
                        f"{best_excl['summary']['assets'][asset]['metrics']['% +ve']:.3f}",
                        f"{best_excl['summary']['assets'][asset]['metrics']['Ave P/L']:.3f}",
                        f"{best_excl['summary']['assets'][asset]['errors']['% +ve']:.2f}%",
                        f"{best_excl['summary']['assets'][asset]['errors']['Ave P/L']:.2f}%",
                    ]
                    for asset in ASSETS4
                ],
            ),
            "",
            "## Step D: Calmar Trigger Status",
            "",
            f"- trigger Calmar / reporting-return formula line: `{'yes' if calmar_trigger else 'no'}`",
            "",
        ]
    )

    if calmar_trigger:
        sections.extend(
            [
                "Reason:",
                "",
                "- under the rebuilt, source-clean doctrine, no candidate satisfied the per-asset `% +ve` and `Ave P/L` <= 2% rule, or the frontier remained clearly capped after satisfying the trade-side source constraints",
                "- this means the bottleneck is no longer just trade-side data cleaning; the reporting-return / Calmar line needs a separate audit",
                "",
            ]
        )

    DOC_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {DOC_PATH}")
    print(f"Historical skeleton: <=15 {historical['score15_40']}/40 <=10 {historical['score10_40']}/40 hard_ok={historical['hard_ok']}")
    print(f"Rebuilt baseline: <=15 {rebuilt['score15_40']}/40 <=10 {rebuilt['score10_40']}/40 hard_ok={rebuilt['hard_ok']}")
    print(f"Best source-only: {best_source['label']} <=15 {best_source['summary']['score15_40']}/40 hard_ok={best_source['summary']['hard_ok']}")
    print(f"Best source+excl: excluded={','.join(best_excl['excluded']) or 'none'} <=15 {best_excl['summary']['score15_40']}/40 hard_ok={best_excl['summary']['hard_ok']}")


if __name__ == "__main__":
    main()
