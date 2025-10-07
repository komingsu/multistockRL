#!/usr/bin/env python
"""Generate summary reports from a training run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise metrics and traces for a completed run.")
    parser.add_argument("--run-dir", required=True, help="Path to the run directory (contains metrics.csv).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write reports (defaults to <run-dir>/reports).",
    )
    return parser.parse_args()


def _load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found under {run_dir}")
    return pd.read_csv(metrics_path)


def _final_row(metrics: pd.DataFrame) -> pd.Series:
    if metrics.empty:
        return pd.Series(dtype=float)
    if "step" in metrics:
        metrics = metrics.sort_values("step")
    return metrics.iloc[-1]


def _best_valid(metrics: pd.DataFrame) -> float:
    if "valid_reward" not in metrics or metrics["valid_reward"].isna().all():
        return float("nan")
    return float(metrics["valid_reward"].max())


def _summarise_metrics(metrics: pd.DataFrame) -> Dict[str, float]:
    final = _final_row(metrics)
    summary: Dict[str, float] = {}
    if not final.empty:
        for key in [
            "project_epoch",
            "train_reward",
            "valid_reward",
            "nav",
            "turnover",
            "drawdown",
            "total_env_steps",
        ]:
            if key in final:
                summary[f"final_{key}"] = float(final[key])
    summary["best_valid_reward"] = _best_valid(metrics)
    if "nav" in metrics:
        summary["max_nav"] = float(metrics["nav"].max())
    if "drawdown" in metrics:
        summary["worst_drawdown"] = float(metrics["drawdown"].max())
    return summary


def _gather_traces(run_dir: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for folder in ["validation_traces", "final_eval_traces"]:
        trace_dir = run_dir / folder
        if not trace_dir.exists():
            continue
        for path in sorted(trace_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            frame["source"] = folder
            frame["episode"] = path.stem
            frames.append(frame)
    return frames


def _summarise_traces(traces: List[pd.DataFrame]) -> pd.DataFrame:
    records = []
    for frame in traces:
        if frame.empty:
            continue
        start_value = float(frame["portfolio_value"].iloc[0])
        end_value = float(frame["portfolio_value"].iloc[-1])
        total_return = (end_value - start_value) / start_value if start_value else 0.0
        max_drawdown = float(frame["drawdown"].max()) if "drawdown" in frame else 0.0
        avg_turnover = float(frame["turnover"].mean()) if "turnover" in frame else 0.0
        records.append(
            {
                "source": frame["source"].iloc[0],
                "episode": frame["episode"].iloc[0],
                "final_nav": end_value,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "avg_turnover": avg_turnover,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(run_dir)
    metrics_summary = _summarise_metrics(metrics)

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_summary, handle, indent=2, sort_keys=True)
    pd.DataFrame([metrics_summary]).to_csv(output_dir / "metrics_summary.csv", index=False)

    traces = _gather_traces(run_dir)
    if traces:
        trace_summary = _summarise_traces(traces)
        trace_summary.to_csv(output_dir / "trace_summary.csv", index=False)

        combined = pd.concat(traces, ignore_index=True)
        agg = (
            combined.groupby("date")["portfolio_value"].agg(["mean", "max", "min"]).reset_index()
        )
        agg.to_csv(output_dir / "daily_nav.csv", index=False)


if __name__ == "__main__":
    main()
