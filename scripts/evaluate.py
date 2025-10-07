#!/usr/bin/env python
"""Deterministic evaluation CLI for saved checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.pipelines.evaluation import EvaluationResult, run_evaluation
from src.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation for a saved checkpoint.")
    parser.add_argument("--config", required=True, help="Path to configuration file used for training.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file (model.pt).")
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="Dataset split to evaluate on (default: validation).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of full episodes to evaluate deterministically.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="If set, use stochastic actions instead of deterministic policy outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store evaluation artifacts (defaults to checkpoint parent / evaluation).",
    )
    parser.add_argument(
        "--stress-enabled",
        action="store_true",
        help="Enable stress overrides (commission/slippage multipliers).",
    )
    parser.add_argument(
        "--stress-commission-mult",
        type=float,
        default=1.0,
        help="Multiplier applied to commission rate under stress.",
    )
    parser.add_argument(
        "--stress-slippage-mult",
        type=float,
        default=1.0,
        help="Multiplier applied to slippage bps under stress.",
    )
    return parser.parse_args()


def _build_stress_overrides(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    stress: Dict[str, Any] = {}
    if args.stress_enabled or args.stress_commission_mult != 1.0 or args.stress_slippage_mult != 1.0:
        stress["enabled"] = True
        if args.stress_commission_mult is not None:
            stress["commission_multiplier"] = float(args.stress_commission_mult)
        if args.stress_slippage_mult is not None:
            stress["slippage_multiplier"] = float(args.stress_slippage_mult)
        return stress
    return None


def _write_outputs(result: EvaluationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    csv_summary_path = output_dir / "evaluation_summary.csv"
    episodes_path = output_dir / "episode_metrics.csv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2, sort_keys=True)

    pd.DataFrame([result.summary]).to_csv(csv_summary_path, index=False)
    result.episode_summaries.to_csv(episodes_path, index=False)

    if result.episode_frames:
        daily_nav = (
            pd.concat(result.episode_frames, keys=range(1, len(result.episode_frames) + 1), names=["episode", "row"])
            .reset_index(level="row", drop=True)
        )
        agg = (
            daily_nav.groupby("date")["portfolio_value"].agg(["mean", "max", "min"]).reset_index()
        )
        agg.to_csv(output_dir / "daily_nav_summary.csv", index=False)



def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation"
    trace_dir = output_dir / "traces"

    stress_overrides = _build_stress_overrides(args)

    result = run_evaluation(
        config,
        checkpoint_path,
        eval_split=args.eval_split,
        episodes=max(1, int(args.episodes)),
        deterministic=not args.stochastic,
        output_dir=trace_dir,
        stress_overrides=stress_overrides,
    )

    _write_outputs(result, output_dir)


if __name__ == "__main__":
    main()
