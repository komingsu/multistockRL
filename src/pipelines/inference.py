"""Inference pipeline enabling custom portfolio checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from src.pipelines.evaluation import EvaluationResult, run_evaluation
from src.utils.config import Config, load_config


def _load_json_blob(value: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a JSON payload from a file path or inline string."""

    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    candidate_path = Path(value)
    if candidate_path.exists():
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    return json.loads(value)


def run_inference(
    config: Config,
    checkpoint_path: Path,
    *,
    portfolio_state: Optional[Dict[str, Any]] = None,
    eval_split: str = "validation",
    episodes: int = 1,
    deterministic: bool = True,
    output_dir: Optional[Path] = None,
    stress_overrides: Optional[Dict[str, Any]] = None,
    render_valuation: bool = True,
    integerize: bool = False,
    lot_size: int = 1,
) -> EvaluationResult:
    """Execute deterministic inference runs with optional state overrides."""

    return run_evaluation(
        config,
        checkpoint_path,
        eval_split=eval_split,
        episodes=episodes,
        deterministic=deterministic,
        output_dir=output_dir,
        stress_overrides=stress_overrides,
        portfolio_state=portfolio_state,
        render_nav_curves=render_valuation,
        integerize=integerize,
        lot_size=lot_size,
    )


def _format_summary(result: EvaluationResult) -> str:
    lines = ["Inference summary"]
    for key in sorted(result.summary):
        value = result.summary[key]
        lines.append(f"  {key}: {value}")
    if result.trace_paths:
        lines.append("Trace files:")
        for path in result.trace_paths:
            lines.append(f"  - {path}")
    if result.position_paths:
        lines.append("Position files:")
        for path in result.position_paths:
            lines.append(f"  - {path}")
    if result.valuation_paths:
        lines.append("Valuation charts:")
        for path in result.valuation_paths:
            lines.append(f"  - {path}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run checkpointed inference with custom portfolio state."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment config YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved checkpoint to load.",
    )
    parser.add_argument(
        "--portfolio-state",
        dest="portfolio_state",
        default=None,
        help="JSON file path or inline JSON describing cash/holdings/allocations.",
    )
    parser.add_argument(
        "--stress",
        dest="stress_overrides",
        default=None,
        help="Optional JSON path or literal overrides for environment stress profile.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to roll out deterministically.",
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="Dataset split to evaluate against (e.g. validation or test).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store evaluation traces and plots.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Enable stochastic inference (defaults to deterministic).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip valuation chart rendering (enabled by default).",
    )
    parser.add_argument(
        "--integerize",
        action="store_true",
        help="Integerize allocation weights to integer share counts before applying overrides.",
    )
    parser.add_argument(
        "--lot-size",
        type=int,
        default=1,
        help="Lot size used when integerizing allocations (default: 1).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    portfolio_state = _load_json_blob(args.portfolio_state)
    stress_overrides = _load_json_blob(args.stress_overrides)

    result = run_inference(
        config,
        checkpoint_path,
        portfolio_state=portfolio_state,
        eval_split=args.eval_split,
        episodes=max(1, int(args.episodes)),
        deterministic=not args.stochastic,
        output_dir=output_dir,
        stress_overrides=stress_overrides,
        render_valuation=not args.no_plots,
        integerize=args.integerize,
        lot_size=max(1, int(args.lot_size)),
    )

    print(_format_summary(result))


if __name__ == "__main__":
    main()
