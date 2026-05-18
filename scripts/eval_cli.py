#!/usr/bin/env python3
"""
LossCast-Bench evaluation CLI.

Usage:
    losscast-eval --split val -p preds.json
    losscast-eval -p preds.json -g gt.json [-c configs.json]
    losscast-eval --split val -p preds.json --format table --verbose
    losscast-eval --split val -p preds.json --format json -o results.json
"""

import argparse
import json
import sys

from losscast_bench.schema import (
    load_predictions, load_ground_truths, load_configs, validate_submission,
)
from losscast_bench.data import load_split
from losscast_bench.metrics import evaluate


# ── Formatting helpers ──────────────────────────────────────────────────────

def _print_table_header(title: str, width: int = 56):
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_validation_report(errors: list[str], warnings: list[str]):
    """Print a structured validation summary."""
    _print_table_header("Submission Validation Report")

    n_errors = len([e for e in errors if not e.startswith("WARNING")])
    n_warnings = len(warnings)

    if not errors and not warnings:
        print("  Status: PASSED")
        print(f"  No errors or warnings found.")
    else:
        if n_errors > 0:
            print(f"  Status: FAILED ({n_errors} error(s), {n_warnings} warning(s))")
        else:
            print(f"  Status: PASSED with warnings ({n_warnings} warning(s))")

        if errors:
            print()
            print("  Errors:")
            for e in errors:
                print(f"    - {e}")

        if warnings:
            print()
            print("  Warnings:")
            for w in warnings:
                print(f"    - {w}")

    print("=" * 56)
    print()
    return n_errors


def _validation_json(errors: list[str], warnings: list[str]) -> dict:
    """Format validation status for JSON output."""
    return {
        "status": "FAILED" if errors else "PASSED",
        "errors": errors,
        "warnings": warnings,
    }


def _format_table(result, delta: float, verbose: bool):
    """Format evaluation results as a human-readable table."""
    s = result.summary()
    _print_table_header("LossCast-Bench Evaluation Results")

    print(f"  Runs evaluated:    {s['n_runs']}")
    print(f"  Total data points: {s['n_points']}")
    print("-" * 56)
    print(f"  Huber Loss (δ={delta}):   {s['huber']:.6f}  (primary)")
    print(f"  R²:                    {s['r2']:.4f}")
    print(f"  Curve MAE / RMSE:      {s['curve_mae']:.6f} / {s['curve_rmse']:.6f}")
    print(f"  Curve MAPE:            {s['curve_mape']:.4f}")
    print("-" * 56)
    print(f"  Final Loss Huber:      {s['final_huber']:.6f}")
    print(f"  Final Loss R²:         {s['final_r2']:.4f}")
    print(f"  Final MAE / RMSE:      {s['final_mae']:.6f} / {s['final_rmse']:.6f}")
    print(f"  Final Spearman ρ:      {s['final_spearman']:.4f}")

    if "extrap_huber" in s:
        print("-" * 56)
        print(f"  Extrap Huber:          {s['extrap_huber']:.6f}")
        print(f"  Extrap R²:             {s['extrap_r2']:.4f}")
        print(f"  Extrap Runs:           {s['extrap_n_runs']}")
    print("=" * 56)

    if verbose and result.per_run:
        print()
        print("  Per-run breakdown (sorted by Huber, worst first):")
        print("-" * 56)
        print(f"  {'Run ID':<30s} {'Huber':>8s} {'R²':>7s} {'MAPE':>7s} {'Final':>8s}")
        print("-" * 56)
        for r in sorted(result.per_run, key=lambda x: x.huber, reverse=True):
            print(f"  {r.run_id:<30s} {r.huber:>8.6f} {r.r2:>7.4f} {r.mape:>7.4f} {r.final_huber:>8.6f}")
        print("-" * 56)


def _format_json(result, verbose: bool) -> dict:
    """Format evaluation results as a JSON-serializable dict."""
    output = result.summary()
    if verbose and result.per_run:
        output["per_run"] = [
            {
                "run_id": r.run_id,
                "huber": round(r.huber, 6),
                "final_huber": round(r.final_huber, 6),
                "r2": round(r.r2, 4),
                "mape": round(r.mape, 4),
                "n_steps": r.n_steps,
            }
            for r in sorted(result.per_run, key=lambda x: x.huber, reverse=True)
        ]
    return output


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LossCast-Bench predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  losscast-eval --split val -p preds.json
  losscast-eval -p preds.json -g gt.json -c configs.json --format table -v
  losscast-eval --split val -p preds.json --format json -o results.json
""",
    )
    parser.add_argument(
        "--predictions", "-p", required=True,
        help="Path to predictions JSON",
    )
    parser.add_argument(
        "--split", "-s", default=None,
        help="Data split to evaluate against (train/val/test). "
             "Loads configs and ground truth from per-run directories.",
    )
    parser.add_argument(
        "--ground-truth", "-g", default=None,
        help="Path to ground truth JSON (alternative to --split)",
    )
    parser.add_argument(
        "--configs", "-c", default=None,
        help="Path to run configs JSON (alternative to --split)",
    )
    parser.add_argument(
        "--extrap-threshold", type=float, default=None,
        help="Compute threshold for extrapolation split (in tokens)",
    )
    parser.add_argument(
        "--delta", type=float, default=0.01,
        help="Huber loss delta (default: 0.01)",
    )
    parser.add_argument(
        "--format", "-f", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Include per-run breakdown",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate the submission, do not evaluate",
    )
    parser.add_argument(
        "--allow-invalid", action="store_true",
        help="Debug mode: evaluate even if validation has fatal errors.",
    )
    args = parser.parse_args()

    # Load data — either from --split or from explicit file paths
    if args.split:
        split_configs, split_gts = load_split(args.split)
        configs = split_configs
        ground_truths = split_gts or []
    else:
        if not args.ground_truth:
            parser.error("Either --split or --ground-truth is required")
        ground_truths = load_ground_truths(args.ground_truth)
        configs = load_configs(args.configs) if args.configs else None

    predictions = load_predictions(args.predictions)

    # ── Validation ──────────────────────────────────────────────────────
    errors = []
    warnings = []
    if configs or ground_truths:
        raw_errors = validate_submission(
            predictions,
            configs=configs,
            ground_truths=ground_truths,
        )
        for e in raw_errors:
            if "suspiciously high" in e:
                warnings.append(e)
            else:
                errors.append(e)

        if args.format == "table":
            n_fatal = _print_validation_report(errors, warnings)
        else:
            # JSON: embed validation in output
            pass

        if args.validate_only:
            if args.format == "json":
                print(json.dumps(_validation_json(errors, warnings), indent=2))
            sys.exit(1 if errors else 0)

        if errors and not args.allow_invalid:
            if args.format == "json":
                print(json.dumps({"validation": _validation_json(errors, warnings)}, indent=2))
            else:
                print("Evaluation skipped because validation failed.")
                print("Use --allow-invalid only for local debugging.")
            sys.exit(1)

    elif args.validate_only:
        print(
            "Error: --validate-only requires --configs, --ground-truth, or --split",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── Evaluation ──────────────────────────────────────────────────────
    result = evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
        configs=configs,
        delta=args.delta,
        extrap_compute_threshold=args.extrap_threshold,
    )

    # ── Output ──────────────────────────────────────────────────────────
    if args.format == "json":
        output = _format_json(result, args.verbose)
        if configs or ground_truths:
            output["validation"] = _validation_json(errors, warnings)
        json_str = json.dumps(output, indent=2)
        print(json_str)
    else:
        _format_table(result, args.delta, args.verbose)

    # ── Save ────────────────────────────────────────────────────────────
    if args.output:
        output = _format_json(result, verbose=True)
        if configs or ground_truths:
            output["validation"] = _validation_json(errors, warnings)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        if args.format == "table":
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
