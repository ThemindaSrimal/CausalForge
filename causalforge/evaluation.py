from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

from .budget import SpendLedger
from .config import load_dotenv
from .meta_api import MuseSparkClient
from .orchestrator import DiscoveryRun
from .world import FACTOR_NAMES, Intervention, LEVELS


LOCKED_SEEDS = (104729, 130363, 155921, 196613, 225307, 262147, 299993, 350377, 410117, 499979)
SUCCESS_RATE_TARGET = 0.8
COEFFICIENT_RMSE_LIMIT = 0.25


def score_run(run: DiscoveryRun) -> dict:
    truth = run.simulator.reveal()
    actual = truth["coefficients"]
    estimate = run.learner.summary()["estimated_coefficients"]
    actual_active = {name for name, value in actual.items() if value != 0}
    estimated_active = set(sorted(FACTOR_NAMES, key=lambda name: abs(estimate[name]), reverse=True)[:2])
    coefficient_rmse = math.sqrt(sum((estimate[name] - actual[name]) ** 2 for name in FACTOR_NAMES) / len(FACTOR_NAMES))
    errors = []
    for temperature in LEVELS:
        for pressure in LEVELS:
            for catalyst in LEVELS:
                for field in LEVELS:
                    intervention = Intervention(temperature, pressure, catalyst, field)
                    values = intervention.vector()[1:]
                    actual_mean = truth["intercept"] + sum(actual[name] * value for name, value in zip(FACTOR_NAMES, values))
                    predicted = run.learner.predict(intervention).mean
                    errors.append((predicted - actual_mean) ** 2)
    prediction_rmse = math.sqrt(sum(errors) / len(errors))
    exact_support = estimated_active == actual_active
    success = exact_support and coefficient_rmse <= COEFFICIENT_RMSE_LIMIT
    return {
        "seed": run.simulator.public.seed,
        "experiments": len(run.simulator.observations),
        "exact_active_factors": exact_support,
        "actual_active_factors": sorted(actual_active),
        "estimated_active_factors": sorted(estimated_active),
        "coefficient_rmse": round(coefficient_rmse, 6),
        "prediction_rmse": round(prediction_rmse, 6),
        "success": success,
        "cost_usd": round(run.total_cost_usd, 8),
    }


def evaluate(seeds: tuple[int, ...], experiments: int, output: Path, runs_dir: Path) -> dict:
    ledger = SpendLedger(runs_dir / "meta-spend.json", 120.0)
    started_spend = ledger.spent_usd
    results = []
    failures = []
    for index, seed in enumerate(seeds, 1):
        print(f"[{index}/{len(seeds)}] seed {seed}: starting", file=sys.stderr, flush=True)
        run = DiscoveryRun(seed, MuseSparkClient(), experiments, 1.0, runs_dir / "evaluation" / f"seed-{seed}.jsonl", ledger)
        try:
            run.run(on_event=lambda event: print(
                f"  experiment {event['observation']['experiment_id']}/{experiments} · ${event['cumulative_cost_usd']:.4f}",
                file=sys.stderr, flush=True,
            ))
            result = score_run(run)
            results.append(result)
            print(f"  {'PASS' if result['success'] else 'FAIL'} · support={result['exact_active_factors']} · coefficient RMSE={result['coefficient_rmse']}", file=sys.stderr, flush=True)
        except Exception as error:
            failures.append({"seed": seed, "error": str(error), "experiments_completed": len(run.simulator.observations), "cost_usd": round(run.total_cost_usd, 8)})
            print(f"  ERROR · {error}", file=sys.stderr, flush=True)
    successes = sum(item["success"] for item in results)
    completed = len(results)
    success_rate = successes / len(seeds)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MuseSparkClient.MODEL,
        "locked_seeds": list(seeds),
        "experiments_per_seed": experiments,
        "success_definition": {"exact_active_factors": True, "maximum_coefficient_rmse": COEFFICIENT_RMSE_LIMIT},
        "target_success_rate": SUCCESS_RATE_TARGET,
        "completed_runs": completed,
        "failed_runs": len(failures),
        "successful_runs": successes,
        "success_rate": round(success_rate, 4),
        "target_met": success_rate >= SUCCESS_RATE_TARGET and not failures,
        "evaluation_cost_usd": round(ledger.spent_usd - started_spend, 8),
        "results": results,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate Muse-driven discovery on locked unseen worlds")
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation-report.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    report = evaluate(LOCKED_SEEDS, args.experiments, args.output, args.runs_dir)
    print(json.dumps({key: report[key] for key in ("completed_runs", "failed_runs", "successful_runs", "success_rate", "target_met", "evaluation_cost_usd")}, indent=2))
    raise SystemExit(0 if report["target_met"] else 1)


if __name__ == "__main__":
    main()
