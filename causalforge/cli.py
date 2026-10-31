from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .budget import SpendLedger
from .config import load_dotenv
from .meta_api import MuseSparkClient
from .orchestrator import DiscoveryRun


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run a real Muse Spark causal discovery session")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--log", type=Path, default=Path("runs/latest.jsonl"))
    parser.add_argument("--ledger", type=Path, default=Path("runs/meta-spend.json"))
    parser.add_argument("--project-cost-limit", type=float, default=120.0)
    args = parser.parse_args()
    ledger = SpendLedger(args.ledger, args.project_cost_limit)
    run = DiscoveryRun(args.seed, MuseSparkClient(), args.experiments, args.max_cost, args.log, ledger)
    def report(event: dict) -> None:
        completed = event["observation"]["experiment_id"]
        calls = event["api"]["calls"]
        cost = event["cumulative_cost_usd"]
        print(f"experiment {completed}/{args.experiments} complete ({calls} API call(s), run cost ${cost:.4f})", file=sys.stderr, flush=True)

    print(json.dumps(run.run(on_event=report), indent=2))


if __name__ == "__main__":
    main()
