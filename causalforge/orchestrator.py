from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .budget import SpendLedger
from .learner import BayesianLinearLearner
from .meta_api import MetaAPIError, MuseSparkClient
from .world import Intervention, LEVELS, WorldGenerator


class BudgetExceeded(RuntimeError):
    pass


class DiscoveryRun:
    def __init__(self, seed: int, client: MuseSparkClient, max_experiments: int = 10, max_cost_usd: float = 1.0, log_path: Path | None = None, spend_ledger: SpendLedger | None = None):
        if max_experiments < 1 or max_cost_usd <= 0:
            raise ValueError("positive experiment and cost limits are required")
        self.simulator = WorldGenerator.generate(seed)
        self.learner = BayesianLinearLearner()
        self.client = client
        self.max_experiments = max_experiments
        self.max_cost_usd = max_cost_usd
        self.total_cost_usd = 0.0
        self.log_path = log_path
        self.spend_ledger = spend_ledger
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")

    def _context(self) -> dict:
        candidates = []
        seen = {tuple(item.intervention.vector()[1:]) for item in self.simulator.observations}
        for t in LEVELS:
            for p in LEVELS:
                for c in LEVELS:
                    for f in LEVELS:
                        intervention = Intervention(t, p, c, f)
                        values = tuple(intervention.vector()[1:])
                        if values not in seen:
                            prediction = self.learner.predict(intervention)
                            candidates.append({"intervention": asdict(intervention), "predicted_mean": prediction.mean, "uncertainty": prediction.uncertainty})
        candidates.sort(key=lambda item: item["uncertainty"], reverse=True)
        return {
            "public_world": self.simulator.public.as_dict(),
            "observations": [item.public_dict() for item in self.simulator.observations],
            "local_learner": self.learner.summary(),
            "highest_uncertainty_candidates": candidates[:12],
            "remaining_experiments": self.max_experiments - len(self.simulator.observations),
        }

    def _log(self, event: dict) -> None:
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")

    def step(self) -> dict:
        if len(self.simulator.observations) >= self.max_experiments:
            raise StopIteration("experiment limit reached")
        if self.total_cost_usd >= self.max_cost_usd:
            raise BudgetExceeded("run cost limit reached")
        # The request is capped at 8192 output tokens and a compact context. Reserving
        # $0.25 is deliberately far above its expected price and prevents overshoot.
        if self.spend_ledger:
            self.spend_ledger.require_capacity(0.25)
        try:
            decision = self.client.choose_experiment(self._context())
        except MetaAPIError as error:
            self.total_cost_usd += error.estimated_cost_usd
            project_cost = self.spend_ledger.record(error.estimated_cost_usd) if self.spend_ledger else None
            self._log({
                "type": "api_error",
                "message": str(error),
                "estimated_cost_usd": error.estimated_cost_usd,
                "cumulative_cost_usd": round(self.total_cost_usd, 8),
                "project_cumulative_cost_usd": project_cost,
            })
            raise
        projected = self.total_cost_usd + decision.usage.estimated_cost_usd
        if projected > self.max_cost_usd:
            raise BudgetExceeded("latest API call would exceed run cost limit; experiment not executed")
        observation = self.simulator.run(decision.intervention)
        before = self.learner.summary()
        self.learner.update(observation)
        self.total_cost_usd = projected
        project_cost = self.spend_ledger.record(decision.usage.estimated_cost_usd) if self.spend_ledger else None
        event = {
            "type": "experiment",
            "api": {"response_id": decision.response_id, "model": decision.model, "calls": decision.api_calls, "latency_seconds": round(decision.latency_seconds, 3), "usage": asdict(decision.usage)},
            "decision": {"intervention": asdict(decision.intervention), "rationale": decision.rationale},
            "observation": observation.public_dict(),
            "learner_before": before,
            "learner_after": self.learner.summary(),
            "cumulative_cost_usd": round(self.total_cost_usd, 8),
            "project_cumulative_cost_usd": project_cost,
        }
        self._log(event)
        return event

    def run(self, on_event=None) -> dict:
        events = []
        for _ in range(self.max_experiments):
            event = self.step()
            events.append(event)
            if on_event:
                on_event(event)
        result = {
            "seed": self.simulator.public.seed,
            "experiments": len(events),
            "estimated_rule": self.learner.summary(),
            "actual_rule": self.simulator.reveal(),
            "total_cost_usd": round(self.total_cost_usd, 8),
        }
        self._log({"type": "result", **result})
        return result
