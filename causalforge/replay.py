from __future__ import annotations

import json
from pathlib import Path

from .presentation import render_state
from .world import Intervention, Observation


def export_replay(source: Path, destination: Path, title: str) -> dict:
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [record for record in records if record.get("type") == "experiment"]
    result = next((record for record in records if record.get("type") == "result"), None)
    if not events or not result:
        raise ValueError("source log must contain experiment and result records")
    payload = {"format": "causalforge-replay-v1", "title": title, "source": "Recorded real Meta Muse Spark API session", "seed": result["seed"], "events": events, "result": result}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class ReplaySession:
    def __init__(self, payload: dict):
        if payload.get("format") != "causalforge-replay-v1" or not payload.get("events"):
            raise ValueError("unsupported or empty replay")
        self.payload = payload
        self.index = 0
        self.revealed = False

    def step(self) -> None:
        if self.index >= len(self.payload["events"]):
            raise StopIteration("replay complete")
        self.index += 1

    def reveal(self) -> None:
        self.revealed = True

    def snapshot(self) -> dict:
        events = self.payload["events"][:self.index]
        latest = events[-1] if events else None
        if latest:
            raw = latest["observation"]
            observation = Observation(raw["experiment_id"], Intervention(**raw["intervention"]), raw["response"])
            learner, cost = latest["learner_after"], latest["cumulative_cost_usd"]
        else:
            observation, cost = None, 0.0
            learner = {"training_rows": 0, "estimated_intercept": 0.0, "estimated_coefficients": {name: 0.0 for name in ("temperature", "pressure", "catalyst", "field")}}
        return {
            "status": "complete" if self.index >= len(self.payload["events"]) else "ready", "busy": False,
            "mode": "replay", "replay_title": self.payload["title"], "replay_source": self.payload["source"],
            "seed": self.payload["seed"], "experiment_count": self.index, "max_experiments": len(self.payload["events"]),
            "observations": [event["observation"] for event in events], "learner": learner, "last_event": latest,
            "total_cost_usd": cost, "revealed_rule": self.payload["result"]["actual_rule"] if self.revealed else None,
            "render": render_state(observation),
        }
