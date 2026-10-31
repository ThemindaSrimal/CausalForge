from __future__ import annotations

import json
from pathlib import Path


class SpendLedger:
    """Persistent cumulative Meta spend guard shared by all runs."""

    def __init__(self, path: Path, hard_limit_usd: float = 120.0):
        if hard_limit_usd <= 0 or hard_limit_usd >= 150:
            raise ValueError("hard limit must be positive and below the $150 project ceiling")
        self.path = path
        self.hard_limit_usd = hard_limit_usd

    @property
    def spent_usd(self) -> float:
        if not self.path.exists():
            return 0.0
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return float(data["spent_usd"])

    def require_capacity(self, maximum_call_cost_usd: float) -> None:
        if self.spent_usd + maximum_call_cost_usd > self.hard_limit_usd:
            raise RuntimeError("project Meta API spend guard would be exceeded")

    def record(self, cost_usd: float) -> float:
        if cost_usd < 0:
            raise ValueError("cost cannot be negative")
        updated = self.spent_usd + cost_usd
        if updated > self.hard_limit_usd:
            raise RuntimeError("provider-reported cost exceeds project spend guard")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"spent_usd": round(updated, 8)}), encoding="utf-8")
        temporary.replace(self.path)
        return updated
