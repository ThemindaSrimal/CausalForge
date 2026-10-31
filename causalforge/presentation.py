from __future__ import annotations

from dataclasses import asdict

from .world import FACTOR_NAMES, Intervention, Observation


FACTOR_COLORS = {
    "temperature": "#ff6b4a",
    "pressure": "#4ca7ff",
    "catalyst": "#ffd166",
    "field": "#a77bff",
}


def render_state(observation: Observation | None) -> dict:
    """Convert authoritative simulation values into renderer-ready state."""
    intervention = observation.intervention if observation else Intervention(0, 0, 0, 0)
    values = asdict(intervention)
    entities = []
    for index, name in enumerate(FACTOR_NAMES):
        value = values[name]
        entities.append({
            "id": f"factor:{name}",
            "kind": "factor",
            "label": name,
            "value": value,
            "position": {"x": (index - 1.5) * 2.2, "y": 0.0, "z": 0.0},
            "color": FACTOR_COLORS[name],
        })
    response = observation.response if observation else 0.0
    entities.append({
        "id": "measurement:response",
        "kind": "response",
        "label": "measured response",
        "value": response,
        "position": {"x": 0.0, "y": 1.0, "z": -2.8},
        "color": "#73fbd3" if response >= 0 else "#ff4d8d",
    })
    return {"experiment_id": observation.experiment_id if observation else 0, "entities": entities}
