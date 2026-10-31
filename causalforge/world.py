from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random


FACTOR_NAMES = ("temperature", "pressure", "catalyst", "field")
LEVELS = (-1, 0, 1)


@dataclass(frozen=True)
class Intervention:
    temperature: int
    pressure: int
    catalyst: int
    field: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value not in LEVELS:
                raise ValueError(f"{name} must be one of {LEVELS}")

    def vector(self) -> list[float]:
        return [1.0, *(float(v) for v in asdict(self).values())]


@dataclass(frozen=True)
class Observation:
    experiment_id: int
    intervention: Intervention
    response: float

    def public_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "intervention": asdict(self.intervention),
            "response": self.response,
        }


@dataclass(frozen=True)
class PublicWorld:
    seed: int
    factors: tuple[str, ...] = FACTOR_NAMES
    levels: tuple[int, ...] = LEVELS
    noise_std: float = 0.03

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _HiddenRule:
    intercept: float
    coefficients: tuple[float, ...]


class Simulator:
    """Authoritative simulator. Hidden rule is never included in public state."""

    def __init__(self, public: PublicWorld, hidden: _HiddenRule):
        self.public = public
        self.__hidden = hidden
        self._observations: list[Observation] = []

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def run(self, intervention: Intervention) -> Observation:
        experiment_id = len(self._observations) + 1
        x = intervention.vector()
        mean = self.__hidden.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.__hidden.coefficients, x[1:])
        )
        digest = hashlib.sha256(
            f"{self.public.seed}:{experiment_id}:{x[1:]}".encode("utf-8")
        ).digest()
        noise_seed = int.from_bytes(digest[:8], "big")
        noise = random.Random(noise_seed).gauss(0.0, self.public.noise_std)
        observation = Observation(experiment_id, intervention, round(mean + noise, 6))
        self._observations.append(observation)
        return observation

    def public_snapshot(self) -> dict:
        return {
            "world": self.public.as_dict(),
            "observations": [item.public_dict() for item in self._observations],
        }

    def reveal(self) -> dict:
        return {
            "intercept": self.__hidden.intercept,
            "coefficients": dict(zip(FACTOR_NAMES, self.__hidden.coefficients)),
        }


class WorldGenerator:
    @staticmethod
    def validate_rule(intercept: float, coefficients: tuple[float, ...]) -> None:
        if len(coefficients) != len(FACTOR_NAMES):
            raise ValueError("one coefficient per factor is required")
        active = [value for value in coefficients if value != 0]
        if len(active) != 2:
            raise ValueError("world must contain exactly two active causal factors")
        if any(abs(value) < 0.75 for value in active):
            raise ValueError("active effects are too weak to identify within the experiment budget")
        if abs(intercept) > 0.4:
            raise ValueError("intercept is outside the supported identifiable range")

    @staticmethod
    def generate(seed: int) -> Simulator:
        rng = random.Random(seed)
        active = set(rng.sample(range(len(FACTOR_NAMES)), 2))
        magnitudes = (0.75, 1.0, 1.25, 1.5)
        coefficients = tuple(
            rng.choice(magnitudes) * rng.choice((-1.0, 1.0)) if i in active else 0.0
            for i in range(len(FACTOR_NAMES))
        )
        intercept = round(rng.uniform(-0.4, 0.4), 3)
        WorldGenerator.validate_rule(intercept, coefficients)
        assert math.isclose(sum(c != 0 for c in coefficients), 2)
        return Simulator(PublicWorld(seed=seed), _HiddenRule(intercept, coefficients))
