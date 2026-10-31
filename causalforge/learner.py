from __future__ import annotations

from dataclasses import dataclass
import math

from .world import FACTOR_NAMES, Intervention, Observation


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    augmented = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [row[n:] for row in augmented]


@dataclass(frozen=True)
class Prediction:
    mean: float
    uncertainty: float


class BayesianLinearLearner:
    """Small-data Bayesian ridge learner updated only from observations."""

    def __init__(self, ridge: float = 0.2, noise_variance: float = 0.0009):
        self.ridge = ridge
        self.noise_variance = noise_variance
        self._observations: list[Observation] = []
        self._weights = [0.0] * (len(FACTOR_NAMES) + 1)
        self._covariance = [[0.0] * len(self._weights) for _ in self._weights]
        for i in range(len(self._weights)):
            self._covariance[i][i] = 1.0 / ridge

    @property
    def training_rows(self) -> int:
        return len(self._observations)

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(self._weights)

    def update(self, observation: Observation) -> None:
        self._observations.append(observation)
        size = len(self._weights)
        precision = [[self.ridge * float(i == j) for j in range(size)] for i in range(size)]
        rhs = [0.0] * size
        for item in self._observations:
            x = item.intervention.vector()
            for i in range(size):
                rhs[i] += x[i] * item.response
                for j in range(size):
                    precision[i][j] += x[i] * x[j]
        self._covariance = _inverse(precision)
        self._weights = [sum(self._covariance[i][j] * rhs[j] for j in range(size)) for i in range(size)]

    def predict(self, intervention: Intervention) -> Prediction:
        x = intervention.vector()
        mean = sum(weight * value for weight, value in zip(self._weights, x))
        variance = self.noise_variance + sum(
            x[i] * self._covariance[i][j] * x[j]
            for i in range(len(x)) for j in range(len(x))
        )
        return Prediction(round(mean, 6), round(math.sqrt(max(variance, 0.0)), 6))

    def summary(self) -> dict:
        return {
            "training_rows": self.training_rows,
            "estimated_intercept": round(self._weights[0], 6),
            "estimated_coefficients": {
                name: round(value, 6) for name, value in zip(FACTOR_NAMES, self._weights[1:])
            },
        }
