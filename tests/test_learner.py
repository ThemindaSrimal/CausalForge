import unittest

from causalforge.learner import BayesianLinearLearner
from causalforge.world import Intervention, WorldGenerator


class LearnerTests(unittest.TestCase):
    def test_learns_from_each_observation(self):
        simulator = WorldGenerator.generate(22)
        learner = BayesianLinearLearner()
        probe = Intervention(1, 1, 1, 1)
        initial = learner.predict(probe)
        designs = [
            Intervention(0, 0, 0, 0), Intervention(1, 0, 0, 0), Intervention(-1, 0, 0, 0),
            Intervention(0, 1, 0, 0), Intervention(0, -1, 0, 0), Intervention(0, 0, 1, 0),
            Intervention(0, 0, -1, 0), Intervention(0, 0, 0, 1), Intervention(0, 0, 0, -1),
        ]
        for index, intervention in enumerate(designs, 1):
            learner.update(simulator.run(intervention))
            self.assertEqual(learner.training_rows, index)
        learned = learner.predict(probe)
        self.assertNotEqual(initial.mean, learned.mean)
        self.assertLess(learned.uncertainty, initial.uncertainty)

    def test_recovers_rule_on_unseen_seeds(self):
        designs = [Intervention(0, 0, 0, 0)]
        for factor in range(4):
            for level in (-1, 1):
                values = [0, 0, 0, 0]
                values[factor] = level
                designs.append(Intervention(*values))
        successes = 0
        for seed in range(10):
            simulator = WorldGenerator.generate(1000 + seed)
            learner = BayesianLinearLearner(ridge=0.01)
            for design in designs:
                learner.update(simulator.run(design))
            actual = simulator.reveal()["coefficients"]
            expected_active = {key for key, value in actual.items() if value != 0}
            estimates = learner.summary()["estimated_coefficients"]
            predicted_active = set(sorted(estimates, key=lambda key: abs(estimates[key]), reverse=True)[:2])
            successes += predicted_active == expected_active
        self.assertGreaterEqual(successes, 8)
