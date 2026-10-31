import unittest

from causalforge.evaluation import score_run
from causalforge.learner import BayesianLinearLearner
from causalforge.world import Intervention, WorldGenerator


class EvaluationTests(unittest.TestCase):
    def test_score_uses_unseen_intervention_grid(self):
        class CompletedRun:
            simulator = WorldGenerator.generate(90210)
            learner = BayesianLinearLearner(ridge=0.01)
            total_cost_usd = 0.0

        run = CompletedRun()
        designs = [Intervention(0, 0, 0, 0)]
        for factor in range(4):
            for level in (-1, 1):
                values = [0, 0, 0, 0]
                values[factor] = level
                designs.append(Intervention(*values))
        for design in designs:
            run.learner.update(run.simulator.run(design))
        score = score_run(run)
        self.assertTrue(score["exact_active_factors"])
        self.assertLess(score["coefficient_rmse"], 0.1)
        self.assertLess(score["prediction_rmse"], 0.1)
