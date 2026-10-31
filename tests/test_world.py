import json
import unittest

from causalforge.world import Intervention, WorldGenerator


class WorldTests(unittest.TestCase):
    def test_same_seed_and_interventions_are_reproducible(self):
        left = WorldGenerator.generate(41)
        right = WorldGenerator.generate(41)
        interventions = [Intervention(-1, 0, 1, 0), Intervention(1, -1, 0, 1)]
        self.assertEqual([left.run(x) for x in interventions], [right.run(x) for x in interventions])

    def test_twenty_seeds_have_two_active_factors_and_vary(self):
        rules = [WorldGenerator.generate(seed).reveal() for seed in range(20)]
        self.assertTrue(all(sum(v != 0 for v in rule["coefficients"].values()) == 2 for rule in rules))
        self.assertGreater(len({json.dumps(rule, sort_keys=True) for rule in rules}), 15)

    def test_hidden_truth_is_not_in_public_snapshot(self):
        simulator = WorldGenerator.generate(5)
        simulator.run(Intervention(1, 0, -1, 0))
        public = json.dumps(simulator.public_snapshot())
        self.assertNotIn("coefficients", public)
        self.assertNotIn("intercept", public)

    def test_invalid_intervention_cannot_mutate_state(self):
        simulator = WorldGenerator.generate(1)
        with self.assertRaises(ValueError):
            Intervention(2, 0, 0, 0)
        self.assertEqual(simulator.observations, ())
