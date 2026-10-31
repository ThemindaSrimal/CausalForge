import unittest

from causalforge.world import WorldGenerator


class GeneratorValidationTests(unittest.TestCase):
    def test_rejects_degenerate_rules(self):
        with self.assertRaises(ValueError):
            WorldGenerator.validate_rule(0.0, (0.1, 0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            WorldGenerator.validate_rule(0.0, (0.5, 0.5, 0.0, 0.0))
        with self.assertRaises(ValueError):
            WorldGenerator.validate_rule(0.8, (1.0, 1.0, 0.0, 0.0))

    def test_one_hundred_generated_rules_are_valid(self):
        for seed in range(100):
            rule = WorldGenerator.generate(seed).reveal()
            WorldGenerator.validate_rule(rule["intercept"], tuple(rule["coefficients"].values()))
