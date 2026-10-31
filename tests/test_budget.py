from pathlib import Path
import tempfile
import unittest

from causalforge.budget import SpendLedger


class BudgetTests(unittest.TestCase):
    def test_persists_spend_and_stops_before_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spend.json"
            first = SpendLedger(path, hard_limit_usd=1.0)
            self.assertEqual(first.record(0.6), 0.6)
            second = SpendLedger(path, hard_limit_usd=1.0)
            self.assertEqual(second.spent_usd, 0.6)
            with self.assertRaises(RuntimeError):
                second.require_capacity(0.41)

    def test_rejects_limit_at_project_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                SpendLedger(Path(directory) / "spend.json", hard_limit_usd=150.0)
