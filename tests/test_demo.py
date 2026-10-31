import unittest

from causalforge.demo import preflight


class DemoTests(unittest.TestCase):
    def test_offline_preflight_is_ready(self):
        report = preflight(port=0, check_meta=False)
        self.assertTrue(report["replay_ready"])
        self.assertTrue(report["checks"]["web_assets"]["ok"])
        self.assertTrue(report["checks"]["offline_replay"]["ok"])
