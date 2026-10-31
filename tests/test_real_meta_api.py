import os
from pathlib import Path
import tempfile
import unittest

from causalforge.config import load_dotenv
from causalforge.meta_api import MuseSparkClient
from causalforge.orchestrator import DiscoveryRun

load_dotenv()

@unittest.skipUnless(os.environ.get("MODEL_API_KEY"), "MODEL_API_KEY is required for the real paid integration test")
class RealMetaIntegrationTests(unittest.TestCase):
    def test_real_tool_call_executes_and_trains(self):
        with tempfile.TemporaryDirectory() as directory:
            run = DiscoveryRun(8675309, MuseSparkClient(), max_experiments=1, max_cost_usd=0.25, log_path=Path(directory) / "run.jsonl")
            event = run.step()
            self.assertEqual(event["type"], "experiment")
            self.assertEqual(event["learner_after"]["training_rows"], 1)
            self.assertTrue(event["api"]["response_id"])
            self.assertIn("muse-spark", event["api"]["model"])
            self.assertGreaterEqual(event["api"]["calls"], 1)
            self.assertLessEqual(event["api"]["calls"], 3)
            self.assertGreater(event["api"]["usage"]["input_tokens"], 0)
            self.assertLessEqual(event["cumulative_cost_usd"], 0.25)
