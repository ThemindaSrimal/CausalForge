import json
from pathlib import Path
import tempfile
import unittest

from causalforge.replay import ReplaySession, export_replay


ROOT = Path(__file__).resolve().parent.parent


class ReplayTests(unittest.TestCase):
    def test_checked_in_replay_is_real_and_clearly_labeled(self):
        payload = json.loads((ROOT / "replays" / "recorded-seed-104729.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["source"], "Recorded real Meta Muse Spark API session")
        self.assertEqual(len(payload["events"]), 10)
        self.assertTrue(all(event["api"]["response_id"].startswith("chatcmpl-") for event in payload["events"]))

    def test_replay_advances_without_changing_recorded_values(self):
        payload = json.loads((ROOT / "replays" / "recorded-seed-104729.json").read_text(encoding="utf-8"))
        replay = ReplaySession(payload)
        self.assertEqual(replay.snapshot()["experiment_count"], 0)
        replay.step()
        state = replay.snapshot()
        self.assertEqual(state["mode"], "replay")
        self.assertEqual(state["experiment_count"], 1)
        self.assertEqual(state["render"]["entities"][-1]["value"], payload["events"][0]["observation"]["response"])
        self.assertIsNone(state["revealed_rule"])
        replay.reveal()
        self.assertEqual(replay.snapshot()["revealed_rule"], payload["result"]["actual_rule"])

    def test_export_rejects_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.jsonl"
            source.write_text('{"type":"experiment"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                export_replay(source, Path(directory) / "out.json", "bad")
