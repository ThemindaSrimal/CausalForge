import unittest

from causalforge.presentation import render_state
from causalforge.world import Intervention, WorldGenerator


class PresentationTests(unittest.TestCase):
    def test_render_entities_match_authoritative_observation(self):
        simulator = WorldGenerator.generate(77)
        intervention = Intervention(-1, 0, 1, -1)
        observation = simulator.run(intervention)
        state = render_state(observation)
        by_id = {entity["id"]: entity for entity in state["entities"]}
        self.assertEqual(by_id["factor:temperature"]["value"], -1)
        self.assertEqual(by_id["factor:pressure"]["value"], 0)
        self.assertEqual(by_id["factor:catalyst"]["value"], 1)
        self.assertEqual(by_id["factor:field"]["value"], -1)
        self.assertEqual(by_id["measurement:response"]["value"], observation.response)
        self.assertEqual(state["experiment_id"], observation.experiment_id)

    def test_initial_render_state_has_stable_entity_ids(self):
        ids = [entity["id"] for entity in render_state(None)["entities"]]
        self.assertEqual(ids, [
            "factor:temperature", "factor:pressure", "factor:catalyst",
            "factor:field", "measurement:response",
        ])
