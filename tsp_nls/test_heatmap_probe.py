import os
import random
import tempfile
import unittest

import numpy as np
import torch

from heatmap_probe import HeatmapProbe
from net import Net


class HeatmapProbeTest(unittest.TestCase):
    def test_three_paired_charts_do_not_advance_training_rng_or_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = HeatmapProbe(
                nodes=8, count=1, repeats=2, seed=654321,
                output=os.path.join(directory, "probe"), ants=3, k_sparse=2,
                local_search=None, refinement_mode="local_gcn",
                max_solution_graph_solutions=None, max_archive_solutions=None,
                max_archive_rounds=None, graph_structure="whole_tour",
                graph_layers=2, graph_quality_mode=None,
                graph_quality_temperature=0.1,
                graph_diversity_strength=0.5,
                graph_candidate_only=False,
            )
            model = Net()
            model.train()
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.get_rng_state().clone()

            first = probe.evaluate(1, model)
            self.assertTrue(model.training)
            self.assertEqual(random.getstate(), python_state)
            self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            self.assertEqual(probe.pool[0].state.archive.num_rounds, 1)
            self.assertEqual(len(probe.pool[0].state.archive), 3)
            self.assertEqual(len(first["ratios"]), 3)

            probe.evaluate(2, model)
            self.assertEqual(probe.pool[0].state.archive.num_rounds, 2)
            self.assertEqual(len(probe.pool[0].state.archive), 6)
            for name in ("h1_over_h0", "prior_over_h0", "h1_over_prior"):
                self.assertTrue(os.path.isfile(os.path.join(probe.output, name + ".png")))
                self.assertGreater(first["ratios"][name]["raw"]["mean_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
