import unittest
from unittest.mock import patch

import torch

from aco import ACO
from solution_sampler import ACOSolutionSampler


class ACOWithoutLocalSearchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        coordinates = torch.rand(6, 2)
        self.distances = torch.cdist(coordinates, coordinates)
        self.distances.fill_diagonal_(1e9)
        self.heatmap = 1 / self.distances

    def test_aco_disables_local_search_by_default(self):
        aco = ACO(
            distances=self.distances,
            heuristic=self.heatmap,
            n_ants=4,
        )

        with patch.object(
            aco,
            "local_search",
            side_effect=AssertionError("local search must remain disabled"),
        ):
            raw_costs, feasible_costs, _, raw_paths, feasible_paths = (
                aco.sample_iteration(require_prob=False)
            )

        self.assertIs(feasible_paths, raw_paths)
        self.assertIs(feasible_costs, raw_costs)

    def test_solution_sampler_disables_local_search_by_default(self):
        sampler = ACOSolutionSampler(
            n_solutions=4,
            heatmap=self.heatmap,
            distances=self.distances,
            device="cpu",
        )

        self.assertIsNone(sampler._aco.local_search_type)


if __name__ == "__main__":
    unittest.main()
