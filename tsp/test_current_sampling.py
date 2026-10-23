import unittest

import torch

from aco import ACO
from graph_net import GraphNet
from train import train_instance


class CurrentSamplingTest(unittest.TestCase):
    def test_complete_s0_updates_pheromone_and_keeps_log_probabilities(self):
        torch.manual_seed(123)
        coordinates = torch.rand(8, 2)
        distances = torch.cdist(coordinates, coordinates)
        distances.fill_diagonal_(1e9)
        heuristic = (1 / distances).requires_grad_()
        aco = ACO(
            distances=distances, heuristic=heuristic, n_ants=4,
        )
        raw_costs, feasible_costs, log_probs, raw_paths, feasible_paths = (
            aco.sample_iteration(require_prob=True)
        )
        self.assertTrue(torch.equal(raw_paths, feasible_paths))
        self.assertTrue(torch.equal(raw_costs, feasible_costs))
        self.assertEqual(tuple(log_probs.shape), (7, 4))
        self.assertIsNone(aco.start_node)
        self.assertTrue(torch.all((raw_paths[0] >= 0) & (raw_paths[0] < 8)))
        self.assertGreater(torch.unique(raw_paths[0]).numel(), 1)
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.allclose(aco.lowest_cost, feasible_costs.min()))
        self.assertFalse(torch.allclose(aco.pheromone, torch.ones_like(distances)))
        (log_probs.sum() * 0.01).backward()
        self.assertIsNotNone(heuristic.grad)

    def test_training_builds_h1_from_post_s0_pheromone(self):
        torch.manual_seed(456)
        model = GraphNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        coordinates = torch.rand(8, 2)
        original_refine = model.refine_heatmap
        observed = {}

        def capture_refine(*args, **kwargs):
            pheromone = kwargs["pheromone"]
            observed["pheromone"] = pheromone.detach().clone()
            return original_refine(*args, **kwargs)

        model.refine_heatmap = capture_refine
        archive, losses = train_instance(
            model, optimizer, coordinates, n_ants=4, k_sparse=3,
        )
        self.assertEqual(len(archive), 4)
        self.assertFalse(torch.allclose(
            observed["pheromone"], torch.ones_like(observed["pheromone"]),
        ))
        self.assertEqual(set(losses), {"reinforce", "kl", "graph"})
        same_archive, _ = train_instance(
            model, optimizer, coordinates, n_ants=4, k_sparse=3,
            archive=archive,
        )
        self.assertIs(same_archive, archive)
        self.assertEqual(len(archive), 8)


if __name__ == "__main__":
    unittest.main()
