import unittest

import torch

from net import Net
from solution_graph import SolutionArchive, future_solution_quality_kl


class LearnableSolutionGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.archive = SolutionArchive(n_nodes=5)
        self.archive.add(
            torch.tensor(
                [
                    [0, 0],
                    [1, 2],
                    [2, 4],
                    [3, 1],
                    [4, 3],
                ]
            ),
            torch.tensor([5.0, 6.0]),
        )
        self.previous = torch.ones(5, 5) - torch.eye(5)
        self.distances = torch.tensor(
            [
                [1e9, 1.0, 2.0, 3.0, 4.0],
                [1.0, 1e9, 2.0, 3.0, 4.0],
                [2.0, 2.0, 1e9, 3.0, 4.0],
                [3.0, 3.0, 3.0, 1e9, 4.0],
                [4.0, 4.0, 4.0, 4.0, 1e9],
            ]
        )

    def test_zero_initialized_residual_starts_from_deterministic_base(self):
        model = Net(solution_graph_hidden=8, solution_graph_layers=2)
        refined, base, residual, graph = model.refine_heatmap(
            self.previous,
            self.distances,
            self.archive,
            return_components=True,
        )

        self.assertEqual(graph["n_solutions"], 2)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.allclose(refined, base, atol=1e-6))
        self.assertTrue(
            torch.allclose(refined.sum(dim=-1), torch.ones(5), atol=1e-6)
        )

    def test_future_supervision_updates_graph_output_not_initial_gnn(self):
        model = Net(solution_graph_hidden=8, solution_graph_layers=2)
        refined, _, _, _ = model.refine_heatmap(
            self.previous,
            self.distances,
            self.archive,
            return_components=True,
        )
        future_target = torch.rand(5, 5) + 0.1
        loss = future_solution_quality_kl(refined, future_target)
        loss.backward()

        output_grad = model.solution_graph_net.output.weight.grad
        self.assertIsNotNone(output_grad)
        self.assertGreater(float(output_grad.abs().sum()), 0.0)
        self.assertIsNone(model.emb_net.v_lin0.weight.grad)

    def test_graph_residual_is_bounded(self):
        model = Net(solution_graph_hidden=8, solution_graph_layers=2)
        with torch.no_grad():
            model.solution_graph_net.output.bias.fill_(100.0)
        _, _, residual, _ = model.refine_heatmap(
            self.previous,
            self.distances,
            self.archive,
            return_components=True,
        )

        self.assertLessEqual(float(residual.abs().max()), 0.25 + 1e-7)


if __name__ == "__main__":
    unittest.main()
