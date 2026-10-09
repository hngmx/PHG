import unittest

import torch

from net import Net
from solution_graph import (
    SolutionArchive,
    normalize_heatmap_rows,
    population_quality_kl,
)


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

    def test_s1_cost_supervision_updates_graph_output_not_initial_gnn(self):
        model = Net(solution_graph_hidden=8, solution_graph_layers=2)
        refined, _, _, _ = model.refine_heatmap(
            self.previous,
            self.distances,
            self.archive,
            return_components=True,
        )
        paths, costs, _ = self.archive.tensors()
        loss = population_quality_kl(refined, paths.transpose(0, 1), costs)
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

    def test_refinement_ablation_modes_separate_components(self):
        model = Net(solution_graph_hidden=8, solution_graph_layers=2)

        h0, h0_base, h0_residual, h0_graph = model.refine_heatmap(
            self.previous,
            self.distances,
            self.archive,
            refinement_mode="h0",
            return_components=True,
        )
        deterministic, deterministic_base, deterministic_residual, graph = (
            model.refine_heatmap(
                self.previous,
                self.distances,
                self.archive,
                refinement_mode="deterministic",
                return_components=True,
            )
        )
        learned, learned_base, _, learned_graph = model.refine_heatmap(
            self.previous,
            self.distances,
            self.archive,
            refinement_mode="learned",
            return_components=True,
        )

        normalized_previous = normalize_heatmap_rows(self.previous)
        self.assertTrue(torch.allclose(h0, normalized_previous))
        self.assertTrue(torch.allclose(h0_base, normalized_previous))
        self.assertTrue(torch.equal(h0_residual, torch.zeros_like(h0_residual)))
        self.assertIsNone(h0_graph)
        self.assertTrue(torch.allclose(deterministic, deterministic_base))
        self.assertTrue(
            torch.equal(
                deterministic_residual,
                torch.zeros_like(deterministic_residual),
            )
        )
        self.assertIsNone(graph)
        self.assertTrue(torch.allclose(learned_base, normalized_previous))
        self.assertTrue(torch.allclose(learned, normalized_previous))
        self.assertIsNotNone(learned_graph)

        with self.assertRaises(ValueError):
            model.refine_heatmap(
                self.previous,
                self.distances,
                self.archive,
                refinement_mode="unknown",
            )


if __name__ == "__main__":
    unittest.main()
