import unittest

import torch

from net import Net
from solution_graph import (
    SolutionArchive,
    build_local_edge_gcn_graph,
    normalize_heatmap_rows,
    refinement_distillation_kl,
    rowwise_heatmap_kl,
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

    def test_local_gcn_starts_from_archive_edge_support(self):
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

    def test_h1_distillation_updates_h0_not_graph_output(self):
        model = Net(solution_graph_hidden=8, solution_graph_layers=2)
        initial = self.previous.clone().requires_grad_()
        refined, _, _, _ = model.refine_heatmap(
            initial,
            self.distances,
            self.archive,
            return_components=True,
        )
        loss = refinement_distillation_kl(initial, refined)
        loss.backward()

        self.assertIsNotNone(initial.grad)
        self.assertGreater(float(initial.grad.abs().sum()), 0.0)
        self.assertIsNone(model.local_edge_gcn.output.weight.grad)
        self.assertIsNone(model.solution_graph_net.output.weight.grad)

    def test_graph_residual_is_bounded(self):
        model = Net(solution_graph_hidden=8, solution_graph_layers=2)
        with torch.no_grad():
            model.solution_graph_net.output.bias.fill_(100.0)
        _, _, residual, _ = model.refine_heatmap(
            self.previous,
            self.distances,
            self.archive,
            refinement_mode="full",
            return_components=True,
        )

        self.assertLessEqual(float(residual.abs().max()), 0.25 + 1e-7)

    def test_local_graph_keeps_unseen_candidates_and_compatible_fragments(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        graph = build_local_edge_gcn_graph(
            self.previous, self.distances, archive,
        )
        keys = list(zip(graph["edge_u"].tolist(), graph["edge_v"].tolist()))
        relations = {
            (keys[source], keys[target]): features
            for (source, target), features in zip(
                graph["edge_index"].T.tolist(),
                graph["relation_features"].tolist(),
            )
        }
        self.assertIn((0, 2), keys)  # not sampled in this tour
        self.assertGreater(relations[((0, 1), (1, 2))][0], 0)
        self.assertGreater(relations[((0, 1), (0, 4))][0], 0)  # cyclic closing
        self.assertEqual(relations[((0, 1), (0, 2))][0], 0)
        self.assertEqual(relations[((0, 1), (0, 2))][1], 0)
        self.assertNotIn(((0, 1), (2, 3)), relations)
        self.assertTrue(torch.all(graph["edge_weight"] > 0))

    def test_pheromone_guides_top_d_selection_without_discarding_samples(self):
        archive = SolutionArchive(n_nodes=6)
        archive.add(
            torch.arange(6)[:, None], torch.tensor([6.0]),
        )
        heatmap = torch.ones(6, 6) - torch.eye(6)
        distances = torch.ones(6, 6) + torch.eye(6) * 1e9
        heatmap[0, 3] = heatmap[3, 0] = 10.0
        heatmap[3, 4] = heatmap[4, 3] = 50.0
        pheromone = torch.ones(6, 6)
        pheromone[0, 4] = pheromone[4, 0] = 20.0
        graph = build_local_edge_gcn_graph(
            heatmap, distances, archive, pheromone=pheromone, incident_topk=1,
        )
        keys = set(zip(graph["edge_u"].tolist(), graph["edge_v"].tolist()))
        self.assertIn((0, 4), keys)
        self.assertNotIn((0, 3), keys)
        self.assertEqual(graph["relation_features"].size(1), 7)

    def test_fragment_features_can_train_relation_gate(self):
        model = Net(local_gcn_hidden=8)
        with torch.no_grad():
            model.local_edge_gcn.output.weight.fill_(0.1)
        refined, _, _, graph = model.refine_heatmap(
            self.previous, self.distances, self.archive, return_components=True,
        )
        rowwise_heatmap_kl(refined, graph["quality_target"]).backward()
        gate_gradient = model.local_edge_gcn.relation_gate[-1].weight.grad
        self.assertIsNotNone(gate_gradient)
        self.assertGreater(float(gate_gradient.abs().sum()), 0.0)

    def test_default_local_weights_reward_quality_and_structural_novelty(self):
        better = [0, 1, 2, 3, 4]
        different = [0, 2, 4, 1, 3]
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([better, better, different]).T,
            torch.tensor([5.0, 5.0, 5.0]),
        )
        graph = build_local_edge_gcn_graph(
            self.previous, self.distances, archive,
        )
        weights = graph["solution_weights"]
        self.assertAlmostEqual(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[2]), float(weights[0]))
        self.assertTrue(torch.isfinite(graph["quality_target"]).all())

        no_diversity = build_local_edge_gcn_graph(
            self.previous, self.distances, archive,
            quality_mode="relative_diversity", diversity_strength=0,
        )
        self.assertTrue(torch.allclose(
            no_diversity["solution_weights"], torch.ones(3),
        ))
        self.assertFalse(torch.allclose(
            graph["edge_prior"], no_diversity["edge_prior"],
        ))
        self.assertFalse(torch.allclose(
            graph["edge_weight"], no_diversity["edge_weight"],
        ))

        quality_archive = SolutionArchive(n_nodes=5)
        quality_archive.add(
            torch.tensor([better, different]).T,
            torch.tensor([5.0, 6.0]),
        )
        quality_graph = build_local_edge_gcn_graph(
            self.previous, self.distances, quality_archive,
        )
        self.assertGreater(
            float(quality_graph["solution_weights"][0]),
            float(quality_graph["solution_weights"][1]),
        )

    def test_legacy_inverse_cost_weights_remain_available(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([[0, 0], [1, 2], [2, 1], [3, 3], [4, 4]]),
            torch.tensor([5.0, 10.0]),
        )
        graph = build_local_edge_gcn_graph(
            self.previous, self.distances, archive,
            quality_mode="inverse_cost",
        )
        self.assertTrue(torch.allclose(
            graph["solution_weights"], torch.tensor([0.2, 0.1]),
        ))

    def test_local_gcn_receives_supervision_and_changes_h1(self):
        model = Net(local_gcn_hidden=8)
        optimizer = torch.optim.Adam(model.local_edge_gcn.parameters(), lr=0.01)
        before_conv = model.local_edge_gcn.convs[0].lin.weight.detach().clone()
        before, _, _, graph = model.refine_heatmap(
            self.previous, self.distances, self.archive, return_components=True,
        )
        target = graph["quality_target"]
        self.assertGreater(float(rowwise_heatmap_kl(before, target)), 0)
        for _ in range(3):
            optimizer.zero_grad()
            refined, _, _, graph = model.refine_heatmap(
                self.previous, self.distances, self.archive,
                return_components=True,
            )
            rowwise_heatmap_kl(refined, graph["quality_target"]).backward()
            self.assertGreater(
                float(model.local_edge_gcn.output.weight.grad.abs().sum()), 0,
            )
            optimizer.step()
        after = model.refine_heatmap(
            self.previous, self.distances, self.archive,
        )
        self.assertFalse(torch.allclose(before, after))
        self.assertFalse(torch.equal(
            before_conv, model.local_edge_gcn.convs[0].lin.weight.detach(),
        ))
        self.assertLess(
            float(rowwise_heatmap_kl(after, target)),
            float(rowwise_heatmap_kl(before, target)),
        )

    def test_h0_values_are_not_blended_when_selected_graph_is_unchanged(self):
        model = Net(local_gcn_hidden=8)
        changed = self.previous.clone()
        changed[0, 1] = 50.0
        changed[2, 4] = 0.01
        first = model.refine_heatmap(
            self.previous, self.distances, self.archive,
        )
        second = model.refine_heatmap(
            changed, self.distances, self.archive,
        )
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

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
