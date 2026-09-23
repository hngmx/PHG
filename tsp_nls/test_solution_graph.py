import unittest

import torch

from solution_graph import (
    InstanceSearchState,
    SolutionArchive,
    compute_quality_weights,
    graph_refined_heatmap,
    quality_target_heatmap,
    refinement_distillation_kl,
)


class SolutionGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_archive_keeps_every_round(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
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
        archive.add(
            torch.tensor([[0], [2], [1], [4], [3]]),
            torch.tensor([4.5]),
        )

        graph = archive.build_incidence()
        self.assertEqual(len(archive), 3)
        self.assertEqual(archive.num_rounds, 2)
        self.assertEqual(graph["n_solutions"], 3)
        self.assertEqual(graph["edge_incidence"].numel(), 15)
        self.assertEqual(graph["solution_incidence"].numel(), 15)

    def test_archive_can_store_compact_cpu_paths(self):
        archive = SolutionArchive(
            n_nodes=5,
            storage_device="cpu",
            path_dtype=torch.int32,
        )
        archive.add(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )

        self.assertEqual(archive._paths[0].device.type, "cpu")
        self.assertEqual(archive._paths[0].dtype, torch.int32)
        graph = archive.build_incidence()
        self.assertEqual(graph["edge_u"].dtype, torch.int64)
        self.assertEqual(graph["edge_incidence"].numel(), 5)

    def test_archive_deduplicates_rotated_and_reversed_tours(self):
        archive = SolutionArchive(n_nodes=5)
        first = archive.add(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        second = archive.add(
            torch.tensor(
                [
                    [2, 0],
                    [3, 4],
                    [4, 3],
                    [0, 2],
                    [1, 1],
                ]
            ),
            torch.tensor([5.0, 5.0]),
        )

        self.assertEqual(first, {"added": 1, "duplicates": 0})
        self.assertEqual(second, {"added": 0, "duplicates": 2})
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.num_rounds, 2)
        _, _, rounds = archive.tensors()
        self.assertEqual(rounds.item(), 1)

        # Duplicate observations are not re-stored as full archive paths, but
        # they still count as a new sampling round in the online statistics.
        archive.online_edge_statistics(age_decay=0.05)
        cache = next(iter(archive._online_statistics_cache.values()))
        self.assertEqual(cache["processed_rounds"], 2)

    def test_online_statistics_only_process_new_rounds(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )

        direct_1, propagated_1 = archive.online_edge_statistics(
            temperature=0.75,
            age_decay=0.05,
            uniform_mix=0.05,
        )
        cache = next(iter(archive._online_statistics_cache.values()))
        self.assertEqual(cache["processed_rounds"], 1)

        # Calling again without sampling is a cache hit, not a full rebuild.
        direct_2, propagated_2 = archive.online_edge_statistics(
            temperature=0.75,
            age_decay=0.05,
            uniform_mix=0.05,
        )
        self.assertEqual(cache["processed_rounds"], 1)
        self.assertTrue(torch.equal(direct_1, direct_2))
        self.assertTrue(torch.equal(propagated_1, propagated_2))
        direct_before_new_round = direct_2.clone()

        archive.add(
            torch.tensor([[0], [2], [4], [1], [3]]),
            torch.tensor([4.5]),
        )
        direct_3, _ = archive.online_edge_statistics(
            temperature=0.75,
            age_decay=0.05,
            uniform_mix=0.05,
        )
        self.assertEqual(cache["processed_rounds"], 2)
        self.assertFalse(torch.equal(direct_before_new_round, direct_3))

    def test_archive_rejects_infeasible_or_mismatched_tours(self):
        archive = SolutionArchive(n_nodes=5)
        with self.assertRaises(ValueError):
            archive.add(
                torch.tensor([[0], [1], [1], [3], [4]]),
                torch.tensor([4.0]),
            )

    def test_instance_state_retains_graph_across_inner_rounds(self):
        state = InstanceSearchState(torch.ones(5, 5))
        state.add_feasible_solutions(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        state.advance(torch.full((5, 5), 2.0))
        state.add_feasible_solutions(
            torch.tensor([[0], [2], [4], [1], [3]]),
            torch.tensor([4.5]),
        )

        self.assertEqual(state.round_index, 1)
        self.assertEqual(state.archive.num_rounds, 2)
        self.assertEqual(len(state.archive), 2)
        self.assertTrue(torch.equal(state.current_heatmap, torch.full((5, 5), 2.0)))

    def test_quality_weights_prefer_better_and_recent_solutions(self):
        weights = compute_quality_weights(
            costs=torch.tensor([1.0, 3.0, 2.0, 2.0]),
            rounds=torch.tensor([1, 1, 0, 1]),
            num_rounds=2,
            temperature=0.5,
            age_decay=0.5,
            uniform_mix=0.05,
        )

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[3]), float(weights[2]))

    def test_quality_target_favors_edges_from_the_better_tour(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor(
                [
                    [0, 0],
                    [1, 2],
                    [2, 1],
                    [3, 4],
                    [4, 3],
                ]
            ),
            torch.tensor([1.0, 4.0]),
        )
        previous = torch.ones(5, 5) - torch.eye(5)
        target = quality_target_heatmap(
            previous,
            archive,
            temperature=0.5,
            uniform_mix=0.0,
            prior_strength=0.0,
        )

        # Edge (0, 1) occurs only in the lower-cost tour, whereas edge (0, 2)
        # occurs only in the higher-cost tour.
        self.assertGreater(float(target[0, 1]), float(target[0, 2]))
        self.assertTrue(torch.isfinite(target).all())

    def test_elite_filter_excludes_non_elite_tour_edges(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor(
                [
                    [0, 0],
                    [1, 2],
                    [2, 4],
                    [3, 1],
                    [4, 3],
                ]
            ),
            torch.tensor([1.0, 10.0]),
        )

        direct, _ = archive.online_edge_statistics(
            uniform_mix=0.0,
            elite_ratio=0.5,
        )
        # (0, 1) belongs to the elite tour; (0, 2) only to the rejected tour.
        self.assertGreater(float(direct[0, 1]), 0.0)
        self.assertEqual(float(direct[0, 2]), 0.0)

    def test_quality_target_is_normalized_and_uses_distance_prior(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        previous = torch.ones(5, 5) - torch.eye(5)
        distances = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 0.0, 2.0, 3.0, 4.0],
                [2.0, 2.0, 0.0, 3.0, 4.0],
                [3.0, 3.0, 3.0, 0.0, 4.0],
                [4.0, 4.0, 4.0, 4.0, 0.0],
            ]
        )
        target = quality_target_heatmap(
            previous,
            archive,
            distances=distances,
            prior_strength=0.0,
            propagation_strength=0.0,
            distance_prior_strength=1.0,
        )

        self.assertTrue(
            torch.allclose(target.sum(dim=-1), torch.ones(5), atol=1e-6)
        )
        self.assertGreater(float(target[0, 1]), float(target[0, 4]))

    def test_graph_refined_heatmap_is_a_fixed_pseudo_label(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([[0, 0], [1, 2], [2, 1], [3, 4], [4, 3]]),
            torch.tensor([1.0, 4.0]),
        )
        previous = (torch.ones(5, 5) - torch.eye(5)).requires_grad_()
        target = graph_refined_heatmap(
            previous.detach(),
            archive,
            propagation_strength=0.5,
        )

        self.assertFalse(target.requires_grad)
        self.assertTrue(torch.isfinite(target).all())

    def test_refinement_distillation_only_updates_previous_heatmap(self):
        previous = (torch.rand(5, 5) + 0.1).requires_grad_()
        refined = (torch.rand(5, 5) + 0.1).requires_grad_()
        loss = refinement_distillation_kl(previous, refined)
        loss.backward()

        self.assertIsNotNone(previous.grad)
        self.assertGreater(float(previous.grad.abs().sum()), 0.0)
        self.assertIsNone(refined.grad)


if __name__ == "__main__":
    unittest.main()
