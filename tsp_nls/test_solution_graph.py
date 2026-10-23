import unittest
from itertools import islice, permutations
from unittest.mock import patch

import torch

from solution_graph import (
    EPS,
    build_learnable_solution_graph,
    InstanceSearchState,
    SolutionArchive,
    compute_quality_weights,
    graph_refined_heatmap,
    normalize_heatmap_rows,
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

    def test_archive_defaults_to_unlimited_tours_and_rounds(self):
        archive = SolutionArchive(n_nodes=5)
        self.assertIsNone(archive.max_solutions)
        self.assertIsNone(archive.max_rounds)

        path = torch.tensor([[0], [1], [2], [3], [4]])
        for _ in range(11):
            archive.add(path, torch.tensor([5.0]))

        self.assertEqual(len(archive), 11)
        self.assertEqual(archive.num_rounds, 11)
        self.assertEqual(len(archive._round_edge_keys), 11)

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

    def test_archive_skips_full_rebuild_while_below_limits(self):
        archive = SolutionArchive(n_nodes=5, max_solutions=4, max_rounds=3)
        with patch.object(
            archive, "tensors", side_effect=AssertionError("unexpected rebuild")
        ):
            archive.add(
                torch.tensor([[0], [1], [2], [3], [4]]),
                torch.tensor([5.0]),
            )
            archive.add(
                torch.tensor([[0], [1], [3], [2], [4]]),
                torch.tensor([6.0]),
            )

        self.assertEqual(len(archive), 2)
        self.assertEqual(archive.num_rounds, 2)

    def test_archive_enforces_solution_limit(self):
        archive = SolutionArchive(n_nodes=5, max_solutions=2, max_rounds=10)
        archive.add(
            torch.tensor(
                [
                    [0, 0, 0],
                    [1, 1, 2],
                    [2, 3, 4],
                    [3, 2, 1],
                    [4, 4, 3],
                ]
            ),
            torch.tensor([5.0, 6.0, 7.0]),
        )

        _, costs, _ = archive.tensors()
        self.assertEqual(len(archive), 2)
        self.assertEqual(sorted(costs.tolist()), [5.0, 6.0])

    def test_archive_enforces_round_limit_and_rebuilds_statistics(self):
        archive = SolutionArchive(n_nodes=5, max_solutions=10, max_rounds=2)
        populations = [
            [0, 1, 2, 3, 4],
            [0, 1, 3, 2, 4],
            [0, 2, 1, 4, 3],
        ]
        for cost, population in enumerate(populations, start=5):
            archive.add(
                torch.tensor(population).unsqueeze(1),
                torch.tensor([float(cost)]),
            )
            archive.online_edge_statistics()

        _, _, rounds = archive.tensors()
        direct, propagated = archive.online_edge_statistics()
        self.assertEqual(archive.num_rounds, 3)
        self.assertEqual(len(archive._round_edge_keys), 2)
        self.assertEqual(len(archive), 2)
        self.assertEqual(rounds.min().item(), 1)
        self.assertTrue(torch.isfinite(direct).all())
        self.assertTrue(torch.isfinite(propagated).all())

    def test_archive_keeps_rotated_and_reversed_tours_as_sampled(self):
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
        self.assertEqual(second, {"added": 2, "duplicates": 0})
        self.assertEqual(len(archive), 3)
        self.assertEqual(archive.num_rounds, 2)
        tours, costs, rounds = archive.tensors()
        self.assertTrue(torch.equal(tours, torch.tensor([
            [0, 1, 2, 3, 4],
            [2, 3, 4, 0, 1],
            [0, 4, 3, 2, 1],
        ])))
        self.assertEqual(costs.tolist(), [5.0, 5.0, 5.0])
        self.assertEqual(rounds.tolist(), [0, 1, 1])

        # All observations also contribute to the per-round statistics.
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
        )

        # Edge (0, 1) occurs only in the lower-cost tour, whereas edge (0, 2)
        # occurs only in the higher-cost tour.
        self.assertGreater(float(target[0, 1]), float(target[0, 2]))

    def test_quality_target_does_not_mix_previous_heatmap_values(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        previous = torch.tensor(
            [
                [0.0, 5.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 5.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 5.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 5.0],
                [5.0, 1.0, 1.0, 1.0, 0.0],
            ]
        )
        first_target = quality_target_heatmap(
            previous,
            archive,
            propagation_strength=0.0,
            distance_prior_strength=0.0,
        )
        changed_previous = previous.clone()
        changed_previous[0, 1] = 100.0
        changed_previous[2, 4] = 0.01
        second_target = quality_target_heatmap(
            changed_previous,
            archive,
            propagation_strength=0.0,
            distance_prior_strength=0.0,
        )

        self.assertTrue(
            torch.allclose(first_target, second_target, atol=1e-6)
        )

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
            layers=2,
        )

        self.assertFalse(target.requires_grad)
        self.assertTrue(torch.isfinite(target).all())

    def test_h1_graph_result_does_not_mix_fresh_h0_values(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        first_h0 = torch.ones(5, 5) - torch.eye(5)
        second_h0 = first_h0.clone()
        second_h0[0, 1] = 100.0
        second_h0[2, 4] = 0.01

        self.assertTrue(torch.allclose(
            graph_refined_heatmap(first_h0, archive),
            graph_refined_heatmap(second_h0, archive),
            atol=1e-6,
        ))

    def test_statistical_gcn_matches_inverse_cost_mean_and_cooccurrence(self):
        archive = SolutionArchive(n_nodes=5)
        paths = torch.tensor([
            [0, 0], [1, 2], [2, 1], [3, 3], [4, 4],
        ])
        costs = torch.tensor([2.0, 4.0])
        archive.add(paths, costs)
        prior = torch.ones(5, 5) - torch.eye(5)

        actual = graph_refined_heatmap(
            prior, archive, layers=1
        )

        tours = paths.transpose(0, 1)
        observed_edges = sorted({
            tuple(sorted((int(a), int(b))))
            for tour in tours
            for a, b in zip(tour.tolist(), torch.roll(tour, -1).tolist())
        })
        membership = torch.tensor([
            [float(edge in {
                tuple(sorted((int(a), int(b))))
                for a, b in zip(tour.tolist(), torch.roll(tour, -1).tolist())
            }) for tour in tours]
            for edge in observed_edges
        ])
        inverse_costs = costs.reciprocal()
        initial = (membership @ inverse_costs) / membership.sum(dim=1)
        adjacency = (membership * inverse_costs) @ membership.T
        degree = adjacency.sum(dim=1)
        aggregated = (
            adjacency @ (initial / degree.sqrt())
        ) / degree.sqrt()
        hidden = 0.5 * initial + 0.5 * aggregated
        expected = torch.zeros(5, 5)
        for edge, value in zip(observed_edges, hidden):
            expected[edge[0], edge[1]] = value
            expected[edge[1], edge[0]] = value
        expected = expected / expected.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_local_edge_graph_matches_explicit_consecutive_adjacency(self):
        archive = SolutionArchive(n_nodes=5)
        paths = torch.tensor([
            [0, 0], [1, 2], [2, 1], [3, 3], [4, 4],
        ])
        costs = torch.tensor([2.0, 4.0])
        archive.add(paths, costs)
        prior = torch.ones(5, 5) - torch.eye(5)
        actual = graph_refined_heatmap(
            prior, archive, layers=1,
            structure="local",
        )

        tours = paths.transpose(0, 1)
        tour_edges = [
            [tuple(sorted((int(a), int(b)))) for a, b in zip(
                tour.tolist(), torch.roll(tour, -1).tolist(),
            )]
            for tour in tours
        ]
        observed_edges = sorted(set().union(*(set(edges) for edges in tour_edges)))
        indices = {edge: index for index, edge in enumerate(observed_edges)}
        initial = torch.tensor([
            sum(float(cost.reciprocal()) for edges, cost in zip(tour_edges, costs)
                if edge in edges)
            / sum(edge in edges for edges in tour_edges)
            for edge in observed_edges
        ])
        adjacency = torch.zeros(len(observed_edges), len(observed_edges))
        for edges, cost in zip(tour_edges, costs):
            weight = float(cost.reciprocal())
            for position, edge in enumerate(edges):
                source = indices[edge]
                neighbor = indices[edges[(position + 1) % len(edges)]]
                adjacency[source, source] += weight
                adjacency[source, neighbor] += weight
                adjacency[neighbor, source] += weight
        degree = adjacency.sum(dim=1)
        hidden = 0.5 * initial + 0.5 * (
            adjacency @ (initial / degree.sqrt())
        ) / degree.sqrt()
        expected = torch.zeros(5, 5)
        for edge, value in zip(observed_edges, hidden):
            expected[edge[0], edge[1]] = value
            expected[edge[1], edge[0]] = value
        expected = normalize_heatmap_rows(expected)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertFalse(torch.allclose(
            actual,
            graph_refined_heatmap(prior, archive, layers=1),
        ))

    def test_relative_quality_graph_favors_better_route_and_keeps_duplicates(self):
        paths = torch.tensor([
            [0, 0], [1, 2], [2, 1], [3, 3], [4, 4],
        ])
        costs = torch.tensor([2.0, 4.0])
        archive = SolutionArchive(n_nodes=5)
        archive.add(paths, costs)
        repeated = SolutionArchive(n_nodes=5)
        repeated.add(paths, costs)
        repeated.add(paths[:, :1], costs[:1])
        prior = torch.ones(5, 5) - torch.eye(5)
        kwargs = dict(
            layers=1, structure="local",
            quality_mode="relative", quality_temperature=0.1,
        )
        target = graph_refined_heatmap(prior, archive, **kwargs)
        repeated_target = graph_refined_heatmap(prior, repeated, **kwargs)

        self.assertGreater(float(target[0, 1]), float(target[0, 2]))
        self.assertFalse(torch.allclose(target, repeated_target, atol=1e-6))
        self.assertTrue(torch.isfinite(target).all())

    def test_candidate_only_graph_restricts_nodes_and_directed_output(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([
                [0, 0], [1, 2], [2, 1], [3, 3], [4, 4],
            ]),
            torch.tensor([2.0, 4.0]),
        )
        prior = torch.full((5, 5), EPS)
        candidate_edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
        for u, v in candidate_edges:
            prior[u, v] = 1.0
            prior[v, u] = 1.0
        support = prior > 10.0 * EPS
        kwargs = dict(
            layers=1, structure="local",
            quality_mode="relative", quality_temperature=0.1,
        )
        restricted = graph_refined_heatmap(
            prior, archive, candidate_only=True, **kwargs,
        )
        projected_unrestricted = normalize_heatmap_rows(
            graph_refined_heatmap(prior, archive, **kwargs).masked_fill(
                ~support, 0.0,
            ),
        )

        self.assertLessEqual(float(restricted[~support].max()), 2.0 * EPS)
        self.assertTrue(torch.isfinite(restricted).all())
        self.assertFalse(torch.allclose(
            restricted, projected_unrestricted, atol=1e-6,
        ))

    def test_candidate_only_graph_never_falls_back_to_h0_values(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        h0 = torch.full((5, 5), EPS)
        for u, v in ((0, 2), (2, 4), (4, 1), (1, 3), (3, 0)):
            h0[u, v] = 1.0
            h0[v, u] = 1.0

        with self.assertRaisesRegex(ValueError, "no observed candidate-edge"):
            graph_refined_heatmap(h0, archive, candidate_only=True)

    def test_statistical_gcn_does_not_decay_old_solutions(self):
        paths = torch.tensor([
            [0, 0], [1, 2], [2, 1], [3, 3], [4, 4],
        ])
        costs = torch.tensor([2.0, 4.0])
        first = SolutionArchive(n_nodes=5)
        first.add(paths, costs)
        split_rounds = SolutionArchive(n_nodes=5)
        split_rounds.add(paths[:, :1], costs[:1])
        split_rounds.add(paths[:, 1:], costs[1:])
        prior = torch.ones(5, 5) - torch.eye(5)

        self.assertTrue(torch.allclose(
            graph_refined_heatmap(prior, first),
            graph_refined_heatmap(prior, split_rounds),
            atol=1e-6,
        ))

    def test_statistical_gcn_counts_repeated_sampled_tours(self):
        paths = torch.tensor([
            [0, 0], [1, 2], [2, 1], [3, 3], [4, 4],
        ])
        costs = torch.tensor([2.0, 4.0])
        first = SolutionArchive(n_nodes=5)
        first.add(paths, costs)
        repeated = SolutionArchive(n_nodes=5)
        repeated.add(paths, costs)
        repeated.add(paths[:, :1], costs[:1])
        prior = torch.ones(5, 5) - torch.eye(5)

        self.assertEqual(len(repeated), 3)
        self.assertFalse(torch.allclose(
            graph_refined_heatmap(prior, first),
            graph_refined_heatmap(prior, repeated),
            atol=1e-6,
        ))

    def test_learnable_graph_uses_bounded_quality_ranked_archive(self):
        archive = SolutionArchive(n_nodes=5)
        archive.add(
            torch.tensor(
                [
                    [0, 0, 0],
                    [1, 2, 3],
                    [2, 4, 1],
                    [3, 1, 4],
                    [4, 3, 2],
                ]
            ),
            torch.tensor([1.0, 3.0, 8.0]),
        )
        previous = torch.ones(5, 5) - torch.eye(5)
        distances = torch.ones(5, 5) + torch.eye(5)
        graph = build_learnable_solution_graph(
            previous,
            distances,
            archive,
            max_solutions=2,
            uniform_mix=0.0,
        )

        self.assertEqual(graph["n_solutions"], 2)
        self.assertEqual(graph["solution_features"].shape, (2, 3))
        self.assertEqual(graph["edge_features"].size(1), 4)
        self.assertEqual(graph["edge_incidence"].numel(), 10)
        self.assertTrue(torch.isfinite(graph["edge_features"]).all())
        self.assertTrue(torch.isfinite(graph["solution_features"]).all())

    def test_learnable_graph_defaults_to_all_archived_tours(self):
        archive = SolutionArchive(n_nodes=8)
        tours = list(
            islice(
                (
                    (0, *tail)
                    for tail in permutations(range(1, 8))
                    if tail[0] < tail[-1]
                ),
                257,
            )
        )
        archive.add(
            torch.tensor(tours).transpose(0, 1),
            torch.linspace(1.0, 2.0, len(tours)),
        )
        previous = torch.ones(8, 8) - torch.eye(8)
        distances = torch.ones(8, 8) + torch.eye(8)
        graph = build_learnable_solution_graph(previous, distances, archive)

        self.assertEqual(len(archive), 257)
        self.assertEqual(graph["n_solutions"], 257)
        self.assertEqual(graph["solution_incidence"].numel(), 257 * 8)

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
