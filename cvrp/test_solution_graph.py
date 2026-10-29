import unittest

import torch

from graph_net import LocalEdgeGCN
from solution_graph import SolutionArchive, build_local_edge_graph, rowwise_heatmap_kl


class CVRPSolutionGraphTests(unittest.TestCase):
    def setUp(self):
        self.demands = torch.tensor([0., 2., 2., 2.])
        coordinates = torch.tensor([
            [0.5, 0.5], [0.1, 0.2], [0.8, 0.2], [0.2, 0.8],
        ])
        self.distances = torch.cdist(coordinates, coordinates).clamp_min(1e-10)
        # Ant 0 finishes first; its final zero is repeated only as padding.
        self.paths = torch.tensor([
            [0, 0], [1, 1], [2, 0], [0, 2], [3, 0], [0, 3], [0, 0],
        ])

    def test_depot_padding_is_not_a_route_edge(self):
        archive = SolutionArchive(self.demands, capacity=4)
        archive.add(self.paths, torch.tensor([3.0, 4.0]))
        self.assertEqual(archive.lengths.tolist(), [6, 7])
        self.assertEqual(archive.paths[0, 6].item(), 0)
        graph = build_local_edge_graph(
            torch.ones((4, 4)), self.distances, self.demands, archive,
            incident_topk=3,
        )
        edges = list(zip(graph["edge_u"].tolist(), graph["edge_v"].tolist()))
        self.assertNotIn((0, 0), edges)
        self.assertAlmostEqual(
            graph["edge_features"][edges.index((0, 1)), 1].item(), 1.0,
        )
        self.assertAlmostEqual(
            graph["edge_features"][edges.index((1, 2)), 1].item(), 0.5,
        )

    def test_compatible_unseen_fragment_still_has_a_relation(self):
        archive = SolutionArchive(self.demands, capacity=4)
        archive.add(self.paths, torch.tensor([3.0, 4.0]))
        graph = build_local_edge_graph(
            torch.ones((4, 4)), self.distances, self.demands, archive,
            incident_topk=3,
        )
        edges = list(zip(graph["edge_u"].tolist(), graph["edge_v"].tolist()))
        relations = list(zip(*graph["edge_index"].tolist()))
        seen = relations.index((edges.index((0, 1)), edges.index((1, 2))))
        unseen = relations.index((edges.index((2, 1)), edges.index((1, 3))))
        self.assertAlmostEqual(graph["relation_features"][seen, 0].item(), 0.5)
        self.assertEqual(graph["relation_features"][unseen, 0].item(), 0.0)
        self.assertAlmostEqual(graph["edge_weight"][unseen].item(), 0.05)

    def test_capacity_and_history_limits(self):
        archive = SolutionArchive(self.demands, capacity=4, max_rounds=1)
        archive.add(self.paths, torch.tensor([3.0, 4.0]))
        archive.add(self.paths, torch.tensor([3.1, 4.1]))
        self.assertEqual(len(archive), 2)
        self.assertTrue((archive.rounds == 1).all())
        invalid = SolutionArchive(torch.tensor([0., 3., 3.]), capacity=5)
        with self.assertRaises(ValueError):
            invalid.add(torch.tensor([[0], [1], [2], [0]]), torch.tensor([2.0]))

    def test_graph_supervision_reaches_gcn(self):
        archive = SolutionArchive(self.demands, capacity=4)
        archive.add(self.paths, torch.tensor([3.0, 4.0]))
        graph = build_local_edge_graph(
            torch.ones((4, 4)), self.distances, self.demands, archive,
            incident_topk=3,
        )
        model = LocalEdgeGCN(hidden=8, layers=2)
        h1 = model(graph)
        rowwise_heatmap_kl(h1, graph["quality_target"]).backward()
        self.assertIsNotNone(model.output.weight.grad)
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
