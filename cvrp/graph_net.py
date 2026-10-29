"""Original CVRP H0 network with a trainable local solution-edge H1."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from net import Net as OriginalCVRPNet
from solution_graph import build_local_edge_graph, normalize_heatmap_rows


class LocalEdgeGCN(nn.Module):
    def __init__(self, hidden=32, layers=2):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.convs = nn.ModuleList(
            GCNConv(5 if index == 0 else hidden, hidden)
            for index in range(layers)
        )
        self.relation_gate = nn.Sequential(
            nn.Linear(7, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.relation_gate[-1].weight)
        nn.init.zeros_(self.relation_gate[-1].bias)
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, graph):
        hidden = graph["edge_features"]
        relation_weight = graph["edge_weight"] * (
            2 * torch.sigmoid(
                self.relation_gate(graph["relation_features"]).squeeze(-1)
            )
        )
        for conv in self.convs:
            hidden = F.silu(conv(hidden, graph["edge_index"], relation_weight))
        scores = graph["edge_prior"] * torch.exp(
            4 * torch.tanh(self.output(hidden).squeeze(-1))
        )
        heatmap = scores.new_zeros((graph["n_nodes"], graph["n_nodes"]))
        heatmap[graph["edge_u"], graph["edge_v"]] = scores
        return normalize_heatmap_rows(heatmap)


class GraphNet(OriginalCVRPNet):
    """Retain pretrained DeepACO H0 parameters and add the H1 graph module."""

    def __init__(self, graph_layers=2):
        super().__init__()
        self.local_edge_gcn = LocalEdgeGCN(layers=graph_layers)

    def refine_heatmap(
        self, h0, distances, demands, archive, pheromone=None,
        max_solutions=None, quality_mode="relative_diversity",
        quality_temperature=0.1, diversity_strength=0.5,
    ):
        graph = build_local_edge_graph(
            h0.detach(), distances.detach(), demands.detach(), archive,
            pheromone=pheromone, max_solutions=max_solutions,
            quality_mode=quality_mode,
            quality_temperature=quality_temperature,
            diversity_strength=diversity_strength,
        )
        return self.local_edge_gcn(graph), graph
