import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn

from solution_graph import (
    build_learnable_solution_graph,
    graph_refined_heatmap,
    normalize_heatmap_rows,
)

# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=1, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
    def reset_parameters(self):
        raise NotImplementedError
    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w

# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device
    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x) # last layer
        return x

# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)
    def forward(self, x):
        return super().forward(x).squeeze(dim = -1)


def _weighted_scatter_mean(values, index, weights, output_size):
    weighted_values = values * weights.unsqueeze(-1)
    output = values.new_zeros((output_size, values.size(-1)))
    output.index_add_(0, index, weighted_values)
    mass = values.new_zeros(output_size)
    mass.index_add_(0, index, weights)
    return output / mass.clamp_min(1e-10).unsqueeze(-1)


class SolutionGraphLayer(nn.Module):
    """One sparse edge-to-solution-to-edge message-passing layer."""

    def __init__(self, units):
        super().__init__()
        self.solution_update = nn.Linear(2 * units, units)
        self.edge_update = nn.Linear(2 * units, units)
        self.solution_norm = nn.LayerNorm(units)
        self.edge_norm = nn.LayerNorm(units)

    def forward(
        self,
        edge_hidden,
        solution_hidden,
        edge_incidence,
        solution_incidence,
        incidence_weights,
    ):
        solution_context = _weighted_scatter_mean(
            edge_hidden.index_select(0, edge_incidence),
            solution_incidence,
            incidence_weights,
            solution_hidden.size(0),
        )
        solution_delta = F.silu(
            self.solution_update(
                torch.cat((solution_hidden, solution_context), dim=-1)
            )
        )
        solution_hidden = self.solution_norm(solution_hidden + solution_delta)

        edge_context = _weighted_scatter_mean(
            solution_hidden.index_select(0, solution_incidence),
            edge_incidence,
            incidence_weights,
            edge_hidden.size(0),
        )
        edge_delta = F.silu(
            self.edge_update(torch.cat((edge_hidden, edge_context), dim=-1))
        )
        edge_hidden = self.edge_norm(edge_hidden + edge_delta)
        return edge_hidden, solution_hidden


class SolutionGraphNet(nn.Module):
    """Predict a bounded heatmap residual from the cumulative solution graph."""

    def __init__(self, hidden=32, layers=2, max_residual=0.25):
        super().__init__()
        self.max_residual = float(max_residual)
        self.edge_input = nn.Linear(4, hidden)
        self.solution_input = nn.Linear(3, hidden)
        self.layers = nn.ModuleList(
            [SolutionGraphLayer(hidden) for _ in range(layers)]
        )
        self.output = nn.Linear(hidden, 1)
        # The first forward pass reproduces the deterministic base heatmap.
        # Training then learns only the bounded correction requested by PHG-ACO.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, graph):
        edge_hidden = F.silu(self.edge_input(graph["edge_features"]))
        solution_hidden = F.silu(
            self.solution_input(graph["solution_features"])
        )
        for layer in self.layers:
            edge_hidden, solution_hidden = layer(
                edge_hidden,
                solution_hidden,
                graph["edge_incidence"],
                graph["solution_incidence"],
                graph["incidence_weights"],
            )

        edge_residual = self.max_residual * torch.tanh(
            self.output(edge_hidden).squeeze(-1)
        )
        n_nodes = graph["n_nodes"]
        residual = edge_residual.new_zeros((n_nodes, n_nodes))
        residual = residual.index_put(
            (graph["edge_u"], graph["edge_v"]),
            edge_residual,
            accumulate=True,
        )
        residual = residual.index_put(
            (graph["edge_v"], graph["edge_u"]),
            edge_residual,
            accumulate=True,
        )
        return residual

class Net(nn.Module):
    def __init__(
        self,
        solution_graph_hidden=32,
        solution_graph_layers=2,
        solution_graph_max_residual=0.25,
    ):
        super().__init__()
        self.emb_net = EmbNet()
        self.par_net_heu = ParNet()
        self.solution_graph_net = SolutionGraphNet(
            hidden=solution_graph_hidden,
            layers=solution_graph_layers,
            max_residual=solution_graph_max_residual,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, pyg):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)
        return heu

    def refine_heatmap(
        self,
        current_heatmap,
        distances,
        archive,
        quality_temperature=0.75,
        age_decay=0.1,
        uniform_mix=0.01,
        elite_ratio=0.25,
        prior_strength=0.05,
        propagation_strength=0.1,
        distance_prior_strength=0.1,
        max_solutions=128,
        return_components=False,
    ):
        """Combine deterministic aggregation with a learned bounded residual."""
        # The search state is deliberately detached.  H0 distillation updates
        # the initial GNN, while future-quality supervision updates this graph
        # network through the residual only.
        current_heatmap = current_heatmap.detach()
        distances = distances.detach()
        base_heatmap = graph_refined_heatmap(
            current_heatmap,
            archive,
            distances=distances,
            temperature=quality_temperature,
            age_decay=age_decay,
            uniform_mix=uniform_mix,
            elite_ratio=elite_ratio,
            prior_strength=prior_strength,
            propagation_strength=propagation_strength,
            distance_prior_strength=distance_prior_strength,
        )
        graph = build_learnable_solution_graph(
            current_heatmap,
            distances,
            archive,
            max_solutions=max_solutions,
            temperature=quality_temperature,
            age_decay=age_decay,
            uniform_mix=uniform_mix,
        )
        residual = self.solution_graph_net(graph)
        refined = normalize_heatmap_rows(base_heatmap.detach() + residual)
        if return_components:
            return refined, base_heatmap.detach(), residual, graph
        return refined

    @staticmethod
    def deterministic_heatmap(
        current_heatmap,
        distances,
        archive,
        quality_temperature=0.75,
        age_decay=0.1,
        uniform_mix=0.01,
        elite_ratio=0.25,
        prior_strength=0.05,
        propagation_strength=0.1,
        distance_prior_strength=0.1,
    ):
        """Build the detached final-archive target used by future KL."""
        return graph_refined_heatmap(
            current_heatmap.detach(),
            archive,
            distances=distances.detach(),
            temperature=quality_temperature,
            age_decay=age_decay,
            uniform_mix=uniform_mix,
            elite_ratio=elite_ratio,
            prior_strength=prior_strength,
            propagation_strength=propagation_strength,
            distance_prior_strength=distance_prior_strength,
        ).detach()
    
    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False
            
    @staticmethod
    def reshape(pyg, vector):
        '''Turn phe/heu vector into matrix with zero padding 
        '''
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix
