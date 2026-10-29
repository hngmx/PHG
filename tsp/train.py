"""Train the original TSP model with the PHG-ACO H0 -> graph -> H1 idea.

The original notebook's optimizer, fresh-instance sampling, and per-size
hyperparameters remain the defaults. Each training instance samples one ACO
population from H0. Its tours form a local solution-edge graph; a trainable
GCN produces H1, which supervises H0 through a detached KL target.
"""

import argparse
import os
import time

import torch

from aco import ACO
from graph_net import GraphNet
from solution_graph import (
    SolutionArchive,
    refinement_distillation_kl,
    rowwise_heatmap_kl,
)
from utils import gen_pyg_data, load_val_dataset


EPS = 1e-10
ORIGINAL_PROFILES = {
    20: (20, 10),
    100: (20, 20),
    500: (50, 50),
}
ORIGINAL_STEPS = 128
ORIGINAL_EPOCHS = 5
ORIGINAL_LR = 3e-4
ORIGINAL_VALIDATION_ITERATIONS = 5
DEFAULT_OUTPUT = "../pretrained/tsp/phg_current_sampling"


def original_profile(n_nodes):
    return ORIGINAL_PROFILES.get(n_nodes, (20, max(1, n_nodes // 10)))


def reinforce_cost_loss(costs, log_probs):
    """Use the original notebook's mean-cost REINFORCE baseline."""
    advantage = (costs - costs.mean()).detach()
    return (advantage * log_probs.sum(dim=0)).mean()


def train_instance(
    model,
    optimizer,
    coordinates,
    n_ants,
    k_sparse,
    archive=None,
    kl_weight=0.03,
    reinforce_weight=1.0,
    graph_supervision_weight=1.0,
    quality_mode="relative_diversity",
    quality_temperature=0.1,
    diversity_strength=0.5,
):
    """Sample S0 once, build H1 from its tours, and update both networks."""
    model.train()
    data, distances = gen_pyg_data(coordinates, k_sparse=k_sparse)
    h0 = model.reshape(data, model(data)) + EPS
    aco = ACO(
        n_ants=n_ants,
        heuristic=h0,
        distances=distances,
        device=str(coordinates.device),
    )
    costs, feasible_costs, log_probs, paths, feasible_paths = (
        aco.sample_iteration(require_prob=True)
    )

    if archive is None:
        archive = SolutionArchive(
            len(coordinates), storage_device="cpu", path_dtype=torch.int32,
        )
    archive.add(feasible_paths, feasible_costs)
    h1, graph = model.refine_heatmap(
        h0,
        distances,
        archive,
        quality_mode=quality_mode,
        quality_temperature=quality_temperature,
        diversity_strength=diversity_strength,
        pheromone=aco.pheromone,
    )
    reinforce = reinforce_cost_loss(costs, log_probs)
    kl = refinement_distillation_kl(h0, h1)
    graph_loss = rowwise_heatmap_kl(h1, graph["quality_target"])
    loss = (
        reinforce_weight * reinforce
        + kl_weight * kl
        + graph_supervision_weight * graph_loss
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return archive, {
        "reinforce": reinforce.detach().item(),
        "kl": kl.detach().item(),
        "graph": graph_loss.detach().item(),
    }


def train_epoch(
    model,
    optimizer,
    n_nodes,
    n_ants,
    k_sparse,
    steps,
    device,
    fixed_pool=None,
    archives=None,
    **loss_options,
):
    totals = {"reinforce": 0.0, "kl": 0.0, "graph": 0.0}
    order = torch.randperm(steps).tolist() if fixed_pool is not None else range(steps)
    for index in order:
        coordinates = (
            fixed_pool[index]
            if fixed_pool is not None
            else torch.rand((n_nodes, 2), device=device)
        )
        archive, values = train_instance(
            model,
            optimizer,
            coordinates,
            n_ants,
            k_sparse,
            archive=None if archives is None else archives[index],
            **loss_options,
        )
        if archives is not None:
            archives[index] = archive
        for name, value in values.items():
            totals[name] += value
    return {name: value / steps for name, value in totals.items()}


@torch.no_grad()
def validation(model, dataset, n_ants, iterations=ORIGINAL_VALIDATION_ITERATIONS):
    """Keep the original notebook's H0-only validation definitions."""
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    for data, distances in dataset:
        h0 = model.reshape(data, model(data)) + EPS
        aco = ACO(
            n_ants=n_ants,
            heuristic=h0,
            distances=distances,
            device=str(distances.device),
        )
        sampled_costs, _ = aco.sample()
        best_cost = aco.run(iterations)
        totals += torch.tensor(
            [sampled_costs.mean(), sampled_costs.min(), best_cost],
            dtype=torch.float64,
        )
    return (totals / len(dataset)).tolist()


def train(
    n_nodes,
    n_ants,
    k_sparse,
    steps=ORIGINAL_STEPS,
    epochs=ORIGINAL_EPOCHS,
    lr=ORIGINAL_LR,
    seed=1234,
    device=None,
    output=DEFAULT_OUTPUT,
    pretrained=None,
    validation_size=None,
    fixed_pool=False,
    kl_weight=0.03,
    reinforce_weight=1.0,
    graph_supervision_weight=1.0,
    quality_mode="relative_diversity",
    quality_temperature=0.1,
    diversity_strength=0.5,
    graph_layers=2,
):
    if steps < 1 or epochs < 1 or n_ants < 1:
        raise ValueError("steps, epochs, and ants must be positive")
    if not 1 <= k_sparse < n_nodes:
        raise ValueError("k_sparse must be between 1 and n_nodes - 1")
    if min(kl_weight, reinforce_weight, graph_supervision_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    if max(kl_weight, reinforce_weight, graph_supervision_weight) == 0:
        raise ValueError("at least one training loss weight must be positive")
    if quality_temperature <= 0 or not 0 <= diversity_strength <= 1:
        raise ValueError("invalid solution graph weighting parameters")
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = GraphNet(graph_layers=graph_layers).to(device)
    if pretrained:
        incompatible = model.load_state_dict(
            torch.load(pretrained, map_location=device), strict=False,
        )
        unexpected = incompatible.unexpected_keys
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("local_edge_gcn.")
        ]
        if unexpected or missing:
            raise ValueError(
                f"incompatible pretrained TSP model: missing={missing}, "
                f"unexpected={unexpected}"
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    val_data = load_val_dataset(n_nodes, k_sparse, device)
    if validation_size is not None:
        val_data = val_data[:validation_size]
    if not val_data:
        raise ValueError("validation dataset cannot be empty")
    os.makedirs(output, exist_ok=True)
    fixed_coordinates = (
        torch.rand((steps, n_nodes, 2), device=device) if fixed_pool else None
    )
    archives = [None] * steps if fixed_pool else None

    best_score = float("inf")
    best_state = None
    start = time.time()
    for epoch in range(epochs + 1):
        if epoch > 0:
            losses = train_epoch(
                model,
                optimizer,
                n_nodes,
                n_ants,
                k_sparse,
                steps,
                device,
                fixed_pool=fixed_coordinates,
                archives=archives,
                kl_weight=kl_weight,
                reinforce_weight=reinforce_weight,
                graph_supervision_weight=graph_supervision_weight,
                quality_mode=quality_mode,
                quality_temperature=quality_temperature,
                diversity_strength=diversity_strength,
            )
        metrics = validation(model, val_data, n_ants)
        print(f"epoch {epoch}: {metrics}", flush=True)
        if epoch > 0:
            print(f"epoch {epoch} losses: {losses}", flush=True)
            torch.save(model.state_dict(), os.path.join(output, f"tsp{n_nodes}-last.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(output, f"tsp{n_nodes}-init.pt"))
        if metrics[-1] <= best_score:
            best_score = metrics[-1]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    best_path = os.path.join(output, f"tsp{n_nodes}-best.pt")
    temporary_path = f"{best_path}.{os.getpid()}.tmp"
    torch.save(best_state, temporary_path)
    os.replace(temporary_path, best_path)
    print(f"total duration: {time.time() - start:.3f}s", flush=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int)
    parser.add_argument("--ants", type=int, default=None)
    parser.add_argument("--k_sparse", type=int, default=None)
    parser.add_argument("--steps", type=int, default=ORIGINAL_STEPS)
    parser.add_argument("--epochs", type=int, default=ORIGINAL_EPOCHS)
    parser.add_argument("--lr", type=float, default=ORIGINAL_LR)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--validation_size", type=int, default=None)
    parser.add_argument("--fixed_pool", action="store_true")
    parser.add_argument("--kl_weight", type=float, default=0.03)
    parser.add_argument("--reinforce_weight", type=float, default=1.0)
    parser.add_argument("--graph_supervision_weight", type=float, default=1.0)
    parser.add_argument(
        "--graph_quality_mode",
        choices=("inverse_cost", "relative", "relative_diversity"),
        default="relative_diversity",
    )
    parser.add_argument("--graph_quality_temperature", type=float, default=0.1)
    parser.add_argument("--graph_diversity_strength", type=float, default=0.5)
    parser.add_argument("--graph_layers", type=int, default=2)
    options = parser.parse_args()
    default_ants, default_k = original_profile(options.nodes)
    train(
        options.nodes,
        default_ants if options.ants is None else options.ants,
        default_k if options.k_sparse is None else options.k_sparse,
        steps=options.steps,
        epochs=options.epochs,
        lr=options.lr,
        seed=options.seed,
        device=options.device,
        output=options.output,
        pretrained=options.pretrained,
        validation_size=options.validation_size,
        fixed_pool=options.fixed_pool,
        kl_weight=options.kl_weight,
        reinforce_weight=options.reinforce_weight,
        graph_supervision_weight=options.graph_supervision_weight,
        quality_mode=options.graph_quality_mode,
        quality_temperature=options.graph_quality_temperature,
        diversity_strength=options.graph_diversity_strength,
        graph_layers=options.graph_layers,
    )
