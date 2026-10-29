"""Train CVRP H0 with route-cost REINFORCE and a solution-edge H1 teacher.

Each fixed training instance owns its route archive.  A visit starts a fresh
ACO search from the current H0, appends valid routes to that instance's
archive, and trains H1 on elite-route edge support.  H1 is distilled into H0;
ordinary validation and test search still use H0 alone.
"""

import argparse
import os
import time
from pathlib import Path

import torch

from aco import ACO
from graph_net import GraphNet
from solution_graph import rowwise_heatmap_kl
from training_pool import create_training_pool, iter_pool_batches
from utils import gen_instance, gen_pyg_data


EPS = 1e-10
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "pretrained" / "cvrp" / "phg_current"


def reinforce_cost_loss(costs, log_probs):
    """Minimize directly sampled CVRP route length with a leave-one-out baseline."""
    if log_probs is None:
        raise ValueError("REINFORCE needs sampling log probabilities")
    detached = costs.detach()
    if detached.numel() == 1:
        baseline = torch.zeros_like(detached)
    else:
        baseline = (detached.sum() - detached) / (detached.numel() - 1)
    return ((detached - baseline) * log_probs.sum(dim=0)).mean()


def train_instance(
    model, instance, n_ants, device, kl_weight=0.03,
    reinforce_weight=0.1, graph_supervision_weight=1.0,
    max_archive_solutions=None, max_archive_rounds=None,
    max_solution_graph_solutions=None,
    graph_quality_mode="relative_diversity",
    graph_quality_temperature=0.1, graph_diversity_strength=0.5,
):
    demands, distances = instance.materialize(device)
    pyg = gen_pyg_data(demands, distances, device)
    h0 = model(pyg).reshape_as(distances) + EPS
    aco = ACO(
        distances=distances, demand=demands, n_ants=n_ants,
        heuristic=h0, device=str(device),
    )
    costs, log_probs, paths = aco.sample_iteration(
        require_prob=reinforce_weight > 0,
    )
    loss = h0.sum() * 0
    stats = {"reinforce": 0.0, "graph": 0.0, "kl": 0.0}
    if reinforce_weight > 0:
        reinforce = reinforce_cost_loss(costs, log_probs)
        loss = loss + reinforce_weight * reinforce
        stats["reinforce"] = reinforce.detach().item()
    if kl_weight > 0:
        archive = instance.get_archive(
            capacity=aco.capacity, max_solutions=max_archive_solutions,
            max_rounds=max_archive_rounds,
        )
        archive.add(paths, costs)
        h1, graph = model.refine_heatmap(
            h0, distances, demands, archive, pheromone=aco.pheromone,
            max_solutions=max_solution_graph_solutions,
            quality_mode=graph_quality_mode,
            quality_temperature=graph_quality_temperature,
            diversity_strength=graph_diversity_strength,
        )
        graph_loss = rowwise_heatmap_kl(h1, graph["quality_target"])
        kl_loss = rowwise_heatmap_kl(h0, h1)
        loss = loss + graph_supervision_weight * graph_loss + kl_weight * kl_loss
        stats["graph"] = graph_loss.detach().item()
        stats["kl"] = kl_loss.detach().item()
    return loss, stats


def train_epoch(model, optimizer, pool, steps, batch_size, n_ants, device, **loss_options):
    model.train()
    totals = {"reinforce": 0.0, "graph": 0.0, "kl": 0.0}
    count = 0
    for batch in iter_pool_batches(pool, steps, batch_size):
        optimizer.zero_grad()
        for instance in batch:
            loss, stats = train_instance(
                model, instance, n_ants, device, **loss_options,
            )
            (loss / len(batch)).backward()
            for name, value in stats.items():
                totals[name] += value
            count += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def validate(model, instances, n_ants, device, iterations=10, sampling_seed=12345):
    """Fixed-seed H0-only ACO validation; no training archive or H1 is used."""
    if iterations < 1 or not instances:
        raise ValueError("validation needs instances and positive iterations")
    model.eval()
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    totals = torch.zeros(3, dtype=torch.float64)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(sampling_seed)
        for demands_cpu, distances_cpu in instances:
            demands = demands_cpu.to(device)
            distances = distances_cpu.to(device)
            pyg = gen_pyg_data(demands, distances, device)
            h0 = model(pyg).reshape_as(distances) + EPS
            aco = ACO(
                distances=distances, demand=demands, n_ants=n_ants,
                heuristic=h0, device=str(device),
            )
            sampled_costs, _ = aco.sample()
            best = aco.run(iterations)
            totals += torch.tensor(
                [sampled_costs.mean().item(), sampled_costs.min().item(),
                 float(best)], dtype=torch.float64,
            )
    return (totals / len(instances)).tolist()


def train(
    n_customers, n_ants=None, steps=20, batch_size=20, epochs=20,
    pool_size=400, validation_size=20, iterations=10, lr=3e-4,
    seed=1234, device=None, output=None, pretrained=None,
    kl_weight=0.03, reinforce_weight=0.1, graph_supervision_weight=1.0,
    max_archive_solutions=None, max_archive_rounds=None,
    max_solution_graph_solutions=None,
    graph_quality_mode="relative_diversity",
    graph_quality_temperature=0.1, graph_diversity_strength=0.5,
):
    if min(n_customers, steps, batch_size, epochs, pool_size, validation_size, iterations) < 1:
        raise ValueError("sizes, steps, epochs, and iterations must be positive")
    if pool_size != steps * batch_size:
        raise ValueError("pool_size must equal steps * batch_size")
    if min(kl_weight, reinforce_weight, graph_supervision_weight) < 0:
        raise ValueError("loss weights cannot be negative")
    if kl_weight == reinforce_weight == 0:
        raise ValueError("at least one H0 loss must be positive")
    if kl_weight > 0 and graph_supervision_weight == 0:
        raise ValueError("the H1 teacher needs graph supervision")
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    n_ants = n_ants or (20 if n_customers == 20 else 50)
    output = Path(output or DEFAULT_OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    model = GraphNet().to(device)
    if pretrained:
        incompatible = model.load_state_dict(
            torch.load(pretrained, map_location=device, weights_only=True),
            strict=False,
        )
        unexpected = incompatible.unexpected_keys
        missing = [key for key in incompatible.missing_keys
                   if not key.startswith("local_edge_gcn.")]
        if unexpected or missing:
            raise ValueError(f"incompatible pretrained model: {unexpected}, {missing}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    pool = create_training_pool(pool_size, n_customers)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 100000)
        validation = [gen_instance(n_customers, "cpu")
                      for _ in range(validation_size)]
    torch.save(model.state_dict(), output / f"cvrp{n_customers}-init.pt")
    best_score = float("inf")
    best_state = None
    start = time.time()
    for epoch in range(epochs + 1):
        if epoch:
            losses = train_epoch(
                model, optimizer, pool, steps, batch_size, n_ants, device,
                kl_weight=kl_weight, reinforce_weight=reinforce_weight,
                graph_supervision_weight=graph_supervision_weight,
                max_archive_solutions=max_archive_solutions,
                max_archive_rounds=max_archive_rounds,
                max_solution_graph_solutions=max_solution_graph_solutions,
                graph_quality_mode=graph_quality_mode,
                graph_quality_temperature=graph_quality_temperature,
                graph_diversity_strength=graph_diversity_strength,
            )
            scheduler.step()
            torch.save(model.state_dict(), output / f"cvrp{n_customers}-last.pt")
        metrics = validate(
            model, validation, n_ants, device, iterations=iterations,
            sampling_seed=seed + 200000,
        )
        print(f"epoch {epoch}: sample_mean={metrics[0]:.6f} "
              f"sample_best={metrics[1]:.6f} aco_best={metrics[2]:.6f}",
              flush=True)
        if epoch:
            print(f"epoch {epoch} losses: {losses}", flush=True)
        if metrics[-1] < best_score:
            best_score = metrics[-1]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    best_path = output / f"cvrp{n_customers}-best.pt"
    temporary = best_path.with_name(best_path.name + f".{os.getpid()}.tmp")
    torch.save(best_state, temporary)
    os.replace(temporary, best_path)
    print(f"training duration: {time.time() - start:.1f}s", flush=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nodes", type=int, help="number of customers, excluding depot")
    parser.add_argument("--ants", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--train_pool_size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation_size", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--kl_weight", type=float, default=0.03)
    parser.add_argument("--reinforce_weight", type=float, default=0.1)
    parser.add_argument("--graph_supervision_weight", type=float, default=1.0)
    parser.add_argument("--max_archive_solutions", type=int, default=None)
    parser.add_argument("--max_archive_rounds", type=int, default=None)
    parser.add_argument("--max_solution_graph_solutions", type=int, default=None)
    parser.add_argument("--graph_quality_mode",
                        choices=("inverse_cost", "relative", "relative_diversity"),
                        default="relative_diversity")
    parser.add_argument("--graph_quality_temperature", type=float, default=0.1)
    parser.add_argument("--graph_diversity_strength", type=float, default=0.5)
    args = parser.parse_args()
    train(
        args.nodes, n_ants=args.ants, steps=args.steps,
        batch_size=args.batch_size, epochs=args.epochs,
        pool_size=args.train_pool_size, validation_size=args.validation_size,
        iterations=args.iterations, lr=args.lr, seed=args.seed,
        device=args.device, output=args.output, pretrained=args.pretrained,
        kl_weight=args.kl_weight, reinforce_weight=args.reinforce_weight,
        graph_supervision_weight=args.graph_supervision_weight,
        max_archive_solutions=args.max_archive_solutions,
        max_archive_rounds=args.max_archive_rounds,
        max_solution_graph_solutions=args.max_solution_graph_solutions,
        graph_quality_mode=args.graph_quality_mode,
        graph_quality_temperature=args.graph_quality_temperature,
        graph_diversity_strength=args.graph_diversity_strength,
    )
