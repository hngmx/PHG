import argparse
from contextlib import contextmanager
import os
import random
import time

import numpy as np
import torch

from net import Net, REFINEMENT_MODES
from solution_graph import refinement_distillation_kl, rowwise_heatmap_kl
from solution_sampler import ACOSolutionSampler
from test import infer_instance as infer_h0_test_instance
from training_pool import (
    create_training_pool,
    iter_pool_batches,
    refresh_training_pool,
)
from utils import default_k_sparse, load_val_dataset


EPS = 1e-10
DEFAULT_GRAPH_ROUNDS = 1
DEFAULT_KL_WEIGHT = 0.03
DEFAULT_REINFORCE_WEIGHT = 0.1
H0_VALIDATION_ITERATIONS = 10
PRETRAINED_DIR = "../pretrained/tsp_nls"
TSP100_FINETUNE_PROFILE = "tsp100_finetune"


def validate_graph_rounds(graph_rounds):
    """Allow at most one H0-to-H1 update, when the KL objective is enabled."""
    if graph_rounds != DEFAULT_GRAPH_ROUNDS:
        raise ValueError("training supports exactly one H0 -> H1 refinement")


def resolve_loss_weights(n_node, kl_weight=None, reinforce_weight=None):
    """Use the same KL and REINFORCE defaults at every problem size."""
    return (
        DEFAULT_KL_WEIGHT if kl_weight is None else kl_weight,
        DEFAULT_REINFORCE_WEIGHT if reinforce_weight is None else reinforce_weight,
    )


def pool_refresh_fraction_for_mode(mode, mixed_fraction=0.5):
    """Resolve fixed, fully refreshed, and mixed training-pool policies."""
    if not 0 <= mixed_fraction <= 1:
        raise ValueError("mixed pool refresh fraction must be between 0 and 1")
    if mode == "fixed":
        return 0.0
    if mode == "refresh":
        return 1.0
    if mode == "mixed":
        return mixed_fraction
    raise ValueError(f"unknown training pool mode: {mode}")


def resolve_training_profile(
    nodes,
    profile="standard",
    *,
    lr=None,
    epochs=None,
    k_sparse=None,
    train_pool_size=None,
    pretrained=None,
    output=None,
):
    """Resolve reproducible CLI defaults without hiding user overrides."""
    if profile not in {"standard", TSP100_FINETUNE_PROFILE}:
        raise ValueError(f"unknown training profile: {profile}")

    if profile == TSP100_FINETUNE_PROFILE:
        if nodes != 100:
            raise ValueError(
                f"{TSP100_FINETUNE_PROFILE} is restricted to TSP100"
            )
        defaults = {
            "lr": 1e-4,
            "epochs": 3,
            "k_sparse": default_k_sparse(nodes),
            "train_pool_size": 400,
            "pretrained": os.path.join(PRETRAINED_DIR, "tsp100-best.pt"),
            "output": os.path.join(
                PRETRAINED_DIR, "optimized_v3_full_finetune"
            ),
        }
    else:
        defaults = {
            "lr": 3e-4,
            "epochs": 20,
            "k_sparse": default_k_sparse(nodes),
            "train_pool_size": 400,
            "pretrained": None,
            "output": PRETRAINED_DIR,
        }

    supplied = {
        "lr": lr,
        "epochs": epochs,
        "k_sparse": k_sparse,
        "train_pool_size": train_pool_size,
        "pretrained": pretrained,
        "output": output,
    }
    return {
        name: defaults[name] if value is None else value
        for name, value in supplied.items()
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_random_seed(seed):
    """Run deterministic validation without changing the training RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def validate_loss_weights(kl_weight, reinforce_weight):
    if kl_weight < 0 or reinforce_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if kl_weight == 0 and reinforce_weight == 0:
        raise ValueError("at least one training loss weight must be positive")


def reinforce_cost_loss(costs, log_probs):
    """Minimize expected sampled-tour length with a leave-one-out baseline."""
    if log_probs is None:
        raise ValueError("REINFORCE requires differentiable log probabilities")
    if costs.ndim != 1 or log_probs.ndim != 2:
        raise ValueError("expected costs [ants] and log probabilities [steps, ants]")
    if log_probs.size(1) != costs.numel():
        raise ValueError("cost and log-probability ant counts must match")
    costs = costs.detach()
    if costs.numel() == 1:
        baseline = torch.zeros_like(costs)
    else:
        baseline = (costs.sum() - costs) / (costs.numel() - 1)
    advantage = (costs - baseline).detach()
    return (advantage * log_probs.sum(dim=0)).mean()


def train_instance(
    model,
    optimizer,
    data,
    n_ants,
    k_sparse,
    graph_rounds=DEFAULT_GRAPH_ROUNDS,
    kl_weight=DEFAULT_KL_WEIGHT,
    reinforce_weight=DEFAULT_REINFORCE_WEIGHT,
    max_solution_graph_solutions=None,
    max_archive_solutions=None,
    max_archive_rounds=None,
    refinement_mode="local_gcn",
    local_search=None,
    sampler_factory=ACOSolutionSampler,
    graph_structure="whole_tour",
    graph_layers=2,
    graph_quality_mode=None,
    graph_quality_temperature=0.1,
    graph_diversity_strength=0.5,
    graph_candidate_only=False,
    graph_supervision_weight=1.0,
    loss_stats=None,
):
    """Train H0 from one S0 population with KL, REINFORCE, or both."""
    validate_graph_rounds(graph_rounds)
    validate_loss_weights(kl_weight, reinforce_weight)
    if refinement_mode == "local_gcn" and kl_weight > 0 and graph_supervision_weight <= 0:
        raise ValueError("local GCN requires a positive graph supervision weight")
    if not data:
        raise ValueError("training batch must contain at least one instance")
    model.train()
    optimizer.zero_grad()

    for training_instance in data:
        pyg_data, distances = training_instance.materialize(k_sparse, device)
        heu_vec = model(pyg_data)
        initial_heatmap = model.reshape(pyg_data, heu_vec) + EPS
        state = None
        if kl_weight > 0:
            state = training_instance.start_visit(
                initial_heatmap,
                archive_max_solutions=max_archive_solutions,
                archive_max_rounds=max_archive_rounds,
            )

        sampler = sampler_factory(
            n_solutions=n_ants,
            # The persistent state stores a detached H0. REINFORCE must use
            # this visit's live H0 so sampled log probabilities reach the GNN.
            heatmap=(
                initial_heatmap if reinforce_weight > 0
                else state.current_heatmap.to(device)
            ),
            distances=distances,
            device=device,
            local_search=local_search,
        )

        solutions = sampler.sample(
            inference=False, require_log_probs=reinforce_weight > 0,
        )
        instance_loss = 0.0
        if reinforce_weight > 0:
            reinforce_loss = reinforce_cost_loss(
                solutions.sampled_costs, solutions.log_probs,
            )
            instance_loss = reinforce_weight * reinforce_loss
            if loss_stats is not None:
                loss_stats["reinforce_sum"] += reinforce_loss.detach()
        if kl_weight > 0:
            state.add_feasible_solutions(
                solutions.feasible_paths,
                solutions.feasible_costs,
            )
            if refinement_mode == "local_gcn":
                refined_heatmap, _, _, graph = model.refine_heatmap(
                    state.initial_heatmap.to(model.device),
                    distances,
                    state.archive,
                    max_solutions=max_solution_graph_solutions,
                    refinement_mode=refinement_mode,
                    return_components=True,
                    graph_structure=graph_structure,
                    graph_layers=graph_layers,
                    graph_quality_mode=graph_quality_mode,
                    graph_quality_temperature=graph_quality_temperature,
                    graph_diversity_strength=graph_diversity_strength,
                    graph_candidate_only=graph_candidate_only,
                    graph_pheromone=getattr(sampler, "pheromone", None),
                )
                graph_target = graph.get("quality_target") if isinstance(graph, dict) else None
                if graph_target is None:
                    raise RuntimeError("local GCN must return a graph quality target")
                graph_loss = rowwise_heatmap_kl(refined_heatmap, graph_target)
                instance_loss = instance_loss + graph_supervision_weight * graph_loss
                if loss_stats is not None:
                    loss_stats["graph_sum"] += graph_loss.detach()
            else:
                # Legacy teachers are fixed; their optional residual is frozen.
                with torch.no_grad():
                    refined_heatmap, _, _, _ = model.refine_heatmap(
                        state.initial_heatmap.to(model.device),
                        distances,
                        state.archive,
                        max_solutions=max_solution_graph_solutions,
                        refinement_mode=refinement_mode,
                        return_components=True,
                        graph_structure=graph_structure,
                        graph_layers=graph_layers,
                        graph_quality_mode=graph_quality_mode,
                        graph_quality_temperature=graph_quality_temperature,
                        graph_diversity_strength=graph_diversity_strength,
                        graph_candidate_only=graph_candidate_only,
                    )
            kl_loss = refinement_distillation_kl(initial_heatmap, refined_heatmap)
            instance_loss = instance_loss + kl_weight * kl_loss
            if loss_stats is not None:
                loss_stats["kl_sum"] += kl_loss.detach()
        # Backpropagating each instance separately avoids retaining every ant's
        # sampling graph for all 20 instances in a training batch.
        (instance_loss / len(data)).backward()
        training_instance.finish_visit()
        if loss_stats is not None:
            loss_stats["count"] += 1

    torch.nn.utils.clip_grad_norm_(
        parameters=model.parameters(), max_norm=3.0, norm_type=2
    )
    optimizer.step()


def train_epoch(
    n_ants,
    k_sparse,
    steps_per_epoch,
    net,
    optimizer,
    training_pool,
    batch_size=1,
    graph_rounds=DEFAULT_GRAPH_ROUNDS,
    kl_weight=DEFAULT_KL_WEIGHT,
    reinforce_weight=DEFAULT_REINFORCE_WEIGHT,
    max_solution_graph_solutions=None,
    max_archive_solutions=None,
    max_archive_rounds=None,
    refinement_mode="local_gcn",
    local_search=None,
    graph_structure="whole_tour",
    graph_layers=2,
    graph_quality_mode=None,
    graph_quality_temperature=0.1,
    graph_diversity_strength=0.5,
    graph_candidate_only=False,
    graph_supervision_weight=1.0,
    collect_loss_values=False,
):
    loss_stats = (
        {"kl_sum": 0.0, "reinforce_sum": 0.0, "graph_sum": 0.0, "count": 0}
        if collect_loss_values else None
    )
    for training_batch in iter_pool_batches(
        training_pool,
        steps=steps_per_epoch,
        batch_size=batch_size,
    ):
        train_instance(
            net,
            optimizer,
            training_batch,
            n_ants,
            k_sparse,
            graph_rounds=graph_rounds,
            kl_weight=kl_weight,
            reinforce_weight=reinforce_weight,
            max_solution_graph_solutions=max_solution_graph_solutions,
            max_archive_solutions=max_archive_solutions,
            max_archive_rounds=max_archive_rounds,
            refinement_mode=refinement_mode,
            local_search=local_search,
            graph_structure=graph_structure,
            graph_layers=graph_layers,
            graph_quality_mode=graph_quality_mode,
            graph_quality_temperature=graph_quality_temperature,
            graph_diversity_strength=graph_diversity_strength,
            graph_candidate_only=graph_candidate_only,
            graph_supervision_weight=graph_supervision_weight,
            loss_stats=loss_stats,
        )
    if loss_stats is not None:
        if loss_stats["count"] == 0:
            raise ValueError("cannot report losses for an empty training epoch")
        return {
            "kl": float(loss_stats["kl_sum"] / loss_stats["count"]),
            "reinforce": float(
                loss_stats["reinforce_sum"] / loss_stats["count"]
            ),
            "graph": float(loss_stats["graph_sum"] / loss_stats["count"]),
        }
    return None


@torch.no_grad()
def validation_h0_only(
    n_ants,
    net,
    val_dataset,
    n_iterations=H0_VALIDATION_ITERATIONS,
    local_search=None,
    validation_seed=12345,
    return_training_metrics=False,
):
    """Evaluate H0 with the same fixed-heatmap ACO path as test.py."""
    if not val_dataset:
        raise ValueError("H0-only validation requires at least one instance")
    curves = []
    training_metrics = []
    with isolated_random_seed(validation_seed):
        for data, distances in val_dataset:
            result = infer_h0_test_instance(
                net,
                data,
                distances,
                n_ants,
                n_iterations=n_iterations,
                local_search=local_search,
                sampling_backend="numba",
                **({"return_training_metrics": True} if return_training_metrics else {}),
            )
            if return_training_metrics:
                curve, metrics = result
                curves.append(curve)
                training_metrics.append(metrics)
            else:
                curves.append(result)
    mean_curve = np.asarray(curves, dtype=np.float64).mean(axis=0).tolist()
    if return_training_metrics:
        return mean_curve, np.asarray(training_metrics, dtype=np.float64).mean(axis=0).tolist()
    return mean_curve


def format_training_epoch(epoch, training_metrics):
    """Print DeepACO's four metrics: initial mean/min, then ACO +1/+5 best."""
    if len(training_metrics) != 4:
        raise ValueError("training output requires four DeepACO metrics")
    return f"epoch {epoch}: {training_metrics}"


def format_training_losses(
    epoch, loss_values, kl_weight, reinforce_weight, graph_supervision_weight=1.0,
):
    """Show unweighted epoch means and their weighted loss contributions."""
    kl_loss = loss_values["kl"]
    reinforce_loss = loss_values["reinforce"]
    message = (
        f"epoch {epoch} losses: "
        f"KL={kl_loss:.6g} (weighted={kl_weight * kl_loss:.6g}), "
        f"REINFORCE={reinforce_loss:.6g} "
        f"(weighted={reinforce_weight * reinforce_loss:.6g})"
    )
    if "graph" in loss_values:
        graph_loss = loss_values["graph"]
        message += (
            f", GCN={graph_loss:.6g} "
            f"(weighted={graph_supervision_weight * graph_loss:.6g})"
        )
    return message


def train(
    n_node,
    n_ants,
    steps_per_epoch,
    epochs,
    k_sparse=None,
    batch_size=20,
    test_size=None,
    pretrained=None,
    savepath="../pretrained/tsp_nls",
    graph_rounds=DEFAULT_GRAPH_ROUNDS,
    kl_weight=None,
    reinforce_weight=None,
    train_pool_size=400,
    max_solution_graph_solutions=None,
    max_archive_solutions=None,
    max_archive_rounds=None,
    refinement_mode="local_gcn",
    local_search=None,
    train_pool_mode="fixed",
    pool_refresh_fraction=0.5,
    seed=1234,
    graph_structure="whole_tour",
    graph_layers=2,
    graph_quality_mode=None,
    graph_quality_temperature=0.1,
    graph_diversity_strength=0.5,
    graph_candidate_only=False,
    graph_supervision_weight=1.0,
    show_loss_values=False,
    probe_instances=0,
    probe_repeats=3,
    probe_seed=None,
    probe_output=None,
):
    seed_everything(seed)
    kl_weight, reinforce_weight = resolve_loss_weights(
        n_node, kl_weight, reinforce_weight,
    )
    validate_graph_rounds(graph_rounds)
    validate_loss_weights(kl_weight, reinforce_weight)
    if refinement_mode == "h0":
        raise ValueError("h0 refinement mode is evaluation-only")
    if refinement_mode not in REFINEMENT_MODES:
        raise ValueError(f"unknown refinement mode: {refinement_mode}")
    if graph_structure not in {"whole_tour", "local"}:
        raise ValueError("unknown graph structure")
    if graph_layers < 1 or graph_quality_temperature <= 0:
        raise ValueError("graph layers and quality temperature must be positive")
    if graph_quality_mode not in {None, "inverse_cost", "relative", "relative_diversity"}:
        raise ValueError("unknown graph quality mode")
    if graph_quality_mode == "relative_diversity" and refinement_mode != "local_gcn":
        raise ValueError("relative_diversity requires local_gcn refinement")
    if not 0 <= graph_diversity_strength <= 1:
        raise ValueError("graph diversity strength must be between 0 and 1")
    if refinement_mode == "local_gcn" and kl_weight > 0 and graph_supervision_weight <= 0:
        raise ValueError("local GCN requires a positive graph supervision weight")
    if probe_instances < 0 or probe_repeats < 1:
        raise ValueError("probe instances must be non-negative and repeats positive")
    if probe_instances and refinement_mode != "local_gcn":
        raise ValueError("paired heatmap probe requires local_gcn refinement")
    if (
        max_solution_graph_solutions is not None
        and max_archive_solutions is not None
        and max_solution_graph_solutions > max_archive_solutions
    ):
        raise ValueError(
            "learned graph solution limit cannot exceed archive capacity"
        )
    k_sparse = default_k_sparse(n_node) if k_sparse is None else k_sparse
    required_pool_size = steps_per_epoch * batch_size
    if train_pool_size is None:
        train_pool_size = required_pool_size
    if train_pool_size != required_pool_size:
        raise ValueError(
            "training pool size must equal steps_per_epoch * batch_size "
            f"({train_pool_size} != {steps_per_epoch} * {batch_size})"
        )

    os.makedirs(savepath, exist_ok=True)
    net = Net(local_gcn_layers=graph_layers).to(device)
    if pretrained:
        incompatible = net.load_state_dict(
            torch.load(pretrained, map_location=device), strict=False
        )
        if incompatible.missing_keys:
            print("initialized new parameters:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print(
                "ignored obsolete trainable graph-updater parameters:",
                incompatible.unexpected_keys,
            )

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    training_pool = create_training_pool(train_pool_size, n_node)
    resolved_refresh_fraction = pool_refresh_fraction_for_mode(
        train_pool_mode,
        mixed_fraction=pool_refresh_fraction,
    )
    val_list = load_val_dataset(n_node, k_sparse, device, start_node=0)
    if test_size is not None:
        val_list = val_list[:test_size]

    h0_stats, training_metrics = validation_h0_only(
        n_ants,
        net,
        val_list,
        local_search=local_search,
        validation_seed=seed + 100000,
        return_training_metrics=True,
    )
    best_result = h0_stats[-1]
    best_path = os.path.join(savepath, f"tsp{n_node}-best.pt")
    last_path = os.path.join(savepath, f"tsp{n_node}-last.pt")
    init_path = os.path.join(savepath, f"tsp{n_node}-init.pt")
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in net.state_dict().items()
    }
    # Save initialization under an explicit name.  The public best checkpoint
    # is replaced atomically only after the whole run finishes, so an aborted
    # run cannot overwrite a previously trained best model with epoch 0.
    torch.save(best_state, init_path)
    print(format_training_epoch(0, training_metrics))

    probe = None
    if probe_instances:
        from heatmap_probe import HeatmapProbe

        probe = HeatmapProbe(
            nodes=n_node, count=probe_instances, repeats=probe_repeats,
            seed=seed + 200000 if probe_seed is None else probe_seed,
            output=probe_output or os.path.join(savepath, "heatmap_probe"),
            ants=n_ants, k_sparse=k_sparse, local_search=local_search,
            refinement_mode=refinement_mode,
            max_solution_graph_solutions=max_solution_graph_solutions,
            max_archive_solutions=max_archive_solutions,
            max_archive_rounds=max_archive_rounds,
            graph_structure=graph_structure, graph_layers=graph_layers,
            graph_quality_mode=graph_quality_mode,
            graph_quality_temperature=graph_quality_temperature,
            graph_diversity_strength=graph_diversity_strength,
            graph_candidate_only=graph_candidate_only,
        )

    sum_time = 0
    for epoch in range(1, epochs + 1):
        if epoch > 1:
            refresh_training_pool(
                training_pool,
                resolved_refresh_fraction,
            )
        start = time.time()
        epoch_losses = train_epoch(
            n_ants,
            k_sparse,
            steps_per_epoch,
            net,
            optimizer,
            training_pool,
            batch_size=batch_size,
            graph_rounds=graph_rounds,
            kl_weight=kl_weight,
            reinforce_weight=reinforce_weight,
            max_solution_graph_solutions=max_solution_graph_solutions,
            max_archive_solutions=max_archive_solutions,
            max_archive_rounds=max_archive_rounds,
            refinement_mode=refinement_mode,
            local_search=local_search,
            graph_structure=graph_structure,
            graph_layers=graph_layers,
            graph_quality_mode=graph_quality_mode,
            graph_quality_temperature=graph_quality_temperature,
            graph_diversity_strength=graph_diversity_strength,
            graph_candidate_only=graph_candidate_only,
            graph_supervision_weight=graph_supervision_weight,
            collect_loss_values=show_loss_values,
        )
        sum_time += time.time() - start
        h0_stats, training_metrics = validation_h0_only(
            n_ants,
            net,
            val_list,
            local_search=local_search,
            validation_seed=seed + 100000,
            return_training_metrics=True,
        )
        print(format_training_epoch(epoch, training_metrics))
        if show_loss_values:
            print(format_training_losses(
                epoch, epoch_losses, kl_weight, reinforce_weight,
                graph_supervision_weight,
            ))
        if probe is not None:
            probe.evaluate(epoch, net)
        scheduler.step()
        curr_result = h0_stats[-1]
        if curr_result <= best_result:
            best_result = curr_result
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
        torch.save(net.state_dict(), last_path)

    temporary_best_path = f"{best_path}.{os.getpid()}.tmp"
    torch.save(best_state, temporary_best_path)
    os.replace(temporary_best_path, best_path)
    print("\ntotal training duration:", sum_time)
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar="N", type=int, help="Problem scale")
    parser.add_argument(
        "--profile",
        choices=("standard", TSP100_FINETUNE_PROFILE),
        default="standard",
        help=(
            "Training defaults. tsp100_finetune uses k=10 and a low learning "
            "rate with the 400-instance pool; explicit flags override it"
        ),
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=float,
        default=None,
        help="Learning rate (profile default: 3e-4 or 1e-4 for TSP100 fine-tuning)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="The device used to train neural networks",
    )
    parser.add_argument("-p", "--pretrained", type=str, default=None,
                        help="Path to a pretrained DeepACO model")
    parser.add_argument("-a", "--ants", type=int, default=48,
                        help="Number of ants (matches test.py default)")
    parser.add_argument("-b", "--batch_size", type=int, default=20,
                        help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20,
                        help="Steps per epoch (default: 20 with batch size 20)")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=None,
        help="Epochs to run (profile default: 20 or 3 for TSP100 fine-tuning)",
    )
    parser.add_argument(
        "--k_sparse",
        type=int,
        default=None,
        help="Candidate neighbors per node (default: max(1, nodes // 10))",
    )
    parser.add_argument(
        "--train_pool_size",
        type=int,
        default=None,
        help=(
            "Fixed pool size; must equal steps * batch_size (default: 400)"
        ),
    )
    parser.add_argument(
        "--train_pool_mode",
        choices=("fixed", "refresh", "mixed"),
        default="fixed",
        help="Reuse coordinates, refresh all each epoch, or refresh a fraction",
    )
    parser.add_argument(
        "--pool_refresh_fraction",
        type=float,
        default=0.5,
        help="Fraction replaced per epoch when --train_pool_mode mixed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Training seed; validation uses a fixed isolated derived seed",
    )
    parser.add_argument(
        "-r",
        "--graph_rounds",
        type=int,
        default=DEFAULT_GRAPH_ROUNDS,
        help="Fixed at 1: construct one H1 when the KL loss is enabled",
    )
    parser.add_argument(
        "--kl_weight",
        "--distill_kl_weight",
        dest="kl_weight",
        type=float,
        default=None,
        help="Weight of KL(detached refined heatmap || H0) (default: 0.03 at every size)",
    )
    parser.add_argument(
        "--reinforce_weight",
        type=float,
        default=None,
        help="Weight of S0 tour-cost REINFORCE loss (default: 0.1 at every size)",
    )
    parser.add_argument(
        "--show_loss_values",
        action="store_true",
        help="Print per-epoch mean KL, REINFORCE, and GCN losses; off by default",
    )
    parser.add_argument(
        "--probe_instances", type=int, default=0,
        help="Number of separate fixed diagnostic instances (0 disables H0/H1 probe)",
    )
    parser.add_argument(
        "--probe_repeats", type=int, default=3,
        help="Independent frozen-pheromone populations per arm and instance",
    )
    parser.add_argument(
        "--probe_seed", type=int, default=None,
        help="Diagnostic pool and sampling seed (default: training seed + 200000)",
    )
    parser.add_argument(
        "--probe_output", type=str, default=None,
        help="Separate folder for paired-probe records and three ratio charts",
    )
    parser.add_argument(
        "--graph_structure", choices=("whole_tour", "local"),
        default="whole_tour",
        help="Legacy fixed H1 relation: whole-tour or consecutive tour edges",
    )
    parser.add_argument(
        "--graph_layers", type=int, default=2,
        help="Number of local GCN or legacy fixed aggregation layers",
    )
    parser.add_argument(
        "--graph_quality_mode",
        choices=("inverse_cost", "relative", "relative_diversity"),
        default=None,
        help="H1 route weighting (default: relative_diversity for local_gcn; inverse_cost for legacy modes)",
    )
    parser.add_argument(
        "--graph_quality_temperature", type=float, default=0.1,
        help="Scale of relative route-quality weights (fraction of mean cost)",
    )
    parser.add_argument(
        "--graph_diversity_strength", type=float, default=0.5,
        help="Structural novelty contribution for relative_diversity (0 to 1)",
    )
    parser.add_argument(
        "--graph_candidate_only", action="store_true",
        help="Restrict H1 graph nodes and output to H0 candidate edges",
    )
    parser.add_argument(
        "--graph_supervision_weight", type=float, default=1.0,
        help="Weight of elite-tour edge supervision for the trainable local GCN",
    )
    parser.add_argument(
        "--max_solution_graph_solutions",
        type=int,
        default=None,
        help="Optional graph tour cap; default keeps all archived tours",
    )
    parser.add_argument(
        "--max_archive_solutions",
        type=int,
        default=None,
        help="Optional cap on sampled tours retained by each instance archive",
    )
    parser.add_argument(
        "--max_archive_rounds",
        type=int,
        default=None,
        help="Optional cap on ACO populations retained by each instance archive",
    )
    parser.add_argument(
        "--refinement_mode",
        choices=REFINEMENT_MODES,
        default="local_gcn",
        help=(
            "H1 teacher: local_gcn trains a local edge GCN; deterministic "
            "and learned/full retain the earlier fixed ablations"
        ),
    )
    parser.add_argument(
        "--local_search",
        choices=("none", "2opt", "nls"),
        default="nls",
        help="Local search used in training and validation (default: nls)",
    )
    parser.add_argument("-t", "--test_size", type=int, default=None,
                        help="Number of instances used for validation")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Directory used to store checkpoints (selected by the profile)",
    )
    opt = parser.parse_args()

    try:
        resolved = resolve_training_profile(
            opt.nodes,
            opt.profile,
            lr=opt.lr,
            epochs=opt.epochs,
            k_sparse=opt.k_sparse,
            train_pool_size=opt.train_pool_size,
            pretrained=opt.pretrained,
            output=opt.output,
        )
    except ValueError as exc:
        parser.error(str(exc))

    opt.lr = resolved["lr"]
    opt.epochs = resolved["epochs"]
    opt.k_sparse = resolved["k_sparse"]
    opt.train_pool_size = resolved["train_pool_size"]
    opt.pretrained = resolved["pretrained"]
    opt.output = resolved["output"]
    opt.local_search = None if opt.local_search == "none" else opt.local_search
    opt.kl_weight, opt.reinforce_weight = resolve_loss_weights(
        opt.nodes, opt.kl_weight, opt.reinforce_weight,
    )

    if opt.graph_rounds != DEFAULT_GRAPH_ROUNDS:
        parser.error("--graph_rounds must be 1 (H0 -> H1 only)")
    required_pool_size = opt.steps * opt.batch_size
    if opt.train_pool_size != required_pool_size:
        parser.error(
            "--train_pool_size must equal --steps * --batch_size "
            f"({required_pool_size}) so every instance is visited once per epoch"
        )
    if not 0 <= opt.pool_refresh_fraction <= 1:
        parser.error("--pool_refresh_fraction must be between 0 and 1")
    if opt.k_sparse is not None and not 1 <= opt.k_sparse < opt.nodes:
        parser.error("--k_sparse must be in [1, nodes - 1]")
    try:
        validate_loss_weights(opt.kl_weight, opt.reinforce_weight)
    except ValueError as exc:
        parser.error(str(exc))
    if opt.graph_layers < 1:
        parser.error("--graph_layers must be positive")
    if opt.graph_quality_temperature <= 0:
        parser.error("--graph_quality_temperature must be positive")
    if not 0 <= opt.graph_diversity_strength <= 1:
        parser.error("--graph_diversity_strength must be between 0 and 1")
    if opt.graph_quality_mode == "relative_diversity" and opt.refinement_mode != "local_gcn":
        parser.error("--graph_quality_mode relative_diversity requires local_gcn")
    if opt.refinement_mode == "local_gcn" and opt.kl_weight > 0 and opt.graph_supervision_weight <= 0:
        parser.error("--graph_supervision_weight must be positive for local_gcn")
    if opt.max_solution_graph_solutions is not None and opt.max_solution_graph_solutions < 1:
        parser.error("--max_solution_graph_solutions must be positive")
    if opt.max_archive_solutions is not None and opt.max_archive_solutions < 1:
        parser.error("--max_archive_solutions must be positive")
    if opt.max_archive_rounds is not None and opt.max_archive_rounds < 1:
        parser.error("--max_archive_rounds must be positive")
    if opt.probe_instances < 0 or opt.probe_repeats < 1:
        parser.error("--probe_instances must be non-negative and --probe_repeats positive")
    if opt.probe_instances and opt.refinement_mode != "local_gcn":
        parser.error("--probe_instances requires --refinement_mode local_gcn")
    if (
        opt.max_solution_graph_solutions is not None
        and opt.max_archive_solutions is not None
        and opt.max_solution_graph_solutions > opt.max_archive_solutions
    ):
        parser.error(
            "--max_solution_graph_solutions cannot exceed "
            "--max_archive_solutions"
        )
    if opt.refinement_mode == "h0":
        parser.error(
            "--refinement_mode h0 cannot be trained because it has no "
            "refined H1 target"
        )

    if opt.pretrained is not None and not os.path.isfile(opt.pretrained):
        parser.error(f"pretrained checkpoint not found: {opt.pretrained}")

    os.makedirs(opt.output, exist_ok=True)
    lr = opt.lr
    device = opt.device

    train(
        opt.nodes,
        opt.ants,
        opt.steps,
        opt.epochs,
        k_sparse=opt.k_sparse,
        batch_size=opt.batch_size,
        test_size=opt.test_size,
        pretrained=opt.pretrained,
        savepath=opt.output,
        graph_rounds=opt.graph_rounds,
        kl_weight=opt.kl_weight,
        reinforce_weight=opt.reinforce_weight,
        train_pool_size=opt.train_pool_size,
        max_solution_graph_solutions=opt.max_solution_graph_solutions,
        max_archive_solutions=opt.max_archive_solutions,
        max_archive_rounds=opt.max_archive_rounds,
        refinement_mode=opt.refinement_mode,
        local_search=opt.local_search,
        train_pool_mode=opt.train_pool_mode,
        pool_refresh_fraction=opt.pool_refresh_fraction,
        seed=opt.seed,
        graph_structure=opt.graph_structure,
        graph_layers=opt.graph_layers,
        graph_quality_mode=opt.graph_quality_mode,
        graph_quality_temperature=opt.graph_quality_temperature,
        graph_diversity_strength=opt.graph_diversity_strength,
        graph_candidate_only=opt.graph_candidate_only,
        graph_supervision_weight=opt.graph_supervision_weight,
        show_loss_values=opt.show_loss_values,
        probe_instances=opt.probe_instances,
        probe_repeats=opt.probe_repeats,
        probe_seed=opt.probe_seed,
        probe_output=opt.probe_output,
    )
