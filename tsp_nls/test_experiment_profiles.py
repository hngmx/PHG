import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import torch

import test as test_entrypoint
import train as train_entrypoint
from solution_graph import InstanceSearchState
from utils import gen_pyg_data
from train import (
    DEFAULT_GRAPH_ROUNDS,
    H0_VALIDATION_ITERATIONS,
    format_training_epoch,
    pool_refresh_fraction_for_mode,
    resolve_training_profile,
    train_instance,
    validate_graph_rounds,
    validation_h0_only,
)


class TrainingProfileTest(unittest.TestCase):
    def test_default_training_uses_one_heatmap_refinement(self):
        self.assertEqual(DEFAULT_GRAPH_ROUNDS, 1)

    def test_training_rejects_extra_heatmap_refinements(self):
        validate_graph_rounds(1)
        with self.assertRaises(ValueError):
            validate_graph_rounds(0)
        with self.assertRaises(ValueError):
            validate_graph_rounds(2)

    def test_training_samples_s0_once_and_optimizes_only_distillation(self):
        calls = []
        sample_kwargs = []

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.3))

            @property
            def device(self):
                return self.scale.device

            def forward(self, data):
                return None

            def reshape(self, data, values):
                base = torch.tensor([
                    [0.0, 1.0, 2.0, 3.0],
                    [2.0, 0.0, 1.0, 3.0],
                    [3.0, 2.0, 0.0, 1.0],
                    [1.0, 3.0, 2.0, 0.0],
                ])
                return torch.sigmoid(self.scale * base)

            def refine_heatmap(self, prior, distances, archive, **kwargs):
                calls.append("refine")
                self.assert_no_grad = not torch.is_grad_enabled()
                self.archive_size = len(archive)
                self.refinement_prior = prior.detach().clone()
                self.prior_strength = kwargs["prior_strength"]
                teacher = torch.tensor([
                    [0.0, 0.7, 0.2, 0.1],
                    [0.7, 0.0, 0.2, 0.1],
                    [0.2, 0.2, 0.0, 0.6],
                    [0.1, 0.1, 0.8, 0.0],
                ])
                return teacher, None, None, None

        class FakeInstance:
            visits = 0

            def materialize(self, k_sparse, device):
                return None, torch.ones(4, 4)

            def start_visit(self, initial_heatmap, **kwargs):
                self.state = InstanceSearchState(initial_heatmap.detach().cpu())
                return self.state

            def finish_visit(self):
                self.visits += 1

        class FakeSampler:
            def __init__(self, **kwargs):
                calls.append("create")

            def sample(self, **kwargs):
                calls.append("sample")
                sample_kwargs.append(kwargs)
                return SimpleNamespace(
                    feasible_paths=torch.tensor([
                        [0, 0], [1, 2], [2, 1], [3, 3],
                    ]),
                    feasible_costs=torch.tensor([4.0, 5.0]),
                )

        model = FakeModel()
        instance = FakeInstance()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_scale = model.scale.item()
        with patch.object(train_entrypoint, "device", "cpu", create=True):
            train_instance(
                model, optimizer, [instance], n_ants=2, k_sparse=2,
                sampler_factory=FakeSampler,
            )

        self.assertEqual(calls, ["create", "sample", "refine"])
        self.assertFalse(sample_kwargs[0]["require_log_probs"])
        self.assertTrue(model.assert_no_grad)
        self.assertEqual(model.archive_size, 2)
        self.assertTrue(torch.equal(
            model.refinement_prior, instance.state.initial_heatmap
        ))
        self.assertEqual(model.prior_strength, 0.1)
        self.assertEqual(instance.state.round_index, 0)
        self.assertTrue(torch.equal(
            instance.state.current_heatmap, instance.state.initial_heatmap
        ))
        self.assertEqual(instance.visits, 1)
        self.assertNotEqual(model.scale.item(), initial_scale)

    def test_h0_checkpoint_validation_uses_test_inference_path(self):
        with patch(
            "train.infer_h0_test_instance",
            side_effect=[[8.0, 7.0], [6.0, 5.0]],
        ) as infer:
            curve = validation_h0_only(
                48, None, [(None, None), (None, None)], n_iterations=2,
            )
        self.assertEqual(curve, [7.0, 6.0])
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(infer.call_args.kwargs["sampling_backend"], "torch")
        self.assertEqual(H0_VALIDATION_ITERATIONS, 10)

    def test_training_epoch_display_keeps_four_value_style(self):
        metrics = [11.0, 9.0, 8.0, 7.0]
        self.assertEqual(
            format_training_epoch(2, metrics),
            "epoch 2: [11.0, 9.0, 8.0, 7.0]",
        )
        with self.assertRaises(ValueError):
            format_training_epoch(2, metrics[:3])

    def test_training_validation_keeps_h0_t10_checkpoint_curve(self):
        results = [
            ([10.0] * 9 + [7.0], [12.0, 10.0, 9.0, 8.0]),
            ([9.0] * 9 + [6.0], [11.0, 9.0, 8.0, 7.0]),
        ]
        with patch("train.infer_h0_test_instance", side_effect=results) as infer:
            curve, metrics = validation_h0_only(
                48, None, [(None, None), (None, None)],
                return_training_metrics=True,
            )
        self.assertEqual(curve[-1], 6.5)
        self.assertEqual(metrics, [11.5, 9.5, 8.5, 7.5])
        self.assertTrue(infer.call_args.kwargs["return_training_metrics"])

    def test_deepaco_training_metrics_use_initial_mean_min_and_aco_best(self):
        populations = [
            [12.0, 8.0], [9.0, 7.0], [10.0, 9.0],
            [8.0, 7.5], [9.0, 8.0], [6.0, 5.0],
        ]

        class FakeModel:
            def eval(self):
                pass

            def __call__(self, data):
                return None

            def reshape(self, data, values):
                return torch.ones(4, 4)

        class FakeSampler:
            def __init__(self, **kwargs):
                self.best_cost = float("inf")
                self.iteration = 0

            def probe(self, **kwargs):
                self.iteration += 1
                return torch.tensor(populations[0])

            def sample(self, **kwargs):
                costs = torch.tensor(populations[self.iteration])
                self.iteration += 1
                self.best_cost = min(self.best_cost, float(costs.min()))
                return SimpleNamespace(sampled_costs=costs)

        curve, metrics = test_entrypoint.infer_instance(
            FakeModel(), None, torch.ones(4, 4), 2,
            n_iterations=5, sampler_factory=FakeSampler,
            return_training_metrics=True,
        )
        self.assertEqual(metrics, [10.0, 8.0, 7.0, 5.0])
        self.assertEqual(curve, [7.0, 7.0, 7.0, 7.0, 5.0])

    def test_test_entrypoint_samples_only_s0(self):
        calls = []

        class FakeModel:
            def eval(self):
                calls.append("eval")

            def __call__(self, data):
                return None

            def reshape(self, data, values):
                return torch.ones(4, 4)

            def refine_heatmap(self, *args, **kwargs):
                raise AssertionError("S0-only test must not construct H1")

        class FakeSampler:
            def __init__(self, **kwargs):
                calls.append("create")
                self.best_cost = 10.0

            def sample(self, **kwargs):
                calls.append("sample")
                self.best_cost -= 1.0

        costs = test_entrypoint.infer_instance(
            FakeModel(), None, torch.ones(4, 4), 4,
            n_iterations=3,
            sampler_factory=FakeSampler,
        )

        self.assertEqual(calls, ["eval", "create", "sample", "sample", "sample"])
        self.assertEqual(costs, [9.0, 8.0, 7.0])

    def test_test_aggregates_without_printing_each_instance(self):
        output = StringIO()
        with patch.object(
            test_entrypoint,
            "infer_instance",
            side_effect=[[4.0, 3.0, 3.0], [6.0, 5.0, 4.0]],
        ), redirect_stdout(output):
            averages, _ = test_entrypoint.test(
                [(None, None), (None, None)], None, 4,
                n_iterations=3,
            )

        self.assertEqual(averages, [5.0, 4.0, 3.5])
        self.assertEqual(output.getvalue(), "")

    def test_test_summary_keeps_t_by_t_output_style(self):
        class FakeNet:
            def to(self, device):
                return self

            def load_state_dict(self, state, strict=False):
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        output = StringIO()
        with (
            patch.object(test_entrypoint, "load_test_dataset", return_value=[None, None]),
            patch.object(test_entrypoint, "Net", FakeNet),
            patch.object(test_entrypoint.torch, "load", return_value={}),
            patch.object(test_entrypoint, "test", return_value=([8.0, 7.0], 1.5)),
            redirect_stdout(output),
        ):
            test_entrypoint.main(50, "checkpoint.pt", n_iterations=2)

        lines = output.getvalue().splitlines()
        self.assertEqual(lines[:3], [
            "problem scale: 50",
            "checkpoint: checkpoint.pt",
            "number of instances: 2",
        ])
        self.assertEqual(lines[-3:], [
            "total duration: 1.5",
            "T=1, average cost is 8.0.",
            "T=2, average cost is 7.0.",
        ])
        self.assertFalse(any("sampling backend:" in line for line in lines))

    def test_pool_modes_resolve_expected_refresh_fraction(self):
        self.assertEqual(pool_refresh_fraction_for_mode("fixed"), 0.0)
        self.assertEqual(pool_refresh_fraction_for_mode("refresh"), 1.0)
        self.assertEqual(pool_refresh_fraction_for_mode("mixed", 0.25), 0.25)
        with self.assertRaises(ValueError):
            pool_refresh_fraction_for_mode("unknown")

    def test_standard_profile_uses_one_epoch_sized_training_pool(self):
        resolved = resolve_training_profile(50, "standard")

        self.assertEqual(resolved["lr"], 3e-4)
        self.assertEqual(resolved["epochs"], 20)
        self.assertEqual(resolved["k_sparse"], 49)
        self.assertEqual(resolved["train_pool_size"], 400)

    def test_tsp100_finetune_profile_uses_400_instance_pool(self):
        resolved = resolve_training_profile(100, "tsp100_finetune")

        self.assertEqual(resolved["lr"], 1e-4)
        self.assertEqual(resolved["epochs"], 3)
        self.assertEqual(resolved["k_sparse"], 99)
        self.assertEqual(resolved["train_pool_size"], 400)
        self.assertTrue(resolved["pretrained"].endswith("tsp100-best.pt"))
        self.assertTrue(
            resolved["output"].endswith("optimized_v3_full_finetune")
        )

    def test_explicit_values_override_profile_defaults(self):
        resolved = resolve_training_profile(
            100,
            "tsp100_finetune",
            lr=2e-4,
            epochs=5,
            k_sparse=10,
            output="custom-output",
        )

        self.assertEqual(resolved["lr"], 2e-4)
        self.assertEqual(resolved["epochs"], 5)
        self.assertEqual(resolved["output"], "custom-output")
        self.assertEqual(resolved["k_sparse"], 10)

    def test_full_candidate_graph_excludes_only_self_edges(self):
        coordinates = torch.tensor([
            [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
        ])
        graph, _ = gen_pyg_data(coordinates, k_sparse=3, start_node=0)
        edges = {tuple(edge) for edge in graph.edge_index.T.tolist()}

        self.assertEqual(len(edges), 4 * 3)
        self.assertEqual(edges, {
            (source, target)
            for source in range(4)
            for target in range(4)
            if source != target
        })

    def test_tsp100_profile_rejects_other_problem_sizes(self):
        with self.assertRaises(ValueError):
            resolve_training_profile(200, "tsp100_finetune")


class CheckpointSelectionTest(unittest.TestCase):
    def test_tsp100_defaults_to_root_best_checkpoint(self):
        candidates = test_entrypoint.default_checkpoint_candidates(100)
        selected = test_entrypoint.resolve_checkpoint_path(100)

        self.assertEqual(selected, candidates[0])
        self.assertTrue(selected.endswith("tsp100-best.pt"))
        self.assertNotIn("optimized_v3_k10_finetune", selected)

    def test_explicit_model_always_wins(self):
        selected = test_entrypoint.resolve_checkpoint_path(
            100, requested="manual.pt"
        )
        self.assertEqual(selected, "manual.pt")


if __name__ == "__main__":
    unittest.main()
