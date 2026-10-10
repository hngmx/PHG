import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch

import test as test_entrypoint
from train import (
    DEFAULT_GRAPH_ROUNDS,
    VALIDATION_SAMPLING_ROUNDS,
    infer_instance as infer_validation_instance,
    pool_refresh_fraction_for_mode,
    resolve_training_profile,
    sampling_rounds_for_refinements,
)


class TrainingProfileTest(unittest.TestCase):
    def test_default_training_uses_one_heatmap_refinement(self):
        self.assertEqual(DEFAULT_GRAPH_ROUNDS, 1)
        self.assertEqual(VALIDATION_SAMPLING_ROUNDS, 2)

    def test_each_refined_heatmap_gets_an_aco_population(self):
        self.assertEqual(sampling_rounds_for_refinements(1), 2)
        with self.assertRaises(ValueError):
            sampling_rounds_for_refinements(0)
        with self.assertRaises(ValueError):
            sampling_rounds_for_refinements(2)

    def test_validation_rejects_h2_rounds(self):
        with self.assertRaises(ValueError):
            infer_validation_instance(None, None, None, 48, graph_rounds=3)

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

    def test_pool_modes_resolve_expected_refresh_fraction(self):
        self.assertEqual(pool_refresh_fraction_for_mode("fixed"), 0.0)
        self.assertEqual(pool_refresh_fraction_for_mode("refresh"), 1.0)
        self.assertEqual(pool_refresh_fraction_for_mode("mixed", 0.25), 0.25)
        with self.assertRaises(ValueError):
            pool_refresh_fraction_for_mode("unknown")

    def test_standard_profile_uses_one_epoch_sized_training_pool(self):
        resolved = resolve_training_profile(100, "standard")

        self.assertEqual(resolved["lr"], 3e-4)
        self.assertEqual(resolved["epochs"], 20)
        self.assertEqual(resolved["train_pool_size"], 800)

    def test_tsp100_finetune_profile_uses_800_instance_pool(self):
        resolved = resolve_training_profile(100, "tsp100_finetune")

        self.assertEqual(resolved["lr"], 1e-4)
        self.assertEqual(resolved["epochs"], 3)
        self.assertEqual(resolved["k_sparse"], 10)
        self.assertEqual(resolved["train_pool_size"], 800)
        self.assertTrue(resolved["pretrained"].endswith("tsp100-best.pt"))
        self.assertTrue(
            resolved["output"].endswith("optimized_v3_k10_finetune")
        )

    def test_explicit_values_override_profile_defaults(self):
        resolved = resolve_training_profile(
            100,
            "tsp100_finetune",
            lr=2e-4,
            epochs=5,
            output="custom-output",
        )

        self.assertEqual(resolved["lr"], 2e-4)
        self.assertEqual(resolved["epochs"], 5)
        self.assertEqual(resolved["output"], "custom-output")
        self.assertEqual(resolved["k_sparse"], 10)

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
