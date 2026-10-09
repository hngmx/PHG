import unittest

import torch

import test as test_entrypoint
from train import (
    DEFAULT_GRAPH_ROUNDS,
    population_cost_heatmap,
    pool_refresh_fraction_for_mode,
    refined_population_cost_loss,
    resolve_training_profile,
    sampling_rounds_for_refinements,
)


class TrainingProfileTest(unittest.TestCase):
    def test_default_training_uses_one_heatmap_refinement(self):
        self.assertEqual(DEFAULT_GRAPH_ROUNDS, 1)

    def test_each_refined_heatmap_gets_an_aco_population(self):
        self.assertEqual(sampling_rounds_for_refinements(3), 4)
        with self.assertRaises(ValueError):
            sampling_rounds_for_refinements(0)

    def test_initial_cost_metric_does_not_train_h0(self):
        h0 = torch.rand(4, 4, requires_grad=True)

        selected = population_cost_heatmap(h0, [], sampling_round=0)

        self.assertFalse(selected.requires_grad)
        self.assertEqual(selected.data_ptr(), h0.data_ptr())

    def test_refined_cost_loss_can_train_graph_updater(self):
        h0 = torch.rand(4, 4, requires_grad=True)
        h1 = torch.rand(4, 4, requires_grad=True)

        selected = population_cost_heatmap(h0, [h1], sampling_round=1)

        self.assertIs(selected, h1)
        self.assertTrue(selected.requires_grad)

    def test_cost_objective_excludes_h0_diagnostic(self):
        h0_metric = torch.tensor(100.0)
        h1_loss = torch.tensor(2.0, requires_grad=True)

        loss = refined_population_cost_loss([h0_metric, h1_loss])
        loss.backward()

        self.assertEqual(float(loss.detach()), 2.0)
        self.assertEqual(float(h1_loss.grad), 1.0)

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
        self.assertEqual(resolved["train_pool_size"], 400)
        self.assertEqual(resolved["kl_round_power"], 1.0)

    def test_tsp100_finetune_profile_matches_legacy_preset(self):
        resolved = resolve_training_profile(100, "tsp100_finetune")

        self.assertEqual(resolved["lr"], 1e-4)
        self.assertEqual(resolved["epochs"], 3)
        self.assertEqual(resolved["k_sparse"], 10)
        self.assertEqual(resolved["train_pool_size"], 400)
        self.assertEqual(resolved["kl_round_power"], 0.0)
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
