import unittest

import test as test_entrypoint
from train import resolve_training_profile


class TrainingProfileTest(unittest.TestCase):
    def test_tsp100_finetune_profile_matches_validated_experiment(self):
        resolved = resolve_training_profile(100, "tsp100_finetune")

        self.assertEqual(resolved["lr"], 1e-4)
        self.assertEqual(resolved["epochs"], 3)
        self.assertEqual(resolved["k_sparse"], 10)
        self.assertEqual(resolved["train_pool_size"], 800)
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
