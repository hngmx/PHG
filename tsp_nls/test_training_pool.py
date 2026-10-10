import unittest

import torch

from training_pool import (
    create_training_pool,
    iter_pool_batches,
    refresh_training_pool,
)
from solution_graph import normalize_heatmap_rows


class PersistentTrainingPoolTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_default_epoch_style_batching_visits_every_instance_once(self):
        pool = create_training_pool(count=6, n_nodes=5)
        batches = list(iter_pool_batches(pool, steps=2, batch_size=3))
        visited = [id(instance) for batch in batches for instance in batch]

        self.assertEqual(len(visited), 6)
        self.assertEqual(len(set(visited)), 6)

    def test_800_instance_epoch_visits_every_instance_once(self):
        pool = create_training_pool(count=800, n_nodes=5)
        batches = list(iter_pool_batches(pool, steps=40, batch_size=20))
        visited = [id(instance) for batch in batches for instance in batch]

        self.assertEqual(len(batches), 40)
        self.assertEqual(len(visited), 800)
        self.assertEqual(len(set(visited)), 800)

    def test_same_pool_is_visited_once_per_epoch_for_all_epochs(self):
        pool = create_training_pool(count=6, n_nodes=5)
        expected = {id(instance) for instance in pool}

        for _ in range(4):
            batches = iter_pool_batches(pool, steps=2, batch_size=3)
            visited = [instance for batch in batches for instance in batch]
            self.assertEqual({id(instance) for instance in visited}, expected)
            self.assertEqual(len(visited), len(expected))
            for instance in visited:
                instance.finish_visit()

        self.assertEqual([instance.visits for instance in pool], [4] * 6)

    def test_instance_reuses_archive_and_h1_but_samples_from_fresh_h0(self):
        instance = create_training_pool(count=1, n_nodes=5)[0]
        first_state = instance.start_visit(torch.ones(5, 5))
        self.assertIs(first_state.refinement_prior_heatmap, first_state.current_heatmap)
        first_state.add_feasible_solutions(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        first_state.advance(torch.full((5, 5), 2.0))
        first_state.add_feasible_solutions(
            torch.tensor([[0], [2], [1], [3], [4]]),
            torch.tensor([4.0]),
        )
        instance.finish_visit()

        latest_h0 = torch.ones(5, 5) - torch.eye(5)
        latest_h0[0, 1] = 4.0
        second_state = instance.start_visit(latest_h0)

        self.assertIsNot(second_state, first_state)
        self.assertEqual(instance.visits, 1)
        self.assertIs(second_state.archive, first_state.archive)
        self.assertEqual(len(second_state.archive), 2)
        self.assertEqual(second_state.archive.num_rounds, 2)
        self.assertEqual(second_state.round_index, 0)
        self.assertTrue(torch.equal(second_state.current_heatmap, latest_h0))
        expected_prior = 0.5 * (
            normalize_heatmap_rows(latest_h0)
            + normalize_heatmap_rows(first_state.current_heatmap)
        )
        self.assertTrue(
            torch.allclose(second_state.refinement_prior_heatmap, expected_prior)
        )
        self.assertGreater(
            float(second_state.current_heatmap[0, 1]),
            float(second_state.current_heatmap[0, 2]),
        )
        self.assertFalse(second_state.current_heatmap.requires_grad)
        self.assertEqual(second_state.current_heatmap.device.type, "cpu")
        self.assertFalse(second_state.refinement_prior_heatmap.requires_grad)

    def test_history_prior_weight_and_limits_are_checked(self):
        instance = create_training_pool(count=1, n_nodes=5)[0]
        with self.assertRaises(ValueError):
            instance.start_visit(torch.ones(5, 5), history_heatmap_weight=1.1)
        first = instance.start_visit(
            torch.ones(5, 5), archive_max_solutions=8
        )
        first.advance(torch.eye(5) + 2)
        instance.finish_visit()
        fresh_h0 = torch.ones(5, 5) - torch.eye(5)
        with self.assertRaises(ValueError):
            instance.start_visit(fresh_h0, archive_max_solutions=4)
        second = instance.start_visit(
            fresh_h0,
            archive_max_solutions=8,
            history_heatmap_weight=0,
        )
        self.assertTrue(torch.allclose(
            second.refinement_prior_heatmap,
            normalize_heatmap_rows(fresh_h0),
        ))

    def test_pool_rejects_size_different_from_epoch_demand(self):
        pool = create_training_pool(count=2, n_nodes=5)
        with self.assertRaises(ValueError):
            list(iter_pool_batches(pool, steps=1, batch_size=3))
        with self.assertRaises(ValueError):
            list(iter_pool_batches(pool, steps=1, batch_size=1))

    def test_pool_refresh_modes_replace_exact_fraction(self):
        pool = create_training_pool(count=6, n_nodes=5)
        original_instances = list(pool)

        self.assertEqual(refresh_training_pool(pool, fraction=0.0), 0)
        self.assertTrue(
            all(
                instance is original
                for instance, original in zip(pool, original_instances)
            )
        )

        self.assertEqual(refresh_training_pool(pool, fraction=0.5), 3)
        changed = sum(
            instance is not original
            for instance, original in zip(pool, original_instances)
        )
        self.assertEqual(changed, 3)

        instances_before_full_refresh = list(pool)
        self.assertEqual(refresh_training_pool(pool, fraction=1.0), 6)
        self.assertTrue(
            all(
                instance is not original
                for instance, original in zip(
                    pool,
                    instances_before_full_refresh,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
