import unittest

import torch

from training_pool import (
    create_training_pool,
    iter_pool_batches,
    refresh_training_pool,
)


class PersistentTrainingPoolTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_default_epoch_style_batching_visits_every_instance_once(self):
        pool = create_training_pool(count=6, n_nodes=5)
        batches = list(iter_pool_batches(pool, steps=2, batch_size=3))
        visited = [id(instance) for batch in batches for instance in batch]

        self.assertEqual(len(visited), 6)
        self.assertEqual(len(set(visited)), 6)

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

    def test_instance_starts_each_visit_with_an_empty_archive(self):
        instance = create_training_pool(count=1, n_nodes=5)[0]
        first_state = instance.start_visit(torch.ones(5, 5))
        first_state.add_feasible_solutions(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        first_state.advance(torch.full((5, 5), 2.0))
        instance.finish_visit()

        latest_h0 = torch.ones(5, 5) - torch.eye(5)
        latest_h0[0, 1] = 4.0
        second_state = instance.start_visit(latest_h0)

        self.assertIsNot(second_state, first_state)
        self.assertEqual(instance.visits, 1)
        self.assertEqual(len(second_state.archive), 0)
        self.assertEqual(second_state.archive.num_rounds, 0)
        self.assertTrue(torch.equal(second_state.current_heatmap, latest_h0))
        self.assertGreater(
            float(second_state.current_heatmap[0, 1]),
            float(second_state.current_heatmap[0, 2]),
        )
        self.assertFalse(second_state.current_heatmap.requires_grad)
        self.assertEqual(second_state.current_heatmap.device.type, "cpu")

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
