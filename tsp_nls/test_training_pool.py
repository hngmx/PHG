import unittest

import torch

from training_pool import (
    create_training_pool,
    iter_pool_batches,
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

    def test_instance_reuses_heatmap_and_archive_on_later_visit(self):
        instance = create_training_pool(count=1, n_nodes=5)[0]
        first_state = instance.get_or_create_state(torch.ones(5, 5))
        first_state.add_feasible_solutions(
            torch.tensor([[0], [1], [2], [3], [4]]),
            torch.tensor([5.0]),
        )
        first_state.advance(torch.full((5, 5), 2.0))
        instance.finish_visit()

        latest_h0 = torch.ones(5, 5) - torch.eye(5)
        latest_h0[0, 1] = 4.0
        second_state = instance.get_or_create_state(
            latest_h0,
            memory_strength=0.5,
        )

        self.assertIs(second_state, first_state)
        self.assertEqual(instance.visits, 1)
        self.assertEqual(len(second_state.archive), 1)
        self.assertEqual(second_state.archive.num_rounds, 1)
        self.assertFalse(
            torch.equal(second_state.current_heatmap, torch.full((5, 5), 2.0))
        )
        self.assertGreater(
            float(second_state.current_heatmap[0, 1]),
            float(second_state.current_heatmap[0, 2]),
        )
        self.assertTrue(
            torch.allclose(
                second_state.current_heatmap.sum(dim=-1),
                torch.ones(5),
                atol=1e-6,
            )
        )
        self.assertFalse(second_state.current_heatmap.requires_grad)
        self.assertEqual(second_state.current_heatmap.device.type, "cpu")

    def test_pool_rejects_size_different_from_epoch_demand(self):
        pool = create_training_pool(count=2, n_nodes=5)
        with self.assertRaises(ValueError):
            list(iter_pool_batches(pool, steps=1, batch_size=3))
        with self.assertRaises(ValueError):
            list(iter_pool_batches(pool, steps=1, batch_size=1))


if __name__ == "__main__":
    unittest.main()
