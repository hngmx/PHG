import unittest

import torch

from training_pool import (
    create_training_pool,
    iter_pool_batches,
    refresh_expired_instances,
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

    def test_pool_rejects_batches_larger_than_the_pool(self):
        pool = create_training_pool(count=2, n_nodes=5)
        with self.assertRaises(ValueError):
            list(iter_pool_batches(pool, steps=1, batch_size=3))

    def test_expired_instances_are_replaced_but_others_persist(self):
        pool = create_training_pool(count=3, n_nodes=5)
        old_ids = [id(instance) for instance in pool]
        pool[0].visits = 5
        pool[1].visits = 4
        pool[2].visits = 6

        replaced = refresh_expired_instances(pool, max_visits=5)

        self.assertEqual(replaced, 2)
        self.assertNotEqual(id(pool[0]), old_ids[0])
        self.assertEqual(id(pool[1]), old_ids[1])
        self.assertNotEqual(id(pool[2]), old_ids[2])
        self.assertEqual(pool[0].visits, 0)
        self.assertEqual(pool[0].n_nodes, 5)


if __name__ == "__main__":
    unittest.main()
