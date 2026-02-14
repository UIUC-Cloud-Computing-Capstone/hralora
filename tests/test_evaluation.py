"""
Unit tests for test.py (evaluation utilities).

Tests cover: collate_fn (batch shape and label keys), and test() with minimal
model/dataset (no attack, accuracy and loss return shape). test_vit and test_ledgar
are integration-heavy; only collate_fn and test() are unit-tested here.
"""
import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from project root test.py (evaluation module)
from test import collate_fn, test


# -----------------------------------------------------------------------------
# collate_fn
# -----------------------------------------------------------------------------
class TestCollateFn(unittest.TestCase):
    def test_output_keys(self):
        examples = [
            {"pixel_values": torch.randn(3, 224, 224), "label": 0},
            {"pixel_values": torch.randn(3, 224, 224), "label": 1},
        ]
        batch = collate_fn(examples)
        self.assertIn("pixel_values", batch)
        self.assertIn("labels", batch)

    def test_label_key(self):
        examples = [
            {"pixel_values": torch.randn(3, 32, 32), "label": 5},
            {"pixel_values": torch.randn(3, 32, 32), "label": 7},
        ]
        batch = collate_fn(examples)
        self.assertEqual(batch["labels"].tolist(), [5, 7])

    def test_fine_label_key(self):
        examples = [
            {"pixel_values": torch.randn(3, 32, 32), "fine_label": 10},
            {"pixel_values": torch.randn(3, 32, 32), "fine_label": 20},
        ]
        batch = collate_fn(examples)
        self.assertEqual(batch["labels"].tolist(), [10, 20])

    def test_pixel_values_shape(self):
        examples = [
            {"pixel_values": torch.randn(3, 64, 64), "label": 0},
            {"pixel_values": torch.randn(3, 64, 64), "label": 1},
        ]
        batch = collate_fn(examples)
        self.assertEqual(batch["pixel_values"].shape, (2, 3, 64, 64))


# -----------------------------------------------------------------------------
# test() — generic evaluation, no attack
# -----------------------------------------------------------------------------
class TestTestFunction(unittest.TestCase):
    def _args(self, dataset="mnist", attack="None"):
        a = argparse.Namespace()
        a.dataset = dataset
        a.batch_size = 4
        a.test_batch_size = 4
        a.device = torch.device("cpu")
        a.attack = attack
        a.num_attackers = 0
        a.num_selected_users = 1
        a.trigger_num = 1
        return a

    def test_returns_accuracy_and_loss(self):
        # Minimal model and dataset
        net = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        dataset = TensorDataset(x, y)
        args = self._args(dataset="other")

        acc, loss = test(net, dataset, args)

        self.assertIsInstance(acc, (int, float))
        self.assertIsInstance(loss, (int, float))
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 100)
        self.assertGreaterEqual(loss, 0)

    def test_femnist_uses_collate_fn_batch_structure(self):
        # FEMNIST path: batch is dict with pixel_values and labels
        net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 5))
        # Dataset that returns dicts like FEMNIST
        class FemnistStyleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, i):
                return {
                    "pixel_values": torch.randn(3, 32, 32),
                    "label": i % 5,
                }
        dataset = FemnistStyleDataset()
        args = self._args(dataset="femnist")

        acc, loss = test(net, dataset, args)

        self.assertIsInstance(acc, (int, float))
        self.assertIsInstance(loss, (int, float))

    def test_shakespeare_branch_uses_dim_neg2_for_predicted(self):
        # Shakespeare: torch.max(log_probs, -2) so log_probs must be (batch, vocab, seq)
        class TinyShakespeareNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 4)
            def forward(self, x):
                out = self.linear(x)
                return out.transpose(-2, -1)
        net = TinyShakespeareNet()
        x = torch.randn(4, 5, 10)
        y = torch.randint(0, 4, (4, 5))
        dataset = TensorDataset(x, y)
        args = self._args(dataset="shakespeare")

        acc, loss = test(net, dataset, args)

        self.assertIsInstance(acc, (int, float))
        self.assertIsInstance(loss, (int, float))


if __name__ == "__main__":
    unittest.main()
