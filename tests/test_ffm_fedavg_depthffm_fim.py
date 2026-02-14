"""
Unit tests for algorithms/engine/ffm_fedavg_depthffm_fim.py

Tests cover: collate functions, DataCollatorForMultipleChoice, metric/aggregation
helpers (get_norm, get_train_loss, get_delta_norm, get_norm_updates, get_model_update,
append_delta_norm), decay_learning_rate, get_observed_probability, get_group_cnt,
update_user_groupid_list, and update_global_model (including lora_max_rank validation).
"""
import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Add project root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.engine.ffm_fedavg_depthffm_fim import (
    DataCollatorForMultipleChoice,
    append_delta_norm,
    decay_learning_rate,
    get_delta_norm,
    get_group_cnt,
    get_model_update,
    get_norm,
    get_norm_updates,
    get_observed_probability,
    get_train_loss,
    test_collate_fn,
    update_global_model,
    update_user_groupid_list,
    vit_collate_fn,
)


# -----------------------------------------------------------------------------
# vit_collate_fn
# -----------------------------------------------------------------------------
class TestVitCollateFn(unittest.TestCase):
    def test_output_keys(self):
        examples = [
            (None, 0, torch.randn(3, 224, 224)),
            (None, 1, torch.randn(3, 224, 224)),
        ]
        batch = vit_collate_fn(examples)
        self.assertIn("pixel_values", batch)
        self.assertIn("labels", batch)

    def test_pixel_values_shape(self):
        examples = [
            (None, 0, torch.randn(3, 224, 224)),
            (None, 1, torch.randn(3, 224, 224)),
            (None, 2, torch.randn(3, 224, 224)),
        ]
        batch = vit_collate_fn(examples)
        self.assertEqual(batch["pixel_values"].shape, (3, 3, 224, 224))

    def test_labels_content(self):
        examples = [
            (None, 5, torch.randn(3, 224, 224)),
            (None, 7, torch.randn(3, 224, 224)),
        ]
        batch = vit_collate_fn(examples)
        self.assertEqual(batch["labels"].tolist(), [5, 7])
        self.assertEqual(batch["labels"].dtype, torch.int64)


# -----------------------------------------------------------------------------
# test_collate_fn
# -----------------------------------------------------------------------------
class TestTestCollateFn(unittest.TestCase):
    def test_label_key(self):
        examples = [
            {"pixel_values": torch.randn(3, 224, 224), "label": 0},
            {"pixel_values": torch.randn(3, 224, 224), "label": 1},
        ]
        batch = test_collate_fn(examples)
        self.assertIn("pixel_values", batch)
        self.assertIn("labels", batch)
        self.assertEqual(batch["labels"].tolist(), [0, 1])

    def test_fine_label_key(self):
        examples = [
            {"pixel_values": torch.randn(3, 224, 224), "fine_label": 10},
            {"pixel_values": torch.randn(3, 224, 224), "fine_label": 20},
        ]
        batch = test_collate_fn(examples)
        self.assertEqual(batch["labels"].tolist(), [10, 20])

    def test_pixel_values_shape(self):
        examples = [
            {"pixel_values": torch.randn(3, 224, 224), "label": 0},
            {"pixel_values": torch.randn(3, 224, 224), "label": 1},
        ]
        batch = test_collate_fn(examples)
        self.assertEqual(batch["pixel_values"].shape, (2, 3, 224, 224))


# -----------------------------------------------------------------------------
# DataCollatorForMultipleChoice
# -----------------------------------------------------------------------------
class TestDataCollatorForMultipleChoice(unittest.TestCase):
    def test_call_output_keys(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad.return_value = {
            "input_ids": torch.randint(0, 100, (6, 8)),
            "attention_mask": torch.ones(6, 8),
        }
        collator = DataCollatorForMultipleChoice(tokenizer=mock_tokenizer)
        features = [
            {
                "input_ids": [[1, 2], [3, 4], [5, 6]],
                "attention_mask": [[1, 1], [1, 1], [1, 1]],
                "correct_answer_num": 2,
            },
            {
                "input_ids": [[7, 8], [9, 10], [11, 12]],
                "attention_mask": [[1, 1], [1, 1], [1, 1]],
                "correct_answer_num": 1,
            },
        ]
        batch = collator(features)
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertEqual(batch["labels"].tolist(), [1, 0])

    def test_correct_answer_num_one_indexed(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad.return_value = {
            "input_ids": torch.randint(0, 100, (3, 8)),
            "attention_mask": torch.ones(3, 8),
        }
        collator = DataCollatorForMultipleChoice(tokenizer=mock_tokenizer)
        features = [
            {
                "input_ids": [[1], [2], [3]],
                "attention_mask": [[1], [1], [1]],
                "correct_answer_num": 1,
            },
        ]
        batch = collator(features)
        self.assertEqual(batch["labels"].item(), 0)


# -----------------------------------------------------------------------------
# get_norm
# -----------------------------------------------------------------------------
class TestGetNorm(unittest.TestCase):
    def test_non_empty_returns_median(self):
        delta_norms = [
            torch.tensor(2.0),
            torch.tensor(3.0),
            torch.tensor(1.0),
        ]
        result = get_norm(delta_norms)
        self.assertEqual(result.item(), 2.0)

    def test_empty_returns_100(self):
        result = get_norm([])
        self.assertEqual(result, 100)

    def test_single_element(self):
        result = get_norm([torch.tensor(5.5)])
        self.assertEqual(result.item(), 5.5)


# -----------------------------------------------------------------------------
# get_train_loss
# -----------------------------------------------------------------------------
class TestGetTrainLoss(unittest.TestCase):
    def test_non_empty_returns_mean(self):
        local_losses = [0.4, 0.6, 0.5]
        self.assertAlmostEqual(get_train_loss(local_losses), 0.5)

    def test_empty_returns_100(self):
        self.assertEqual(get_train_loss([]), 100)

    def test_single_loss(self):
        self.assertEqual(get_train_loss([1.5]), 1.5)


# -----------------------------------------------------------------------------
# get_delta_norm, get_norm_updates, append_delta_norm
# -----------------------------------------------------------------------------
class TestDeltaNormHelpers(unittest.TestCase):
    def test_get_norm_updates_flattens_keys(self):
        model_update = {
            "lora_A.0": torch.tensor([1.0, 2.0]),
            "lora_B.0": torch.tensor([3.0]),
        }
        out = get_norm_updates(model_update)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].numel(), 2)
        self.assertEqual(out[1].numel(), 1)

    def test_get_delta_norm_non_empty(self):
        norm_updates = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        ]
        result = get_delta_norm(norm_updates)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.item(), np.sqrt(2.0))

    def test_get_delta_norm_empty(self):
        self.assertIsNone(get_delta_norm([]))

    def test_append_delta_norm_appends_when_non_zero(self):
        delta_norms = []
        norm_updates = [torch.tensor([1.0])]
        append_delta_norm(delta_norms, norm_updates)
        self.assertEqual(len(delta_norms), 1)

    def test_append_delta_norm_does_not_append_when_empty_updates(self):
        delta_norms = []
        append_delta_norm(delta_norms, [])
        self.assertEqual(len(delta_norms), 0)


# -----------------------------------------------------------------------------
# get_model_update
# -----------------------------------------------------------------------------
class TestGetModelUpdate(unittest.TestCase):
    def _make_state(self, *pairs):
        return dict(pairs)

    def test_lora_only_updates_lora_keys(self):
        args = argparse.Namespace(peft="lora", train_classifier=False)
        global_model = self._make_state(
            ("lora_A.0", torch.tensor([1.0])),
            ("base.weight", torch.tensor([10.0])),
        )
        local_model = self._make_state(
            ("lora_A.0", torch.tensor([2.0])),
            ("base.weight", torch.tensor([10.0])),
        )
        no_weight_lora = []
        out = get_model_update(args, global_model, local_model, no_weight_lora)
        self.assertIn("lora_A.0", out)
        self.assertEqual(out["lora_A.0"].item(), 1.0)
        self.assertNotIn("base.weight", out)

    def test_lora_excludes_no_weight_lora_layers(self):
        args = argparse.Namespace(peft="lora", train_classifier=False)
        global_model = self._make_state(
            ("lora_A.layer.0", torch.tensor([1.0])),
        )
        local_model = self._make_state(
            ("lora_A.layer.0", torch.tensor([3.0])),
        )
        # layer 0 in no_weight_lora (extract first number from key)
        no_weight_lora = [0]
        out = get_model_update(args, global_model, local_model, no_weight_lora)
        self.assertNotIn("lora_A.layer.0", out)

    def test_lora_with_train_classifier_updates_classifier(self):
        args = argparse.Namespace(peft="lora", train_classifier=True)
        global_model = self._make_state(
            ("classifier.weight", torch.tensor([1.0, 2.0])),
        )
        local_model = self._make_state(
            ("classifier.weight", torch.tensor([1.5, 2.5])),
        )
        out = get_model_update(args, global_model, local_model, [])
        self.assertIn("classifier.weight", out)
        torch.testing.assert_close(out["classifier.weight"], torch.tensor([0.5, 0.5]))


# -----------------------------------------------------------------------------
# decay_learning_rate
# -----------------------------------------------------------------------------
class TestDecayLearningRate(unittest.TestCase):
    def test_decay_when_condition_met(self):
        # Current implementation: t + (1 % lr_step_size) == 0
        # For lr_step_size=1: 1%1=0, so t+0==0 => t=0 triggers decay
        args = argparse.Namespace(
            lr_step_size=1,
            decay_weight=0.5,
            local_lr=0.01,
        )
        decay_learning_rate(args, 0)
        self.assertEqual(args.local_lr, 0.005)

    def test_no_decay_when_condition_not_met(self):
        args = argparse.Namespace(
            lr_step_size=50,
            decay_weight=0.5,
            local_lr=0.01,
        )
        decay_learning_rate(args, 49)
        self.assertEqual(args.local_lr, 0.01)

    def test_decay_at_negative_t_for_step_50(self):
        # For lr_step_size=50: 1%50=1, so t+1==0 => t=-1
        args = argparse.Namespace(
            lr_step_size=50,
            decay_weight=0.1,
            local_lr=1.0,
        )
        decay_learning_rate(args, -1)
        self.assertEqual(args.local_lr, 0.1)


# -----------------------------------------------------------------------------
# get_observed_probability
# -----------------------------------------------------------------------------
class TestGetObservedProbability(unittest.TestCase):
    def test_normalized_sum_one(self):
        cluster_labels = [0, 1, 2]
        probs = get_observed_probability(cluster_labels)
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_label_mapping(self):
        cluster_labels = [0, 0, 0]
        probs = get_observed_probability(cluster_labels)
        np.testing.assert_array_almost_equal(probs, [1 / 3, 1 / 3, 1 / 3])

    def test_mixed_labels(self):
        cluster_labels = [0, 1, 2, 0]
        probs = get_observed_probability(cluster_labels)
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        for p in probs:
            self.assertGreater(p, 0)
            self.assertLessEqual(p, 1)


# -----------------------------------------------------------------------------
# get_group_cnt
# -----------------------------------------------------------------------------
class TestGetGroupCnt(unittest.TestCase):
    def test_three_equal_groups(self):
        args = argparse.Namespace(
            heterogeneous_group=["1/3", "1/3", "1/3"],
            num_users=9,
        )
        cnt = get_group_cnt(args)
        self.assertEqual(sum(cnt), 9)
        self.assertEqual(cnt, [3, 3, 3])

    def test_remainder_in_last_group(self):
        args = argparse.Namespace(
            heterogeneous_group=["1/3", "1/3", "1/3"],
            num_users=10,
        )
        cnt = get_group_cnt(args)
        self.assertEqual(sum(cnt), 10)
        self.assertEqual(cnt[0], 3)
        self.assertEqual(cnt[1], 3)
        self.assertEqual(cnt[2], 4)

    def test_two_groups(self):
        args = argparse.Namespace(
            heterogeneous_group=["1/2", "1/2"],
            num_users=8,
        )
        cnt = get_group_cnt(args)
        self.assertEqual(cnt, [4, 4])


# -----------------------------------------------------------------------------
# update_user_groupid_list
# -----------------------------------------------------------------------------
class TestUpdateUserGroupidList(unittest.TestCase):
    def test_three_groups_nine_users(self):
        args = argparse.Namespace(
            heterogeneous_group=["1/3", "1/3", "1/3"],
            num_users=9,
        )
        update_user_groupid_list(args)
        self.assertEqual(
            args.user_groupid_list,
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
        )

    def test_two_groups_eight_users(self):
        args = argparse.Namespace(
            heterogeneous_group=["1/2", "1/4", "1/4"],
            num_users=8,
        )
        update_user_groupid_list(args)
        self.assertEqual(len(args.user_groupid_list), 8)
        self.assertEqual(args.user_groupid_list[:4], [0, 0, 0, 0])
        self.assertEqual(args.user_groupid_list[4:6], [1, 1])
        self.assertEqual(args.user_groupid_list[6:], [2, 2])


# -----------------------------------------------------------------------------
# update_global_model
# -----------------------------------------------------------------------------
class TestUpdateGlobalModel(unittest.TestCase):
    def test_lora_max_rank_raises_when_exceeds_full_rank(self):
        args = argparse.Namespace(
            aggregation="average",
            lora_max_rank=100,
        )
        # Minimal global_model: one lora_B key with small hidden size (e.g. rank 8)
        global_model = {
            "lora_B.layer.0": torch.randn(8, 64),
            "lora_A.layer.0": torch.randn(64, 8),
        }
        local_updates = [
            {
                "lora_B.layer.0": torch.zeros(8, 64),
                "lora_A.layer.0": torch.zeros(64, 8),
            },
        ]
        with patch(
            "algorithms.engine.ffm_fedavg_depthffm_fim.average_lora_depthfl",
            side_effect=lambda a, g, u: g,
        ):
            with self.assertRaises(ValueError) as ctx:
                update_global_model(args, global_model, local_updates, [100])
            self.assertIn("lora_max_rank", str(ctx.exception))
            self.assertIn("smaller than the model full rank", str(ctx.exception))

    def test_lora_max_rank_ok_when_less_than_full_rank(self):
        args = argparse.Namespace(
            aggregation="average",
            lora_max_rank=4,
            apply_svd_aggregation=False,
        )
        global_model = {
            "lora_B.layer.0": torch.randn(8, 64),
            "lora_A.layer.0": torch.randn(64, 8),
        }
        local_updates = [
            {
                "lora_B.layer.0": torch.zeros(8, 64),
                "lora_A.layer.0": torch.zeros(64, 8),
            },
        ]
        with patch(
            "algorithms.engine.ffm_fedavg_depthffm_fim.average_lora_depthfl",
            side_effect=lambda a, g, u: g,
        ):
            result = update_global_model(args, global_model, local_updates, [100])
            self.assertIs(result, global_model)


if __name__ == "__main__":
    unittest.main()
