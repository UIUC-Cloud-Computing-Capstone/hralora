"""
Unit tests for algorithms/engine/ffm_fedavg_depthfl.py

Tests cover: vit_collate_fn (batch shape and keys), and ffm_fedavg_depthfl
return shape and metric_keys contract when run with mocked data/model/training.
"""
import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add project root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.engine import ffm_fedavg_depthfl as depthfl_module


# -----------------------------------------------------------------------------
# vit_collate_fn
# -----------------------------------------------------------------------------
class TestVitCollateFn(unittest.TestCase):
    def test_output_keys(self):
        examples = [
            (None, 0, torch.randn(3, 224, 224)),
            (None, 1, torch.randn(3, 224, 224)),
        ]
        batch = depthfl_module.vit_collate_fn(examples)
        self.assertIn("pixel_values", batch)
        self.assertIn("labels", batch)

    def test_pixel_values_shape(self):
        examples = [
            (None, 0, torch.randn(3, 224, 224)),
            (None, 1, torch.randn(3, 224, 224)),
            (None, 2, torch.randn(3, 224, 224)),
        ]
        batch = depthfl_module.vit_collate_fn(examples)
        self.assertEqual(batch["pixel_values"].shape, (3, 3, 224, 224))

    def test_labels_content_and_dtype(self):
        examples = [
            (None, 5, torch.randn(3, 224, 224)),
            (None, 7, torch.randn(3, 224, 224)),
        ]
        batch = depthfl_module.vit_collate_fn(examples)
        self.assertEqual(batch["labels"].tolist(), [5, 7])
        self.assertEqual(batch["labels"].dtype, torch.int64)

    def test_single_example(self):
        examples = [(None, 2, torch.randn(3, 224, 224))]
        batch = depthfl_module.vit_collate_fn(examples)
        self.assertEqual(batch["pixel_values"].shape, (1, 3, 224, 224))
        self.assertEqual(batch["labels"].tolist(), [2])


# -----------------------------------------------------------------------------
# VISION_MODEL constant
# -----------------------------------------------------------------------------
class TestVisionModelConstant(unittest.TestCase):
    def test_vision_model_defined(self):
        self.assertTrue(hasattr(depthfl_module, "VISION_MODEL"))
        self.assertIsInstance(depthfl_module.VISION_MODEL, str)
        self.assertIn("deit", depthfl_module.VISION_MODEL.lower())


# -----------------------------------------------------------------------------
# ffm_fedavg_depthfl (mocked integration)
# -----------------------------------------------------------------------------
class TestFfmFedavgDepthfl(unittest.TestCase):
    """Test ffm_fedavg_depthfl return shape and metric_keys with mocked deps."""

    def _make_args(self):
        args = argparse.Namespace()
        args.num_users = 4
        args.num_selected_users = 2
        args.round = 1
        args.heterogeneous_group = ["1/2", "1/2"]
        args.heterogeneous_group0_lora = 2
        args.heterogeneous_group1_lora = 2
        args.lora_layer = 4
        args.peft = "lora"
        args.LOKR = False
        args.LEGEND = False
        args.HetLoRA = False
        args.FlexLoRA = False
        args.lr_step_size = 1
        args.decay_weight = 0.99
        args.local_lr = 0.01
        args.model = "facebook/deit-small-patch16-224"
        args.dataset = "cifar100"
        args.batch_size = 4
        args.log_path = "/tmp/test_depthfl_log"
        args.logger = MagicMock()
        args.accelerator = MagicMock()
        args.accelerator.is_local_main_process = True
        args.accelerator.wait_for_everyone = MagicMock()
        return args

    @patch("algorithms.engine.ffm_fedavg_depthfl.test_vit")
    @patch("algorithms.engine.ffm_fedavg_depthfl.model_setup")
    @patch("algorithms.engine.ffm_fedavg_depthfl.load_partition")
    def test_returns_tuple_and_metric_keys(
        self, mock_load_partition, mock_model_setup, mock_test_vit
    ):
        args = self._make_args()
        # Minimal partition: 4 users, small train/test
        dict_users = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
        mock_train = MagicMock(__len__=lambda self: 8)
        mock_test = MagicMock(__len__=lambda self: 4)
        mock_val = MagicMock(__len__=lambda self: 0)
        mock_public = MagicMock(__len__=lambda self: 0)
        mock_fim = MagicMock(__len__=lambda self: 0)
        mock_load_partition.return_value = (
            args,
            mock_train,
            mock_test,
            mock_val,
            mock_public,
            dict_users,
            mock_fim,
        )

        # Minimal model: state_dict with one lora key so updates work
        mock_net = MagicMock()
        mock_net.train = MagicMock()
        mock_net.eval = MagicMock()
        mock_net.load_state_dict = MagicMock()
        mock_state = {
            "lora_A.0": torch.randn(8, 4),
            "lora_B.0": torch.randn(4, 8),
        }
        mock_model_setup.return_value = (args, mock_net, mock_state, 384)

        mock_test_vit.return_value = (0.5, 0.5)

        with patch(
            "algorithms.engine.ffm_fedavg_depthfl.SummaryWriter", MagicMock()
        ), patch(
            "algorithms.engine.ffm_fedavg_depthfl.DataLoader"
        ) as mock_dl_cls:
            mock_dl = MagicMock()
            mock_dl.__len__ = lambda self: 2
            mock_dl_cls.return_value = mock_dl

            with patch(
                "algorithms.engine.ffm_fedavg_depthfl.LocalUpdate"
            ) as mock_local_cls:
                mock_solver = MagicMock()
                local_state = {
                    "lora_A.0": torch.randn(8, 4) + 0.1,
                    "lora_B.0": torch.randn(4, 8) + 0.1,
                }
                mock_solver.lora_tuning.return_value = (
                    local_state,
                    0.5,
                    [],  # no_weight_lora
                )
                mock_local_cls.return_value = mock_solver

                with patch(
                    "algorithms.engine.ffm_fedavg_depthfl.np.random.choice",
                    return_value=[0, 1],
                ):
                    result = depthfl_module.ffm_fedavg_depthfl(args)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        metrics_tuple, metric_keys = result
        self.assertIsInstance(metrics_tuple, tuple)
        self.assertEqual(
            metrics_tuple,
            (0.5, 0.0, 0.0, 0.0),
            "best_test_acc from test_vit, others 0",
        )
        self.assertIsInstance(metric_keys, dict)
        self.assertIn("Accuracy", metric_keys)
        self.assertIn("F1", metric_keys)
        self.assertIn("Macro_F1", metric_keys)
        self.assertIn("Micro_F1", metric_keys)


if __name__ == "__main__":
    unittest.main()
