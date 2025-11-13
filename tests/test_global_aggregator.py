import unittest
import torch
import numpy as np
from unittest.mock import Mock
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.solver.global_aggregator import weighted_average_lora_depthfl


class TestWeightedAverageLoraDepthFL(unittest.TestCase):
    """Test cases for weighted_average_lora_depthfl with heterogeneous ranks"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock args object
        self.args = Mock()
        self.args.only_train_b = False
        
        # Create a global model with LoRA parameters
        # Simulating a model with 2 layers, each with lora_A and lora_B
        # Full rank is 12 (max rank)
        self.global_model = {
            'base_model.layer.0.attention.lora_A': torch.zeros(12, 384),  # rank x hidden_dim
            'base_model.layer.0.attention.lora_B': torch.zeros(384, 12),  # hidden_dim x rank
            'base_model.layer.1.attention.lora_A': torch.zeros(12, 384),
            'base_model.layer.1.attention.lora_B': torch.zeros(384, 12),
            'classifier.weight': torch.zeros(100, 384),  # num_classes x hidden_dim
            'base_model.embedding.weight': torch.zeros(1000, 384),  # Should be ignored
        }
        
    def test_heterogeneous_ranks_same_layer(self):
        """Test aggregation with different ranks for the same layer.
        
        Note: The current implementation requires all updates for the same key
        to have the same shape. For true heterogeneous rank support, updates should
        be padded/truncated to match the global model shape (full rank) before aggregation.
        """
        # Simulate heterogeneous ranks by padding smaller ranks to full rank
        # Client 0: rank 4 (padded to 12)
        # Client 1: rank 6 (padded to 12)
        # Client 2: rank 8 (padded to 12)
        full_rank = 12
        
        def pad_lora_A(tensor, target_rank):
            """Pad lora_A tensor to target rank"""
            if tensor.shape[0] < target_rank:
                padding = torch.zeros(target_rank - tensor.shape[0], tensor.shape[1])
                return torch.cat([tensor, padding], dim=0)
            return tensor[:target_rank]
        
        def pad_lora_B(tensor, target_rank):
            """Pad lora_B tensor to target rank"""
            if tensor.shape[1] < target_rank:
                padding = torch.zeros(tensor.shape[0], target_rank - tensor.shape[1])
                return torch.cat([tensor, padding], dim=1)
            return tensor[:, :target_rank]
        
        loc_updates = [
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(4, 384) * 1.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, 4) * 1.0, full_rank),
            },
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(6, 384) * 2.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, 6) * 2.0, full_rank),
            },
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(8, 384) * 3.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, 8) * 3.0, full_rank),
            },
        ]
        
        # Different sample counts for weighting
        num_samples = [10, 20, 30]  # Total: 60
        
        # Make a copy to avoid modifying the original
        global_model = {k: v.clone() for k, v in self.global_model.items()}
        
        result = weighted_average_lora_depthfl(self.args, global_model, loc_updates, num_samples)
        
        # Check that lora_A was aggregated correctly
        self.assertIn('base_model.layer.0.attention.lora_A', result)
        self.assertIn('base_model.layer.0.attention.lora_B', result)
        
        # Verify the shape matches the global model (full rank)
        self.assertEqual(result['base_model.layer.0.attention.lora_A'].shape, (12, 384))
        self.assertEqual(result['base_model.layer.0.attention.lora_B'].shape, (384, 12))
        
        # Verify weighted aggregation worked
        # Expected: (1*10 + 2*20 + 3*30) / 60 = 140/60 ≈ 2.333
        expected_weight = (1.0 * 10 + 2.0 * 20 + 3.0 * 30) / 60
        # Check first few rows (where all clients have non-zero values)
        np.testing.assert_allclose(
            result['base_model.layer.0.attention.lora_A'][:4, 0].numpy(),
            expected_weight,
            rtol=1e-5
        )
        
    def test_heterogeneous_ranks_different_layers(self):
        """Test aggregation where different clients have different layers.
        
        All updates must be padded to full rank shape for the current implementation.
        """
        full_rank = 12
        
        def pad_lora_A(tensor, target_rank):
            if tensor.shape[0] < target_rank:
                padding = torch.zeros(target_rank - tensor.shape[0], tensor.shape[1])
                return torch.cat([tensor, padding], dim=0)
            return tensor[:target_rank]
        
        def pad_lora_B(tensor, target_rank):
            if tensor.shape[1] < target_rank:
                padding = torch.zeros(tensor.shape[0], target_rank - tensor.shape[1])
                return torch.cat([tensor, padding], dim=1)
            return tensor[:, :target_rank]
        
        # Client 0: only layer 0 with rank 4 (padded to 12)
        # Client 1: only layer 1 with rank 6 (padded to 12)
        # Client 2: both layers with rank 8 (padded to 12)
        loc_updates = [
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(4, 384) * 1.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, 4) * 1.0, full_rank),
            },
            {
                'base_model.layer.1.attention.lora_A': pad_lora_A(torch.ones(6, 384) * 2.0, full_rank),
                'base_model.layer.1.attention.lora_B': pad_lora_B(torch.ones(384, 6) * 2.0, full_rank),
            },
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(8, 384) * 3.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, 8) * 3.0, full_rank),
                'base_model.layer.1.attention.lora_A': pad_lora_A(torch.ones(8, 384) * 3.0, full_rank),
                'base_model.layer.1.attention.lora_B': pad_lora_B(torch.ones(384, 8) * 3.0, full_rank),
            },
        ]
        
        num_samples = [15, 25, 35]
        
        global_model = {k: v.clone() for k, v in self.global_model.items()}
        result = weighted_average_lora_depthfl(self.args, global_model, loc_updates, num_samples)
        
        # Both layers should be present
        self.assertIn('base_model.layer.0.attention.lora_A', result)
        self.assertIn('base_model.layer.1.attention.lora_A', result)
        
        # Layer 0 should have updates from clients 0 and 2
        # Layer 1 should have updates from clients 1 and 2
        self.assertEqual(result['base_model.layer.0.attention.lora_A'].shape, (12, 384))
        self.assertEqual(result['base_model.layer.1.attention.lora_A'].shape, (12, 384))
        
        # Layer 0: weighted avg of client 0 (1.0) and client 2 (3.0)
        # Weights: 15/(15+35)=0.3, 35/(15+35)=0.7
        # Expected: 1.0*0.3 + 3.0*0.7 = 2.4
        expected_layer0 = (1.0 * 15 + 3.0 * 35) / (15 + 35)
        np.testing.assert_allclose(
            result['base_model.layer.0.attention.lora_A'][:4, 0].numpy(),
            expected_layer0,
            rtol=1e-5
        )
        
    def test_weighted_average_correctness(self):
        """Test that weighting by num_samples works correctly"""
        # Use same rank for all clients, but pad to full rank
        rank = 6
        full_rank = 12
        
        def pad_lora_A(tensor, target_rank):
            if tensor.shape[0] < target_rank:
                padding = torch.zeros(target_rank - tensor.shape[0], tensor.shape[1])
                return torch.cat([tensor, padding], dim=0)
            return tensor[:target_rank]
        
        def pad_lora_B(tensor, target_rank):
            if tensor.shape[1] < target_rank:
                padding = torch.zeros(tensor.shape[0], target_rank - tensor.shape[1])
                return torch.cat([tensor, padding], dim=1)
            return tensor[:, :target_rank]
        
        loc_updates = [
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(rank, 384) * 1.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, rank) * 1.0, full_rank),
            },
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(rank, 384) * 2.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, rank) * 2.0, full_rank),
            },
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(rank, 384) * 3.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, rank) * 3.0, full_rank),
            },
        ]
        
        # Weights: 10, 20, 30 (total: 60)
        # Expected weighted average for lora_A: (1*10 + 2*20 + 3*30) / 60 = 140/60 ≈ 2.333
        num_samples = [10, 20, 30]
        
        global_model = {k: v.clone() for k, v in self.global_model.items()}
        result = weighted_average_lora_depthfl(self.args, global_model, loc_updates, num_samples)
        
        # Check that the result has the correct shape (full rank)
        self.assertEqual(result['base_model.layer.0.attention.lora_A'].shape, (12, 384))
        
        # For same-rank updates, check that first 'rank' rows are updated
        # Note: The function sums updates weighted by normalized sample counts
        lora_A_result = result['base_model.layer.0.attention.lora_A']
        expected_weight = (1.0 * 10 + 2.0 * 20 + 3.0 * 30) / 60
        
        # Check first 'rank' rows (where updates were applied)
        # Allow some tolerance for floating point
        np.testing.assert_allclose(
            lora_A_result[:rank, 0].numpy(),
            expected_weight,
            rtol=1e-5
        )
        
    def test_classifier_aggregation(self):
        """Test that classifier weights are also aggregated"""
        loc_updates = [
            {
                'classifier.weight': torch.ones(100, 384) * 1.0,
            },
            {
                'classifier.weight': torch.ones(100, 384) * 2.0,
            },
        ]
        
        num_samples = [20, 30]
        
        global_model = {k: v.clone() for k, v in self.global_model.items()}
        result = weighted_average_lora_depthfl(self.args, global_model, loc_updates, num_samples)
        
        # Classifier should be updated
        self.assertIn('classifier.weight', result)
        
        # Expected weighted average: (1*20 + 2*30) / 50 = 80/50 = 1.6
        expected_weight = (1.0 * 20 + 2.0 * 30) / 50
        np.testing.assert_allclose(
            result['classifier.weight'].numpy(),
            expected_weight,
            rtol=1e-5
        )
        
    def test_only_train_b_flag(self):
        """Test that only_train_b flag filters to lora_B only"""
        self.args.only_train_b = True
        
        full_rank = 12
        
        def pad_lora_B(tensor, target_rank):
            if tensor.shape[1] < target_rank:
                padding = torch.zeros(tensor.shape[0], target_rank - tensor.shape[1])
                return torch.cat([tensor, padding], dim=1)
            return tensor[:, :target_rank]
        
        loc_updates = [
            {
                'base_model.layer.0.attention.lora_A': torch.ones(4, 384) * 1.0,  # Should be ignored
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, 4) * 1.0, full_rank),
            },
            {
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, 6) * 2.0, full_rank),
            },
        ]
        
        num_samples = [10, 20]
        
        global_model = {k: v.clone() for k, v in self.global_model.items()}
        original_lora_A = global_model['base_model.layer.0.attention.lora_A'].clone()
        
        result = weighted_average_lora_depthfl(self.args, global_model, loc_updates, num_samples)
        
        # lora_B should be updated
        self.assertIn('base_model.layer.0.attention.lora_B', result)
        
        # lora_A should NOT be updated (only lora_B when only_train_b=True)
        torch.testing.assert_close(result['base_model.layer.0.attention.lora_A'], original_lora_A)
        
        # Verify lora_B weighted average: (1*10 + 2*20) / 30 = 50/30 ≈ 1.667
        expected_weight = (1.0 * 10 + 2.0 * 20) / 30
        np.testing.assert_allclose(
            result['base_model.layer.0.attention.lora_B'][0, :4].numpy(),
            expected_weight,
            rtol=1e-5
        )
        
    def test_empty_updates(self):
        """Test handling of empty updates"""
        loc_updates = []
        num_samples = []
        
        global_model = {k: v.clone() for k, v in self.global_model.items()}
        result = weighted_average_lora_depthfl(self.args, global_model, loc_updates, num_samples)
        
        # Should return global_model unchanged
        for k in global_model.keys():
            if 'lora' in k or 'classifier' in k:
                # These should remain unchanged
                torch.testing.assert_close(result[k], global_model[k])
                
    def test_missing_keys_in_updates(self):
        """Test that missing keys in some updates are handled gracefully"""
        full_rank = 12
        
        def pad_lora_A(tensor, target_rank):
            if tensor.shape[0] < target_rank:
                padding = torch.zeros(target_rank - tensor.shape[0], tensor.shape[1])
                return torch.cat([tensor, padding], dim=0)
            return tensor[:target_rank]
        
        def pad_lora_B(tensor, target_rank):
            if tensor.shape[1] < target_rank:
                padding = torch.zeros(tensor.shape[0], target_rank - tensor.shape[1])
                return torch.cat([tensor, padding], dim=1)
            return tensor[:, :target_rank]
        
        loc_updates = [
            {
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(4, 384) * 1.0, full_rank),
                'base_model.layer.0.attention.lora_B': pad_lora_B(torch.ones(384, 4) * 1.0, full_rank),
            },
            {
                # Missing layer 0, but has layer 1
                'base_model.layer.1.attention.lora_A': pad_lora_A(torch.ones(6, 384) * 2.0, full_rank),
                'base_model.layer.1.attention.lora_B': pad_lora_B(torch.ones(384, 6) * 2.0, full_rank),
            },
        ]
        
        num_samples = [15, 25]
        
        global_model = {k: v.clone() for k, v in self.global_model.items()}
        result = weighted_average_lora_depthfl(self.args, global_model, loc_updates, num_samples)
        
        # Layer 0 should only have update from client 0
        # Layer 1 should only have update from client 1
        self.assertIn('base_model.layer.0.attention.lora_A', result)
        self.assertIn('base_model.layer.1.attention.lora_A', result)
        
        # Layer 0 should have value 1.0 (only from client 0)
        np.testing.assert_allclose(
            result['base_model.layer.0.attention.lora_A'][:4, 0].numpy(),
            1.0,
            rtol=1e-5
        )
        
        # Layer 1 should have value 2.0 (only from client 1)
        np.testing.assert_allclose(
            result['base_model.layer.1.attention.lora_A'][:6, 0].numpy(),
            2.0,
            rtol=1e-5
        )
        
    def test_non_lora_keys_ignored(self):
        """Test that non-LoRA keys are ignored"""
        full_rank = 12
        
        def pad_lora_A(tensor, target_rank):
            if tensor.shape[0] < target_rank:
                padding = torch.zeros(target_rank - tensor.shape[0], tensor.shape[1])
                return torch.cat([tensor, padding], dim=0)
            return tensor[:target_rank]
        
        loc_updates = [
            {
                'base_model.embedding.weight': torch.ones(1000, 384) * 100.0,  # Should be ignored
                'base_model.layer.0.attention.lora_A': pad_lora_A(torch.ones(4, 384) * 1.0, full_rank),
            },
        ]
        
        num_samples = [10]
        
        global_model = {k: v.clone() for k, v in self.global_model.items()}
        original_embedding = global_model['base_model.embedding.weight'].clone()
        
        result = weighted_average_lora_depthfl(self.args, global_model, loc_updates, num_samples)
        
        # Embedding should remain unchanged (not aggregated)
        torch.testing.assert_close(result['base_model.embedding.weight'], original_embedding)
        
        # But lora_A should be updated
        self.assertIn('base_model.layer.0.attention.lora_A', result)
        np.testing.assert_allclose(
            result['base_model.layer.0.attention.lora_A'][:4, 0].numpy(),
            1.0,
            rtol=1e-5
        )


if __name__ == '__main__':
    unittest.main()

