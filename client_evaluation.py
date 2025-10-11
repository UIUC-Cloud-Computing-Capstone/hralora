"""
Evaluation functionality for the Flower client.

This module contains evaluation-related methods and utilities for the Flower client,
including test data evaluation, batch processing, and metrics computation.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

import logging
import torch
from typing import Dict, List, Tuple, Any

from client_constants import (
    CONFIG_KEY_SHUFFLE_EVAL, CONFIG_KEY_DROP_LAST_EVAL, CONFIG_KEY_NUM_WORKERS,
    CONFIG_KEY_EVAL_BATCHES, CONFIG_KEY_TOTAL_LOSS, CONFIG_KEY_TOTAL_CORRECT,
    CONFIG_KEY_TOTAL_SAMPLES, CONFIG_KEY_NUM_BATCHES,
    DEFAULT_BATCH_FORMAT_3_ELEMENTS, DEFAULT_BATCH_FORMAT_2_ELEMENTS,
    DEFAULT_DIMENSION_1, DEFAULT_ZERO_VALUE, DEFAULT_ONE_VALUE,
    DEFAULT_SHUFFLE_EVAL, DEFAULT_DROP_LAST_EVAL, DEFAULT_NUM_WORKERS,
    DEFAULT_EVAL_BATCHES,
    ERROR_NO_TEST_DATASET
)


class ClientEvaluationMixin:
    """
    Mixin class providing evaluation functionality for the Flower client.
    
    This class contains all evaluation-related methods that can be mixed into
    the main FlowerClient class.
    """
    
    def _evaluate_with_actual_data(self, server_round: int) -> Tuple[float, float]:
        """
        Evaluate using actual dataset data with real batch iteration.

        Args:
            server_round: Current server round

        Returns:
            Tuple of (accuracy, loss)
        """
        # Prepare evaluation dataset
        eval_dataset = self._get_evaluation_dataset()
        if eval_dataset is None:
            raise ValueError("No test dataset available for evaluation")

        # Create evaluation DataLoader
        eval_dataloader = self._create_evaluation_dataloader(eval_dataset)
        
        # Perform evaluation
        metrics = self._perform_evaluation(eval_dataloader, server_round)
        
        return metrics
    
    def _get_evaluation_dataset(self):
        """Get evaluation dataset (test only)."""
        return self.dataset_test
    
    def _create_evaluation_dataloader(self, eval_dataset):
        """Create DataLoader for evaluation."""
        from torch.utils.data import DataLoader
        batch_size = len(eval_dataset)  # Use full dataset for evaluation
        collate_fn = self._get_collate_function(self.args_loaded)
        
        return DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=self.args.get(CONFIG_KEY_SHUFFLE_EVAL, DEFAULT_SHUFFLE_EVAL),
            drop_last=self.args.get(CONFIG_KEY_DROP_LAST_EVAL, DEFAULT_DROP_LAST_EVAL),
            num_workers=self.args.get(CONFIG_KEY_NUM_WORKERS, DEFAULT_NUM_WORKERS),
            collate_fn=collate_fn
        )
    
    def _perform_evaluation(self, eval_dataloader, server_round: int) -> Tuple[float, float]:
        """Perform evaluation on all batches."""
        metrics = {
            CONFIG_KEY_TOTAL_LOSS: DEFAULT_ZERO_VALUE, 
            CONFIG_KEY_TOTAL_CORRECT: DEFAULT_ZERO_VALUE, 
            CONFIG_KEY_TOTAL_SAMPLES: DEFAULT_ZERO_VALUE, 
            CONFIG_KEY_NUM_BATCHES: DEFAULT_ZERO_VALUE
        }
        
        logging.info(f"Client {self.client_id} evaluating on {len(eval_dataloader)} test batches")

        for batch_idx, batch in enumerate(eval_dataloader):
            self._process_evaluation_batch(batch, server_round, batch_idx, metrics)

        return self._compute_overall_evaluation_metrics(
            metrics[CONFIG_KEY_TOTAL_LOSS], metrics[CONFIG_KEY_TOTAL_CORRECT], 
            metrics[CONFIG_KEY_TOTAL_SAMPLES], metrics[CONFIG_KEY_NUM_BATCHES]
        )
    
    def _process_evaluation_batch(self, batch, server_round: int, batch_idx: int, metrics: Dict) -> None:
        """Process a single evaluation batch."""
        pixel_values, label = self._extract_evaluation_batch_data(batch)
        batch_loss, batch_correct, batch_size = self._compute_batch_evaluation_metrics(
            pixel_values, label, server_round, batch_idx
        )

        metrics[CONFIG_KEY_TOTAL_LOSS] += batch_loss
        metrics[CONFIG_KEY_TOTAL_CORRECT] += batch_correct
        metrics[CONFIG_KEY_TOTAL_SAMPLES] += batch_size
        metrics[CONFIG_KEY_NUM_BATCHES] += 1

        # Log progress for first few batches
        if batch_idx < self.args.get(CONFIG_KEY_EVAL_BATCHES, DEFAULT_EVAL_BATCHES):
            batch_accuracy = batch_correct / batch_size if batch_size > DEFAULT_ZERO_VALUE else DEFAULT_ZERO_VALUE
            logging.debug(f"Client {self.client_id} eval batch {batch_idx + DEFAULT_ONE_VALUE}: "
                          f"loss={batch_loss:.4f}, acc={batch_accuracy:.4f}")
    
    def _extract_evaluation_batch_data(self, batch) -> Tuple:
        """Extract batch data for evaluation."""
        if len(batch) == DEFAULT_BATCH_FORMAT_3_ELEMENTS:  # DatasetSplit format
            image, label, pixel_values = batch
            return pixel_values, label
        elif len(batch) == DEFAULT_BATCH_FORMAT_2_ELEMENTS:  # Standard format
            return batch[DEFAULT_ZERO_VALUE], batch[DEFAULT_ONE_VALUE]
        else:
            raise ValueError(f"Invalid batch length for evaluation: {len(batch)}")
    
    def _compute_overall_evaluation_metrics(self, total_loss: float, total_correct: int, 
                                          total_samples: int, num_batches: int) -> Tuple[float, float]:
        """Compute overall evaluation metrics."""
        if num_batches > DEFAULT_ZERO_VALUE and total_samples > DEFAULT_ZERO_VALUE:
            avg_loss = total_loss / num_batches
            accuracy = total_correct / total_samples

            logging.info(f"Client {self.client_id} evaluation completed: "
                         f"loss={avg_loss:.4f}, accuracy={accuracy:.4f} "
                         f"({total_samples} samples, {num_batches} batches)")

            return accuracy, avg_loss
        else:
            raise ValueError("No batches processed during evaluation")

    def _compute_batch_evaluation_metrics(self, pixel_values, labels, server_round: int, batch_idx: int) -> Tuple[float, int, int]:
        """
        Compute actual evaluation metrics for a batch of data.

        Args:
            pixel_values: Batch of image data (tensor or None)
            labels: Batch of labels (tensor or None)
            server_round: Current server round
            batch_idx: Batch index within evaluation

        Returns:
            Tuple of (batch_loss, num_correct, batch_size)
        """
        if pixel_values is None or labels is None:
            raise ValueError(f"Invalid pixel_values or labels: {pixel_values}, {labels}")
        
        batch_size = pixel_values.shape[0] if hasattr(pixel_values, 'shape') else labels.shape[0]
        
        # Move to device
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=DEFAULT_DIMENSION_1)
            num_correct = int(torch.sum(predictions == labels).item())

        logging.debug(f"Eval batch {batch_idx}: size={batch_size}, "
                      f"loss={loss:.4f}, correct={num_correct}")

        return float(loss.item()), num_correct, batch_size

    def _perform_evaluation_with_validation(self, server_round: int) -> Tuple[float, float, int]:
        """Perform evaluation with proper validation."""
        from client_constants import CONFIG_KEY_DATA_LOADED, CONFIG_KEY_TEST_SAMPLES
        if not (self.dataset_info.get(CONFIG_KEY_DATA_LOADED, False) and self.dataset_test is not None):
            raise ValueError(ERROR_NO_TEST_DATASET)
        
        accuracy, loss = self._evaluate_with_actual_data(server_round)
        num_examples = self.dataset_info.get(CONFIG_KEY_TEST_SAMPLES)
        
        if num_examples is None:
            raise ValueError("test_samples not available in dataset_info")
        
        logging.info(f"Client {self.client_id} evaluated with actual test dataset: {num_examples} samples")
        return accuracy, loss, num_examples
