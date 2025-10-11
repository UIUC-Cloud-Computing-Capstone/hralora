"""
Training functionality for the Flower client.

This module contains training-related methods and utilities for the Flower client,
including local training, heterogeneous training, and batch processing.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

import copy
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

from client_constants import (
    CONFIG_KEY_HETEROGENEOUS_GROUP, CONFIG_KEY_USER_GROUPID_LIST, CONFIG_KEY_BLOCK_IDS_LIST,
    CONFIG_KEY_PEFT, CONFIG_KEY_LOGGING_BATCHES, CONFIG_KEY_PIXEL_VALUES, CONFIG_KEY_LABELS,
    DEFAULT_BATCH_FORMAT_3_ELEMENTS, DEFAULT_BATCH_FORMAT_2_ELEMENTS,
    DEFAULT_DIMENSION_1, DEFAULT_ZERO_VALUE, DEFAULT_ONE_VALUE,
    DEFAULT_LORA_PEFT, DEFAULT_LOGGING_BATCHES, DEFAULT_GROUP_ID,
    ERROR_NO_DATA_INDICES, ERROR_INVALID_BATCH_FORMAT, ERROR_NO_TRAINING_DATASET
)


class ClientTrainingMixin:
    """
    Mixin class providing training functionality for the Flower client.
    
    This class contains all training-related methods that can be mixed into
    the main FlowerClient class.
    """
    
    def _train_with_actual_data(self, local_epochs: int, learning_rate: float, server_round: int) -> float:
        """
        Train using actual dataset data with real batch iteration and non-IID distribution.
        
        Args:
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for training
            server_round: Current server round
            
        Returns:
            Total training loss
        """
        # Check if we should use LocalUpdate for heterogeneous training
        if self._should_use_local_update():
            return self._train_with_local_update(local_epochs, learning_rate, server_round)
        else:
            return self._train_with_standard_approach(local_epochs, learning_rate, server_round)
    
    def _should_use_local_update(self) -> bool:
        """Check if we should use LocalUpdate for heterogeneous training."""
        # Use LocalUpdate if we have heterogeneous group configuration
        return (hasattr(self.args, CONFIG_KEY_HETEROGENEOUS_GROUP) and 
                hasattr(self.args, CONFIG_KEY_USER_GROUPID_LIST) and
                hasattr(self.args, CONFIG_KEY_BLOCK_IDS_LIST) and
                self.args.get(CONFIG_KEY_PEFT) == DEFAULT_LORA_PEFT)
    
    def _train_with_local_update(self, local_epochs: int, learning_rate: float, server_round: int) -> float:
        """
        Train using LocalUpdate class for heterogeneous federated learning.
        
        Args:
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for training
            server_round: Current server round
            
        Returns:
            Total training loss
        """
        # Ensure block_ids_list is initialized
        if not hasattr(self.args, CONFIG_KEY_BLOCK_IDS_LIST):
            from algorithms.solver.shared_utils import update_block_ids_list
            update_block_ids_list(self.args)
            logging.info(f"Initialized block_ids_list for client {self.client_id}")
        
        # Prepare training data with reduced batch size if needed
        client_indices_list = self._get_client_data_indices()
        client_dataset = self._create_client_dataset(client_indices_list, self.dataset_train, self.args_loaded)
        dataloader = self._create_training_dataloader(client_dataset)
        
        logging.info(f"Client {self.client_id} using LocalUpdate for heterogeneous training")
        
        # Create LocalUpdate instance
        from algorithms.solver.local_solver import LocalUpdate
        local_solver = LocalUpdate(args=self.args)
        
        # Get client group ID for heterogeneous training
        hete_group_id = self._get_client_group_id()
        
        # Use LocalUpdate for training
        local_model, local_loss, no_weight_lora = local_solver.lora_tuning(
                model=copy.deepcopy(self.model),
                ldr_train=dataloader,
                args=self.args,
                client_index=self.client_id,
                client_real_id=self.client_id,
                round=server_round,
                hete_group_id=hete_group_id
            )
            
        # Update model with trained parameters
        self.model.load_state_dict(local_model)
            
        # Log results
        if local_loss is not None:
                logging.info(f"Client {self.client_id} LocalUpdate training completed: loss={local_loss:.4f}")
                return float(local_loss)  # Ensure float type
        else:
                logging.warning(f"Client {self.client_id} LocalUpdate training returned no loss")
                return 0.0

    def _get_client_group_id(self) -> int:
        """Get the heterogeneous group ID for this client."""
        if hasattr(self.args, CONFIG_KEY_USER_GROUPID_LIST) and self.client_id < len(self.args.user_groupid_list):
            return self.args.user_groupid_list[self.client_id]
        return DEFAULT_GROUP_ID  # Default to group 0

    def _create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        """Create optimizer for training."""
        # Only optimize LoRA parameters if using LoRA
        if self.args.get(CONFIG_KEY_PEFT) == DEFAULT_LORA_PEFT:
            # Get only LoRA parameters
            lora_params = []
            for name, param in self.model.named_parameters():
                if 'lora' in name or 'classifier' in name:
                    lora_params.append(param)
            optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
        else:
            # Optimize all parameters
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        logging.debug(f"Created optimizer with {len(list(optimizer.param_groups[0]['params']))} parameters")
        return optimizer
    
    def _get_client_data_indices(self) -> List[int]:
        """Get and validate client data indices."""
        if hasattr(self, 'client_data_indices'):
            client_indices = self.client_data_indices
        else:
            client_indices = self.dataset_info.get(CONFIG_KEY_CLIENT_DATA_INDICES, set())
        
        if not client_indices:
            raise ValueError(ERROR_NO_DATA_INDICES.format(client_id=self.client_id))
        return list(client_indices)
    
    def _create_training_dataloader(self, client_dataset):
        """Create DataLoader for training using shared utilities."""
        collate_fn = self._get_collate_function(self.args_loaded)
        
        # Use shared create_client_dataloader function
        from algorithms.solver.shared_utils import create_client_dataloader
        return create_client_dataloader(client_dataset, self.args, collate_fn)
    
    def _get_collate_function(self, args_loaded):
        """Get the appropriate collate function based on dataset type."""
        if self._is_cifar100_dataset(args_loaded):
            from algorithms.solver.shared_utils import vit_collate_fn
            return vit_collate_fn
        elif self._is_ledgar_dataset(args_loaded):
            return getattr(args_loaded, CONFIG_KEY_DATA_COLLATOR, None)
        else:
            raise ValueError(f"Invalid dataset: {args_loaded.dataset}")
    
    def _is_cifar100_dataset(self, args_loaded) -> bool:
        """Check if the dataset is CIFAR-100."""
        from client_constants import DEFAULT_CIFAR100_DATASET
        return (hasattr(args_loaded, 'dataset') and 
                args_loaded.dataset == DEFAULT_CIFAR100_DATASET)
    
    def _is_ledgar_dataset(self, args_loaded) -> bool:
        """Check if the dataset is LEDGAR."""
        from client_constants import DEFAULT_LEDGAR_DATASET
        return (hasattr(args_loaded, 'dataset') and 
                DEFAULT_LEDGAR_DATASET in args_loaded.dataset)
    
    def _process_training_batch_with_gradients(self, batch, batch_idx: int, epoch: int, optimizer: torch.optim.Optimizer) -> float:
        """Process a single training batch with actual gradients."""
        # Extract batch data
        pixel_values, labels = self._extract_batch_data(batch)
        
        # Move to device
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        batch_loss = loss.item()
        
        # Log progress for first few batches
        if batch_idx < self.args.get(CONFIG_KEY_LOGGING_BATCHES, DEFAULT_LOGGING_BATCHES):
            logging.debug(f"Client {self.client_id} epoch {epoch + 1}, batch {batch_idx + 1}: loss={batch_loss:.4f}")
        
        return batch_loss
    
    def _extract_batch_data(self, batch) -> Tuple:
        """Extract pixel_values and labels from batch."""
        if isinstance(batch, dict):
            return self._extract_from_dict_batch(batch)
        elif len(batch) == DEFAULT_BATCH_FORMAT_3_ELEMENTS:
            return self._extract_from_three_element_batch(batch)
        elif len(batch) == DEFAULT_BATCH_FORMAT_2_ELEMENTS:
            return self._extract_from_two_element_batch(batch)
        else:
            raise ValueError(ERROR_INVALID_BATCH_FORMAT.format(batch_type=type(batch)))
    
    def _extract_from_dict_batch(self, batch: Dict) -> Tuple:
        """Extract data from dictionary format batch."""
        return batch[CONFIG_KEY_PIXEL_VALUES], batch[CONFIG_KEY_LABELS]
    
    def _extract_from_three_element_batch(self, batch) -> Tuple:
        """Extract data from three-element batch (image, label, pixel_values)."""
        image, label, pixel_values = batch
        return pixel_values, label
    
    def _extract_from_two_element_batch(self, batch) -> Tuple:
        """Extract data from two-element batch (pixel_values, labels)."""
        return batch[DEFAULT_ZERO_VALUE], batch[DEFAULT_ONE_VALUE]

    def _create_client_dataset(self, client_indices: List[int], dataset_train, args_loaded):
        """
        Create a client-specific dataset subset using shared utilities.

        Args:
            client_indices: List of indices for this client's data
            dataset_train: Training dataset
            args_loaded: Loaded arguments

        Returns:
            Dataset subset for this client
        """
        # Use shared create_client_dataset function
        from algorithms.solver.shared_utils import create_client_dataset
        client_dataset = create_client_dataset(dataset_train, client_indices, args_loaded)

        logging.debug(f"Client {self.client_id} created dataset subset with {len(client_dataset)} samples")
        return client_dataset

    def _log_training_results(self, client_indices_list: List[int], total_loss: float, local_epochs: int) -> None:
        """Log final training results."""
        from client_constants import DEFAULT_ZERO_VALUE
        avg_total_loss = total_loss / local_epochs if local_epochs > DEFAULT_ZERO_VALUE else DEFAULT_ZERO_VALUE
        logging.info(f"Client {self.client_id} trained on {len(client_indices_list)} actual samples, "
                     f"avg_loss={avg_total_loss:.4f}")
