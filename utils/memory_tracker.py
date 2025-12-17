"""
Memory tracking utilities for PyTorch models.
Supports both GPU and CPU memory profiling including:
- Parameter memory
- Activation memory
- Optimizer state memory
"""

import os
import torch
from typing import Dict, Optional, Union, Tuple
from torch.profiler import profile, ProfilerActivity, record_function

# Constants
MB_TO_BYTES = 1024 * 1024
GB_TO_BYTES = 1024 * 1024 * 1024
FP32_BYTES_PER_PARAM = 4
FP16_BYTES_PER_PARAM = 2
OPTIMIZER_STATE_BYTES_PER_PARAM = 4  # Optimizer states are typically fp32


class MemoryTracker:
    """
    Track memory usage for PyTorch models including parameters, activations, and optimizer states.
    Supports both GPU and CPU.
    """
    
    def __init__(self):
        """Initialize memory tracker. Device is auto-detected from model in each method."""
        pass
    
    @staticmethod
    def _get_device_info(model: torch.nn.Module) -> Tuple[torch.device, bool]:
        """Get device and CUDA status from model."""
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        is_cuda = device.type == 'cuda'
        return device, is_cuda
    
    @staticmethod
    def _get_cpu_process() -> 'psutil.Process':
        """Get current process for CPU memory tracking."""
        import psutil
        return psutil.Process(os.getpid())
    
    @staticmethod
    def _compute_loss(outputs, batch: Union[Dict, torch.Tensor], loss_fn) -> torch.Tensor:
        """Extract or compute loss from model outputs."""
        if hasattr(outputs, 'loss'):
            return outputs.loss
        elif isinstance(outputs, torch.Tensor):
            labels = batch.get('labels') if isinstance(batch, dict) else None
            return loss_fn(outputs, labels) if labels is not None else outputs
        else:
            return outputs
    
    @staticmethod
    def _bytes_to_mb(bytes_value: Union[int, float]) -> float:
        """Convert bytes to megabytes."""
        return bytes_value / MB_TO_BYTES
    
    @staticmethod
    def _bytes_to_gb(bytes_value: Union[int, float]) -> float:
        """Convert bytes to gigabytes."""
        return bytes_value / GB_TO_BYTES
    
    def get_parameter_memory(self, model: torch.nn.Module, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
        """
        Calculate parameter memory usage.
        
        Args:
            model: PyTorch model
            precision: 'fp32' or 'fp16'
        
        Returns:
            Dictionary with parameter counts and memory in MB
        """
        bytes_per_param = FP32_BYTES_PER_PARAM if precision == 'fp32' else FP16_BYTES_PER_PARAM
        
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        
        param_memory_bytes = total_params * bytes_per_param
        trainable_memory_bytes = trainable_params * bytes_per_param
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_memory_MB': self._bytes_to_mb(param_memory_bytes),
            'trainable_memory_MB': self._bytes_to_mb(trainable_memory_bytes),
        }
    
    def get_optimizer_state_memory(self, optimizer: torch.optim.Optimizer, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
        """
        Calculate optimizer state memory usage.
        
        Args:
            optimizer: PyTorch optimizer
            precision: 'fp32' or 'fp16' (optimizer states are typically fp32)
        
        Returns:
            Dictionary with parameter count and optimizer state memory in MB
        """
        total_optimizer_memory = 0
        param_count = 0
        
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            # Adam/AdamW: 2 states per parameter (momentum m and variance v)
            states_per_param = 2
        elif isinstance(optimizer, torch.optim.SGD):
            # SGD: 1 state per parameter (momentum) if momentum > 0
            momentum = optimizer.param_groups[0].get('momentum', 0)
            states_per_param = 1 if momentum > 0 else 0
        else:
            raise NotImplementedError(f'Invalid optimizer: {optimizer}')
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    num_params = p.numel()
                    total_optimizer_memory += states_per_param * num_params * OPTIMIZER_STATE_BYTES_PER_PARAM
                    param_count += num_params
        
        return {
            'param_count': param_count,
            'optimizer_memory_MB': self._bytes_to_mb(total_optimizer_memory),
        }