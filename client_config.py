"""
Configuration classes for the Flower client.

This module contains the Config and DatasetArgs classes that handle
configuration management and dataset loading compatibility.

Author: Team1-FL-RHLA
Version: 1.0.0
"""

import logging
import torch
from typing import Dict, Any

from client_constants import (
    CONFIG_KEY_DATASET, CONFIG_KEY_MODEL, CONFIG_KEY_DATA_TYPE, CONFIG_KEY_PEFT,
    CONFIG_KEY_BATCH_SIZE, CONFIG_KEY_NUM_USERS, CONFIG_KEY_NUM_CLASSES,
    CONFIG_KEY_IID, CONFIG_KEY_LABEL2ID, CONFIG_KEY_ID2LABEL,
    DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_DATA_TYPE, DEFAULT_PEFT,
    DEFAULT_BATCH_SIZE, DEFAULT_NUM_USERS, DEFAULT_NUM_CLASSES,
    DEFAULT_ZERO_VALUE, DEFAULT_ONE_VALUE, DEFAULT_CPU_DEVICE, DEFAULT_CUDA_DEVICE,
    ERROR_INVALID_CONFIG
)


class DatasetArgs:
    """
    Arguments class for dataset loading compatibility.
    
    This class provides a bridge between the configuration dictionary and the 
    dataset loading functions, ensuring compatibility with the existing data 
    preprocessing pipeline.
    
    Example:
        config_dict = {CONFIG_KEY_DATASET: DEFAULT_DATASET, CONFIG_KEY_BATCH_SIZE: DEFAULT_BATCH_SIZE}
        args = DatasetArgs(config_dict, client_id=DEFAULT_CLIENT_ID)
    """
    
    def __init__(self, config_dict: Dict[str, Any], client_id: int):
        """Initialize dataset args from configuration dictionary."""
        # Store config dict for direct access instead of individual attributes
        self.config_dict = config_dict
        self.client_id = client_id
        
        # Only store essential attributes that are frequently accessed
        self.dataset = config_dict.get(CONFIG_KEY_DATASET, DEFAULT_DATASET)
        self.model = config_dict.get(CONFIG_KEY_MODEL, DEFAULT_MODEL)
        self.data_type = config_dict.get(CONFIG_KEY_DATA_TYPE, DEFAULT_DATA_TYPE)
        self.peft = config_dict.get(CONFIG_KEY_PEFT, DEFAULT_PEFT)
        self.batch_size = config_dict.get(CONFIG_KEY_BATCH_SIZE, DEFAULT_BATCH_SIZE)
        self.num_users = config_dict.get(CONFIG_KEY_NUM_USERS, DEFAULT_NUM_USERS)
        
        # Device configuration
        self.device = torch.device(DEFAULT_CUDA_DEVICE if torch.cuda.is_available() else DEFAULT_CPU_DEVICE)
        
        # Logger
        self.logger = self._create_simple_logger()
        
        # Additional attributes that might be needed
        self.num_classes = config_dict.get(CONFIG_KEY_NUM_CLASSES, DEFAULT_NUM_CLASSES)
        self.labels = None  # Will be set by load_partition
        self.label2id = None  # Will be set by load_partition
        self.id2label = None  # Will be set by load_partition
        
        # Computed attributes
        self.iid = config_dict.get(CONFIG_KEY_IID, DEFAULT_ZERO_VALUE) == DEFAULT_ONE_VALUE
        self.noniid = not self.iid

    def _create_simple_logger(self):
        """Create a simple logger for compatibility."""
        class SimpleLogger:
            def info(self, msg, main_process_only=False):
                logging.info(msg)
        return SimpleLogger()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config_dict.get(key, default)
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from config_dict if not found as direct attribute."""
        if name in self.config_dict:
            return self.config_dict[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
  

class Config:
    """
    Configuration class to hold and manage configuration parameters.
    
    This class provides a clean interface for accessing configuration parameters
    with proper validation and default value handling. It acts as a wrapper around
    a configuration dictionary, providing type safety and validation.
    
    Example:
        config_dict = {CONFIG_KEY_DATASET: DEFAULT_DATASET, CONFIG_KEY_BATCH_SIZE: DEFAULT_BATCH_SIZE, 'learning_rate': DEFAULT_ZERO_POINT_ZERO_ONE}
        config = Config(config_dict)
        dataset = config.get(CONFIG_KEY_DATASET, DEFAULT_DEFAULT_VALUE)
        batch_size = config.batch_size
    """
    
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Raises:
            ValueError: If config_dict is None or empty
        """
        if not config_dict:
            raise ValueError(ERROR_INVALID_CONFIG)
            
        for key, value in config_dict.items():
            if not isinstance(key, str):
                raise ValueError(f"Configuration keys must be strings, got {type(key)}")
            setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
    
    def has(self, key: str) -> bool:
        """Check if configuration parameter exists."""
        return hasattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.to_dict()})"
