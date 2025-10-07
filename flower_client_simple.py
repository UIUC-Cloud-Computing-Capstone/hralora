"""
Simplified Flower Client for Testing
This version doesn't require data loading and can be used for basic testing
"""
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
import os
import sys
import yaml
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.solver.fl_utils import (
    compute_model_update,
    compute_update_norm,
    get_optimizer_parameters,
    setup_multiprocessing
)


class SimpleFlowerClient(fl.client.NumPyClient):
    """
    Simplified Flower client for testing without data dependencies.
    """
    
    def __init__(self, args: 'Config', client_id: int = 0):
        """
        Initialize the simplified Flower client.
        
        Args:
            args: Configuration object
            client_id: Client identifier
        """
        self.args = args
        self.client_id = client_id
        
        # Validate required configuration
        self._validate_config()
        
        # Setup multiprocessing for optimal CPU utilization
        self.num_cores = setup_multiprocessing()
        logging.info(f"Client {client_id} initialized with {self.num_cores} CPU cores")
        
        # Initialize loss function based on data type
        self.loss_func = self._get_loss_function()
        
        # Initialize training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'rounds': []
        }
        
        # Create dummy model parameters for testing
        self.model_params = self._create_dummy_model_params()
    
    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = ['data_type', 'peft', 'lora_layer']
        for param in required_params:
            if not hasattr(self.args, param):
                logging.warning(f"Missing configuration parameter: {param}, using default")
    
    def _get_loss_function(self) -> nn.Module:
        """Get appropriate loss function based on data type."""
        data_type = getattr(self.args, 'data_type', 'image')
        
        loss_functions = {
            'image': nn.CrossEntropyLoss(),
            'text': nn.CrossEntropyLoss(),
            'sentiment': nn.NLLLoss()
        }
        
        return loss_functions.get(data_type, nn.CrossEntropyLoss())
    
    def _create_dummy_model_params(self) -> List[np.ndarray]:
        """
        Create dummy model parameters for testing.
        
        Returns:
            List of numpy arrays representing model parameters
        """
        params = []
        
        try:
            # Add some dummy weight matrices
            params.append(np.random.randn(768, 768).astype(np.float32))  # Hidden layer
            params.append(np.random.randn(768).astype(np.float32))       # Bias
            params.append(np.random.randn(100, 768).astype(np.float32))  # Output layer
            params.append(np.random.randn(100).astype(np.float32))       # Output bias
            
            # Add LoRA parameters if specified
            if getattr(self.args, 'peft', '') == 'lora':
                lora_layers = getattr(self.args, 'lora_layer', 12)
                if lora_layers <= 0:
                    logging.warning("Invalid lora_layer value, using default of 12")
                    lora_layers = 12
                    
                for i in range(lora_layers):
                    params.append(np.random.randn(64, 768).astype(np.float32))  # LoRA A
                    params.append(np.random.randn(768, 64).astype(np.float32))  # LoRA B
            
            logging.info(f"Created {len(params)} dummy model parameters")
            return params
            
        except Exception as e:
            logging.error(f"Failed to create dummy model parameters: {e}")
            # Return minimal parameters as fallback
            return [np.random.randn(100, 100).astype(np.float32)]
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get current model parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of model parameters as numpy arrays
        """
        return self.model_params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.
        
        Args:
            parameters: List of model parameters from server
        """
        try:
            if not parameters:
                logging.warning("Received empty parameters from server")
                return
                
            self.model_params = [param.copy() for param in parameters]
            logging.debug(f"Updated model parameters with {len(parameters)} parameter arrays")
        except Exception as e:
            logging.error(f"Failed to set parameters: {e}")
            raise
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train the model on local data (simulated).
        
        Args:
            parameters: Model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        try:
            # Set parameters from server
            self.set_parameters(parameters)
            
            # Extract configuration with validation
            server_round = config.get('server_round', 0)
            local_epochs = max(1, config.get('local_epochs', getattr(self.args, 'tau', 1)))
            learning_rate = max(0.001, config.get('learning_rate', getattr(self.args, 'local_lr', 0.01)))
            
            logging.info(f"Client {self.client_id} starting training for round {server_round}")
            
            # Simulate training by adding small random updates to parameters
            total_loss = 0.0
            
            for epoch in range(local_epochs):
                epoch_loss = 0.0
                
                # Simulate training steps
                for step in range(10):  # Simulate 10 training steps per epoch
                    # Simulate forward pass and loss calculation
                    loss = np.random.exponential(1.0)  # Simulate decreasing loss
                    epoch_loss += loss
                    
                    # Simulate parameter updates
                    for i, param in enumerate(self.model_params):
                        # Add small random update
                        update = np.random.normal(0, 0.01, param.shape).astype(param.dtype)
                        self.model_params[i] = param + learning_rate * update
                
                total_loss += epoch_loss / 10  # Average loss per epoch
            
            avg_loss = total_loss / local_epochs
            
            # Update training history
            self.training_history['losses'].append(avg_loss)
            self.training_history['rounds'].append(server_round)
            
            logging.info(f"Client {self.client_id} completed training: Loss={avg_loss:.4f}")
            
            # Return parameters, number of examples, and metrics
            metrics = {
                'loss': avg_loss,
                'num_epochs': local_epochs,
                'client_id': self.client_id,
                'learning_rate': learning_rate
            }
            
            # Simulate number of examples (random between 100-1000)
            num_examples = np.random.randint(100, 1000)
            
            return self.get_parameters(config), num_examples, metrics
            
        except Exception as e:
            logging.error(f"Training failed for client {self.client_id}: {e}")
            # Return original parameters as fallback
            return parameters, 0, {'loss': float('inf'), 'error': str(e)}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate the model on local test data (simulated).
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        try:
            # Set parameters from server
            self.set_parameters(parameters)
            
            # Extract server round from config
            server_round = config.get('server_round', 0)
            
            # Simulate evaluation
            # Simulate accuracy (should improve over rounds)
            base_accuracy = 0.5 + min(0.4, server_round * 0.01)  # Improve over rounds
            accuracy = base_accuracy + np.random.normal(0, 0.05)  # Add some noise
            accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
            
            # Simulate loss (should decrease over rounds)
            base_loss = 2.0 - min(1.5, server_round * 0.01)  # Decrease over rounds
            loss = base_loss + np.random.normal(0, 0.1)  # Add some noise
            loss = max(0.1, loss)  # Clamp to positive values
            
            # Update training history
            self.training_history['accuracies'].append(accuracy)
            
            logging.info(f"Client {self.client_id} evaluation: "
                        f"Loss={loss:.4f}, Accuracy={accuracy:.4f}")
            
            # Return loss, number of examples, and metrics
            metrics = {
                'accuracy': accuracy,
                'client_id': self.client_id
            }
            
            # Simulate number of test examples
            num_examples = np.random.randint(50, 200)
            
            return loss, num_examples, metrics
            
        except Exception as e:
            logging.error(f"Evaluation failed for client {self.client_id}: {e}")
            # Return high loss as fallback
            return float('inf'), 0, {'accuracy': 0.0, 'error': str(e)}


class Config:
    """Simple configuration class to hold configuration parameters."""
    
    def __init__(self, config_dict: dict):
        """Initialize configuration from dictionary."""
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Get configuration value with default."""
        return getattr(self, key, default)


def load_configuration(config_path: str) -> 'Config':
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise ValueError("Configuration file is empty or invalid")
            
        return Config(config_dict)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")


def setup_logging(client_id: int, log_level: str = "INFO") -> None:
    """
    Setup logging configuration for the client.
    
    Args:
        client_id: Client identifier
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'%(asctime)s - Client {client_id} - %(levelname)s - %(message)s',
        force=True  # Override any existing logging configuration
    )


def create_simple_flower_client(args: 'Config', client_id: int = 0) -> SimpleFlowerClient:
    """
    Create a simplified Flower client instance.
    
    Args:
        args: Configuration object
        client_id: Client identifier
        
    Returns:
        SimpleFlowerClient instance
    """
    return SimpleFlowerClient(args, client_id)


def start_simple_flower_client(args: 'Config', server_address: str = "localhost", 
                              server_port: int = 8080, client_id: int = 0) -> None:
    """
    Start a simplified Flower client.
    
    Args:
        args: Configuration object
        server_address: Server address to connect to
        server_port: Server port to connect to
        client_id: Client identifier
    """
    try:
        # Setup logging
        setup_logging(client_id)
        
        # Create client
        client = create_simple_flower_client(args, client_id)
        
        # Start client
        logging.info(f"Starting simplified Flower client {client_id} connecting to {server_address}:{server_port}")
        
        fl.client.start_numpy_client(
            server_address=f"{server_address}:{server_port}",
            client=client,
        )
    except Exception as e:
        logging.error(f"Failed to start client {client_id}: {e}")
        raise


def setup_random_seeds(seed: int, client_id: int) -> None:
    """
    Setup random seeds for reproducibility.
    
    Args:
        seed: Base seed value
        client_id: Client identifier for unique seeds
    """
    client_seed = seed + client_id
    torch.manual_seed(client_seed)
    np.random.seed(client_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(gpu_id: int) -> torch.device:
    """
    Setup compute device (CPU or GPU).
    
    Args:
        gpu_id: GPU ID (-1 for CPU)
        
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available() and gpu_id != -1:
        if gpu_id >= torch.cuda.device_count():
            logging.warning(f"GPU {gpu_id} not available, falling back to CPU")
            return torch.device('cpu')
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Simplified Flower Client for Testing")
    parser.add_argument("--server_address", type=str, default="localhost", help="Server address")
    parser.add_argument("--server_port", type=int, default=8080, help="Server port")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID")
    parser.add_argument("--config_name", type=str, 
                       default="experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml",
                       help="Configuration file")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID (-1 for CPU)")
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the Flower client."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        config_path = os.path.join('config/', args.config_name)
        config = load_configuration(config_path)
        
        # Merge command line arguments into config
        for arg_name in vars(args):
            setattr(config, arg_name, getattr(args, arg_name))
        
        # Setup device
        config.device = setup_device(args.gpu)
        
        # Setup random seeds
        setup_random_seeds(args.seed, args.client_id)
        
        # Start client
        start_simple_flower_client(
            config, 
            args.server_address, 
            args.server_port, 
            args.client_id
        )
        
    except KeyboardInterrupt:
        logging.info("Client interrupted by user")
    except Exception as e:
        logging.error(f"Client failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
