# 🌸 Flower Federated Learning Framework

Comprehensive Flower-based federated learning implementation with heterogeneous LoRA allocation, multi-core CPU optimization, and robust configuration management.

## 🚀 Quick Start

### Prerequisites

**Choose ONE option:**

#### Option A: Conda Environment (Recommended)
```bash
conda env create --name env.fl --file=environment.yml
conda activate env.fl
```

#### Option B: Pip Installation
```bash
pip install flwr[simulation] accelerate torch torchvision transformers
```

### Running

**Terminal 1 - Start Server**:
```bash
python flower_server.py --server_address 0.0.0.0 --server_port 8080 --config_name experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml  --log_level INFO
```

**Terminal 2 - Start Client**:
```bash
python flower_client.py --server_address localhost --server_port 8080 --client_id 0 --config_name experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml --log_level INFO
```

### Multiple Clients

Start additional clients with different `--client_id` values (1, 2, 3, etc.) in separate terminals.

### Using Different Configurations

You can use different configuration files by specifying the `--config_name` parameter:

```bash
# Use the default Flower configuration
python flower_server.py --config_name experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml

# Use other available configurations
python flower_server.py --config_name experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml
```

### Multi-Machine Setup

Replace `localhost` with the server's IP address on client machines.

## 🔧 Configuration

**Key Parameters:**
- `--server_address`: Server IP (0.0.0.0 for server, localhost for client)
- `--server_port`: Server port (default: 8080)
- `--client_id`: Unique client identifier
- `--config_name`: Configuration file path

**Flower-Specific Configuration:**
The implementation uses a dedicated Flower configuration file:
- `experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml`
- Optimized for federated learning with 500 rounds, 10 selected users
- Configured for CIFAR-100 with ViT and LoRA fine-tuning
- Non-IID data distribution with pathological partitioning

**CPU Optimization:**
```bash
export OMP_NUM_THREADS=8        # Override CPU thread count
export TORCH_NUM_THREADS=8      # PyTorch threads
```

## 🏗️ Architecture

**Components:**
- `flower_server.py`: Central server (FedAvg strategy)
- `flower_client.py`: Client with comprehensive federated learning support
- `algorithms/solver/fl_utils.py`: Common utilities and aggregation

**Features:**
- Multi-core CPU utilization and optimization
- LoRA-aware parameter aggregation
- Heterogeneous client support with different architectures
- Comprehensive configuration management via YAML
- Dataset loading and management
- Realistic training and evaluation simulation
- Modern Flower API compatibility (no deprecation warnings)
- Robust error handling and logging
- Production-ready code quality

## 🐛 Troubleshooting

**Common Issues:**
- **Connection Refused**: Start server before clients, check firewall
- **Missing Dependencies**: `pip install -r requirements_flower.txt`
- **Slow Training**: Check CPU usage with `htop`, verify multi-core setup

**Debug Mode:**
```bash
export FLWR_LOG_LEVEL=DEBUG
python flower_server.py --config_name experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml
```

## 📚 Resources

- [Flower Documentation](https://flower.dev/docs/)
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)
