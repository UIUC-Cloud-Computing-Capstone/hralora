# Smoke test: CIFAR-100 Ours, 1 round, 2 clients (verify/CI).
# From project root: bash scripts/experiments/run-cifar100-smoke.sh
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/smoke_test/smoke.yaml'
