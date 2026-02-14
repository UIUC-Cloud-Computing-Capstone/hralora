
#Hardware

Hardware we used:

NVIDIA H100 80GB HBM3 (Driver version 580.65.06, CUDA version 13.0)


# Troubleshooting
## Rank Estimator

If you see the error `torch.cuda.OutOfMemoryError: CUDA out of memory`, wait a few minutes for memory to be released. If it still doesn't work, use a GPU instance with larger memory. 