FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

RUN conda init bash

WORKDIR /workspace

# Clone repo
RUN git clone https://github.com/UIUC-Cloud-Computing-Capstone/team1-fl-rhla.git
WORKDIR /workspace/team1-fl-rhla

# Create conda environment
RUN conda env create --name env.fl --file=setup/environment-final.yml

# Create HuggingFace accelerate config directory
RUN mkdir -p /root/.cache/huggingface/accelerate

# Execute
CMD source /opt/conda/etc/profile.d/conda.sh && \
    conda activate env.fl && \
    nvidia-smi || true && \
    cp setup/accelerate_default_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml && \
    bash scripts/experiments/run-cifar100-smoke.sh
