#!/bin/bash
# Setup script for running on a RunPod GPU server

# Create a conda environment (adjust Python version as needed)
conda create -n embedding_analysis python=3.9 -y

# Activate the environment
source activate embedding_analysis

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other requirements
pip install -r requirements.txt

# Run the analysis with GPU support
python main.py --use_gpu --use_umap