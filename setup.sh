#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip if not already installed
sudo apt-get install -y python3 python3-pip

# Install CUDA dependencies (adjust version as needed)
sudo apt-get install -y nvidia-cuda-toolkit

# Install UV for faster package management
pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install requirements
uv pip install -r requirements.txt

# Install additional GPU-specific packages
uv pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install triton==2.2.0
uv pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import torch; print('CUDA device count:', torch.cuda.device_count())" 