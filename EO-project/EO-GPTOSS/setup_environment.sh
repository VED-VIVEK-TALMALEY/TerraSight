#!/bin/bash

# ISRO Multimodal Satellite Imagery Project - Environment Setup
# Run this script to set up your development environment

echo "================================================"
echo "ISRO Multimodal Project - Environment Setup"
echo "================================================"

# Update system packages
echo "Updating system packages..."
sudo apt-get update -qq

# Install system dependencies for geospatial libraries
echo "Installing system dependencies..."
sudo apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    libspatialindex-dev

# Create project directory structure
echo "Creating project directories..."
mkdir -p ~/isro_multimodal_project/{data,outputs,notebooks,results,models,logs}
mkdir -p ~/isro_multimodal_project/data/{raw,processed,annotations}
mkdir -p ~/isro_multimodal_project/outputs/{visualizations,predictions}
mkdir -p ~/isro_multimodal_project/results/{baseline,multispectral,comparisons}

cd ~/isro_multimodal_project

# Create Python virtual environment (if not using Colab)
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 11.8 version - adjust if needed)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related libraries
echo "Installing Hugging Face libraries..."
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install bitsandbytes==0.41.3
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3

# Install geospatial libraries
echo "Installing geospatial libraries..."
pip install rasterio==1.3.9
pip install geopandas==0.14.1
pip install shapely==2.0.2
pip install pyproj==3.6.1
pip install earthengine-api==0.1.383

# Install scientific computing libraries
echo "Installing scientific libraries..."
pip install numpy==1.24.3
pip install pandas==2.1.4
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install scikit-learn==1.3.2
pip install scikit-image==0.22.0
pip install opencv-python==4.8.1.78

# Install visualization and UI libraries
echo "Installing visualization libraries..."
pip install plotly==5.18.0
pip install ipywidgets==8.1.1
pip install tqdm==4.66.1
pip install pillow==10.1.0
pip install streamlit==1.29.0

# Install utilities
echo "Installing utilities..."
pip install requests==2.31.0
pip install aiohttp==3.9.1
pip install python-dotenv==1.0.0
pip install pyyaml==6.0.1
pip install wandb==0.16.1

# Install Jupyter for notebooks
echo "Installing Jupyter..."
pip install jupyter==1.0.0
pip install ipykernel==6.27.1

# Verify installations
echo ""
echo "================================================"
echo "Verifying installations..."
echo "================================================"

python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")

try:
    import rasterio
    print(f"✓ Rasterio: {rasterio.__version__}")
except Exception as e:
    print(f"✗ Rasterio: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except Exception as e:
    print(f"✗ Pandas: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
except Exception as e:
    print(f"✗ Matplotlib: {e}")

try:
    from peft import LoraConfig
    print(f"✓ PEFT: Available")
except Exception as e:
    print(f"✗ PEFT: {e}")

print("\n✓ Setup complete!")
EOF

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Project directory: ~/isro_multimodal_project"
echo ""
echo "Next steps:"
echo "1. cd ~/isro_multimodal_project"
echo "2. Download satellite imagery data"
echo "3. Start with baseline model testing"
echo ""
