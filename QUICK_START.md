# ISRO Multimodal Project - Quick Start Guide

## Setup Instructions (Choose Your Platform)

### Option 1: Google Colab (RECOMMENDED for beginners or no local GPU)

**Advantages:**
- Free GPU access (T4 with 15GB RAM)
- No installation needed
- Works on any computer
- Colab Pro ($10/month) gives better GPUs (A100/V100)

**Steps:**
1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `Day1_Setup.ipynb` to Colab
3. Change runtime: Runtime → Change runtime type → GPU
4. Run all cells in the notebook
5. You're ready to start!

**Important Colab Tips:**
- Sessions timeout after 12 hours
- Save important files to Google Drive
- Download model checkpoints regularly
- Use `%%capture` to hide long installation outputs

---

### Option 2: Local Machine with GPU

**Requirements:**
- NVIDIA GPU with at least 8GB VRAM (16GB recommended)
- CUDA 11.8 or later
- Ubuntu 20.04+ (or Windows with WSL2)
- Python 3.8+
- At least 50GB free disk space

**Installation Steps:**

1. **Clone/Create Project Directory**
```bash
mkdir ~/isro_multimodal_project
cd ~/isro_multimodal_project
```

2. **Copy Setup Files**
Place these files in your project directory:
- `setup_environment.sh`
- `requirements.txt`
- `verify_setup.py`

3. **Run Setup Script**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

This will:
- Install system dependencies
- Create virtual environment
- Install all Python packages
- Create directory structure
- Verify GPU availability

4. **Activate Virtual Environment**
```bash
source venv/bin/activate
```

5. **Verify Setup**
```bash
python verify_setup.py
```

You should see all checks passing ✓

---

### Option 3: Docker (Advanced - for consistent environments)

**Coming soon** - Docker setup for containerized deployment

---

## Quick Setup (Manual Installation)

If you want to install packages manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Verify
python verify_setup.py
```

---

## Troubleshooting Common Issues

### Issue: "No GPU detected"
**Solution:**
- If using Colab: Runtime → Change runtime type → GPU
- If local: Check NVIDIA drivers with `nvidia-smi`
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads

### Issue: "CUDA out of memory"
**Solutions:**
- Reduce batch size
- Use gradient checkpointing
- Use smaller model variants
- Clear cache: `torch.cuda.empty_cache()`

### Issue: "Rasterio installation failed"
**Solution (Ubuntu/Linux):**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
pip install rasterio
```

### Issue: "Transformers version conflict"
**Solution:**
```bash
pip install --upgrade transformers==4.36.0
```

### Issue: "Can't download models"
**Solutions:**
- Check internet connection
- Try using a VPN if Hugging Face is blocked
- Download models manually and load from local path

---

## Verifying Your Setup

After installation, verify everything works:

```bash
python verify_setup.py
```

Expected output:
```
✓ Python 3.10.12
✓ PyTorch: 2.1.0
✓ Transformers: 4.36.0
✓ CUDA available
✓ GPU: Tesla T4
✓ GPU Memory: 15.00 GB
✓ All checks passed
```

---

## Next Steps After Setup

Once setup is complete:

1. **Download Data** (Day 1 - Step 2)
   - Sentinel-2 imagery from Copernicus
   - Or sample datasets from Kaggle/HuggingFace

2. **Test Baseline Model** (Day 1 - Step 3)
   - Load BLIP-2
   - Test on sample images
   - Document baseline performance

3. **Follow Day-by-Day Plan**
   - Refer to `7_day_execution_plan.md`
   - Check off tasks as you complete them
   - Take notes in the notes section

---

## Project Structure After Setup

```
isro_multimodal_project/
├── data/
│   ├── raw/              # Original satellite images
│   ├── processed/        # Preprocessed images
│   └── annotations/      # Image-text pairs
├── outputs/
│   ├── visualizations/   # Plots and figures
│   └── predictions/      # Model outputs
├── results/
│   ├── baseline/         # Baseline results
│   ├── multispectral/    # Your model results
│   └── comparisons/      # Comparison charts
├── models/               # Saved model checkpoints
├── logs/                 # Training logs
├── notebooks/            # Jupyter notebooks
└── venv/                 # Python virtual environment
```

---

## Useful Commands

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Check GPU
nvidia-smi

# Check PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check installed packages
pip list

# Update a package
pip install --upgrade package_name

# Clear pip cache
pip cache purge

# Free GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

---

## Resources

- **Sentinel-2 Data**: https://scihub.copernicus.eu
- **Hugging Face Models**: https://huggingface.co/models
- **PyTorch Docs**: https://pytorch.org/docs
- **Transformers Docs**: https://huggingface.co/docs/transformers
- **Rasterio Docs**: https://rasterio.readthedocs.io

---

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Read error messages carefully
3. Search the error on Stack Overflow
4. Check library documentation
5. Ask specific questions with error logs

---

## Ready to Start?

Once your setup is verified (all ✓), proceed to:

**Day 1 - Data Collection and Baseline Testing**

Good luck! 🚀
