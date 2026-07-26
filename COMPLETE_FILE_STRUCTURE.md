# Project Repository Structure

## 1. Directory Tree Architecture

```
TerraSight/
├── earthaware/                          # Active development and implementation core
│   ├── api_server.py                    # FASTAPI inference engine
│   ├── train_isro_eo_enhanced.py        # LoRA-based training orchestration
│   ├── day4_multimodal_model.py         # SpectralViT + GPT-2 model definition
│   ├── geo-research-assistant/          # Full-stack research application
│   │   ├── backend/                     # Node.js orchestration layer
│   │   └── frontend/                    # React interface layer
│   ├── checkpoints/                     # Serialized model weights
│   ├── data/                            # Processed and raw EO datasets
│   ├── metrics/                         # Training history and performance logs
│   └── results/                         # Evaluation benchmarks
├── hf-spaces-demo/                      # Production assets for cloud hosting
├── README.md                            # Executive project summary
├── QUICK_START.md                       # Technical implementation guide
├── BUILD_GUIDE.md                       # Build and compilation protocol
└── DEPLOYMENT_STREAMLIT.md               # Cloud deployment procedure
```

## 2. Functional Component Description

### 2.1 Implementation Core (earthaware/)
The primary directory for model development, dataset processing, and local API serving. Contains the Spectral Attention Module and the integrated Vision-Language alignment logic.

### 2.2 Web Application (geo-research-assistant/)
A decoupled full-stack system providing a geospatial research environment. Features include 3D map selection, multisensor image upload, and real-time AI analysis.

### 2.3 Cloud Demo (hf-spaces-demo/)
An optimized package containing the Streamlit application for public evaluation. Features a mock inference engine for demonstration of spectral index analysis.

