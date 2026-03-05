<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.6+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18+-61dafb?logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/TypeScript-5+-3178c6?logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

# 🌍 TerraSight — Multimodal AI for Earth Observation

**TerraSight** (EarthAware) is an end-to-end geospatial AI platform that combines a **Multispectral Vision-Language Model** with an interactive **3D map research assistant**. Built as a solution to ISRO's challenge of enhancing open-source GPT models with multimodal vision capabilities for Earth Observation (EO) data.

> _"Ask questions about satellite imagery in natural language and get intelligent, domain-aware answers."_

---

## ✨ Highlights

- 🛰️ **Multispectral Vision-Language Model** — SpectralViT encoder + GPT-2 decoder with LoRA fine-tuning  
- 🗺️ **3D Map Research Assistant** — Interactive MapLibre GL interface with polygon selection & real-time analysis  
- 🔬 **ISRO Satellite Support** — Designed for RESOURCESAT, CARTOSAT, RISAT, OCEANSAT, and Sentinel-2 data  
- 📡 **Full-Stack Architecture** — Python ML backend → Node.js orchestration → React frontend  
- 🎯 **Instruction-Tuned** — Fine-tuned on EO-specific instruction data for land cover classification, VQA, change detection & disaster response  
- 📊 **Built-in Evaluation** — Comprehensive metrics: VQA accuracy, BLEU scores, EO terminology analysis  

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TerraSight Platform                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  React +     │───▶│  Express.js  │───▶│  FastAPI     │      │
│  │  MapLibre 3D │    │  Backend     │    │  Model API   │      │
│  │  Frontend    │◀───│  (Node.js)   │◀───│  (Python)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         ▲                                       │               │
│         │                                       ▼               │
│  ┌──────────────┐                      ┌──────────────┐        │
│  │  Zustand     │                      │ SpectralViT  │        │
│  │  State Mgmt  │                      │ + GPT-2      │        │
│  └──────────────┘                      │ Multimodal   │        │
│                                        └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + TypeScript + Vite + MapLibre GL | 3D map UI, research chat, multimodal upload, auth & dashboard |
| **Orchestration Backend** | Express.js + TypeScript | API routing, chat session management, RL feedback, training job management |
| **ML API** | FastAPI + Uvicorn | Model inference (`/analyze`, `/chat`, `/batch_analyze`, `/health`) |
| **Model** | SpectralViT + GPT-2 + LoRA (PEFT) | Multispectral image encoding + text generation |
| **Data Pipeline** | PyTorch DataLoader + NumPy + Pillow | EO image preprocessing, synthetic data generation, dataset expansion |
| **Evaluation** | Custom evaluators | VQA accuracy, BLEU score, EO terminology coverage, satellite-specific benchmarks |
| **Demo UI** | Streamlit | Lightweight single-page demo for model inference |

---

## 📁 Project Structure

```
TerraSight/
│
├── earthaware/                          # Core ML + backend project
│   ├── api_server.py                    # FastAPI inference server
│   ├── train_isro_eo_enhanced.py        # Main training script (LoRA + class balancing)
│   ├── day4_multimodal_model.py         # Multimodal model (SpectralViT + GPT-2)
│   ├── day3_spectral_vit.py             # SpectralViT encoder architecture
│   ├── day3_spectral_attention.py       # Spectral attention mechanism
│   ├── day3_patch_embedding.py          # Patch embedding for multispectral data
│   ├── day3_create_dataset.py           # EO dataset creation
│   ├── day4_train.py                    # Baseline training script
│   ├── day4_evaluate.py                 # Sample-level evaluation
│   ├── day5_evaluate_comprehensive.py   # Comprehensive benchmark evaluation
│   ├── expand_training_dataset.py       # Dataset augmentation (paraphrase + QA)
│   ├── download_satellite_data.py       # Satellite data downloader
│   ├── day2_download_multispectral.py   # Multispectral band downloader
│   ├── day2_create_composites.py        # Band composite creation
│   ├── streamlit_app.py                 # Streamlit demo frontend
│   ├── research_chat_ui.html            # Standalone HTML research chat
│   ├── backend_graphql.py               # GraphQL backend (experimental)
│   ├── run_project.py                   # CLI runner for all components
│   ├── verify_setup.py                  # Environment verification
│   ├── requirements.txt                 # Python dependencies
│   │
│   ├── geo-research-assistant/          # Full-stack web application
│   │   ├── backend/                     # Express.js + TypeScript backend
│   │   │   └── src/
│   │   │       ├── server.ts            # Express app bootstrap
│   │   │       ├── routes/research.ts   # Area/chat/multimodal/training APIs
│   │   │       ├── routes/auth.ts       # Authentication endpoints
│   │   │       ├── services/openaiService.ts  # AI provider orchestration
│   │   │       ├── validation/          # Zod request validation
│   │   │       └── utils/cache.ts       # TTL response cache
│   │   │
│   │   └── frontend/                    # React + TypeScript + Vite frontend
│   │       └── src/
│   │           ├── App.tsx              # App shell with auth gate
│   │           ├── components/Map3D.tsx # MapLibre 3D map with polygon selection
│   │           ├── components/ChatPanel.tsx  # Research chat + analysis UI
│   │           ├── store/useGeoStore.ts # Zustand state management
│   │           └── lib/api.ts           # API client
│   │
│   ├── checkpoints/                     # Saved model weights
│   ├── data/                            # Training/evaluation datasets
│   │   ├── raw/eurosat/                 # EuroSAT image dataset
│   │   ├── raw/sentinel2_multispectral/ # Sentinel-2 band data (.npy)
│   │   └── training/                    # Instruction-tuning JSON annotations
│   ├── metrics/                         # Training loss curves & history
│   └── results/                         # Evaluation outputs
│
├── EO-project/EO-GPTOSS/               # Original EO-GPT prototype
│
├── train_isro_multimodal.py             # Complete 3-stage training pipeline
├── isro_multimodal_proposal.md          # Full technical proposal & architecture
├── evaluation_deployment_guide.md       # Evaluation framework & deployment configs
├── BUILD_GUIDE.md                       # Build-from-scratch guide
├── QUICK_START.md                       # Quick start setup instructions
├── QUICK_START_WEB.md                   # Web app quick start
├── WEB_APP_SETUP_GUIDE.md              # Detailed web app setup
└── COMPLETE_FILE_STRUCTURE.md           # Exhaustive file listing
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **NVIDIA GPU** with CUDA 11.8+ (recommended, 8GB+ VRAM)
- ~50GB free disk space

### 1. Clone & Setup Python Environment

```bash
git clone https://github.com/VED-VIVEK-TALMALEY/TerraSight.git
cd TerraSight/earthaware

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python verify_setup.py
```

Expected output:
```
✓ Python 3.10+
✓ PyTorch: 2.6.0
✓ Transformers: 4.36.0+
✓ CUDA available
✓ All checks passed
```

### 3. Train the Model

```bash
# Quick training (2 epochs, low resource)
python train_isro_eo_enhanced.py --epochs 2 --batch-size 1 --grad-accum 4 --lr 2e-5

# Or use the CLI runner
python run_project.py train --epochs 2 --batch-size 1 --grad-accum 4 --lr 2e-5
```

### 4. Start the Model API

```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### 5. Launch the Web App

```bash
# Terminal 1 — Node.js backend
cd geo-research-assistant/backend
npm install && npm run dev

# Terminal 2 — React frontend
cd geo-research-assistant/frontend
npm install && npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 🧠 Model Architecture

### SpectralViT + GPT-2 Multimodal Model

```
                  Multispectral Input (13 bands, 512×512)
                              │
                    ┌─────────▼──────────┐
                    │  Spectral Attention │  ← Band-aware attention
                    │  + Patch Embedding  │     for multi-band EO data
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   SpectralViT       │  ← Vision Transformer
                    │   Encoder           │     with spectral adapters
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Projection Layer  │  ← Maps vision → language space
                    │   (Linear/MLP)      │
                    └─────────┬──────────┘
                              │
            ┌─────────────────▼────────────────────┐
            │         GPT-2 Language Decoder        │
            │  + LoRA Adapters (PEFT fine-tuning)   │
            └──────────────────────────────────────┘
                              │
                        Text Response
```

**Key Features:**
- **Spectral Attention** — Custom attention module aware of multispectral band relationships  
- **LoRA Fine-Tuning** — Parameter-efficient training via PEFT (trainable params < 1% of total)  
- **Class-Balanced Sampling** — Weighted sampling for underrepresented land cover types  
- **Sensor Normalization** — Separate normalization for optical, SAR, and thermal data  

### Training Pipeline

| Stage | Objective | What's Trained |
|-------|-----------|----------------|
| **Stage 1** — Pretraining | Align vision and language spaces | Projection layer only |
| **Stage 2** — Instruction Tuning | Learn EO question-answering | LoRA adapters + projection |
| **Stage 3** — ISRO Fine-Tuning | Domain specialization | Full fine-tuning on ISRO data |

---

## 🗺️ Geo Research Assistant

The web application provides an interactive research interface:

### Features
- **3D Map** — MapLibre GL with terrain, satellite basemap, and Shift+drag polygon selection  
- **Area Analysis** — Draw a polygon → automatic AI analysis of the region  
- **Research Chat** — Follow-up questions with contextual memory  
- **Multimodal Upload** — Upload images/video for AI-powered analysis  
- **RL Feedback** — Rate model responses to improve future training  
- **Manual Training** — Trigger retraining from the dashboard  
- **Auth & Dashboard** — Login system with training metrics display  

### User Flow

```
Draw polygon on map → Auto-capture area image → Send to AI backend
        │                                              │
        ▼                                              ▼
  Chat follow-ups  ←──────────────────────  Structured analysis
  (flood risk, vegetation health, etc.)     (land cover, features, etc.)
```

---

## 📡 API Reference

### Python Model API (FastAPI — Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Single image + question → AI analysis |
| `/batch_analyze` | POST | Batch image analysis |
| `/analyze_dual` | POST | Two-image comparative analysis + change report |
| `/chat` | POST | Research assistant chat with session memory |
| `/chat/reset` | POST | Reset chat session |
| `/health` | GET | Service health check |

### Node.js Orchestration API (Express — Port 3001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/research/area` | POST | Area polygon analysis |
| `/api/research/chat` | POST | Follow-up research chat |
| `/api/research/multimodal` | POST | Multimodal media analysis |
| `/api/research/feedback` | POST | Submit RL feedback |
| `/api/research/manual-train` | POST | Trigger training job |
| `/api/research/manual-train/:jobId` | GET | Check training job status |
| `/api/research/dashboard-metrics` | GET | Aggregate metrics |
| `/api/auth/login` | POST | User authentication |
| `/api/auth/me` | GET | Current user info |

---

## 📊 Evaluation & Metrics

### Quantitative Metrics

| Metric | Description |
|--------|-------------|
| **VQA Accuracy** | Visual Question Answering correctness with keyword matching |
| **BLEU Score** | Caption generation quality (corpus-level) |
| **EO Terminology** | Average domain-specific terms per response |
| **Satellite-Specific** | Accuracy on RESOURCESAT / CARTOSAT / RISAT queries |

### Training Results (Latest)

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 5.0410 | 4.1517 |
| 2 | 3.8561 | 3.9485 |

Run evaluation:
```bash
python day4_evaluate.py                    # Quick evaluation
python day5_evaluate_comprehensive.py      # Full benchmark suite
```

---

## 🛰️ Supported Satellite Systems

| Satellite | Sensor | Bands | Resolution | Use Case |
|-----------|--------|-------|------------|----------|
| **RESOURCESAT** | LISS-III | 4 | 23.5m | Agriculture, forestry, land use |
| **RESOURCESAT** | LISS-IV | 3 | 5.8m | High-res mapping |
| **RESOURCESAT** | AWiFS | 4 | 56m | Wide-area monitoring |
| **CARTOSAT** | PAN | 1 | 2.5m | Stereo mapping, DEM |
| **CARTOSAT** | MX | 4 | 2.5m | Urban planning |
| **RISAT** | SAR | 1 | 1m | All-weather/night imaging |
| **Sentinel-2** | MSI | 13 | 10-60m | Reference multispectral data |

---

## 🐳 Deployment

### Docker

```bash
docker-compose up --build
```

### Kubernetes

```bash
kubectl apply -f kubernetes/deployment.yaml
```

See [`evaluation_deployment_guide.md`](evaluation_deployment_guide.md) for full Docker, Kubernetes, and production deployment configurations.

---

## 🔧 CLI Runner

The `run_project.py` script provides a unified interface:

```bash
python run_project.py check      # Verify environment
python run_project.py train      # Train model
python run_project.py eval       # Run evaluation
python run_project.py api        # Start FastAPI server
python run_project.py streamlit  # Launch Streamlit demo
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`isro_multimodal_proposal.md`](isro_multimodal_proposal.md) | Complete technical proposal with architecture design |
| [`evaluation_deployment_guide.md`](evaluation_deployment_guide.md) | Evaluation framework + Docker/K8s deployment |
| [`BUILD_GUIDE.md`](BUILD_GUIDE.md) | Step-by-step build guide |
| [`QUICK_START.md`](QUICK_START.md) | Setup instructions (Colab, local, Docker) |
| [`WEB_APP_SETUP_GUIDE.md`](WEB_APP_SETUP_GUIDE.md) | Detailed web application setup |
| [`COMPLETE_FILE_STRUCTURE.md`](COMPLETE_FILE_STRUCTURE.md) | Exhaustive file listing |
| [`earthaware/DETAILED_PROJECT_REPORT.md`](earthaware/DETAILED_PROJECT_REPORT.md) | File-by-file functional map |

---

## 🛠️ Tech Stack

**Machine Learning:**  
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)

**Backend:**  
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Express](https://img.shields.io/badge/Express.js-000?logo=express&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-339933?logo=node.js&logoColor=white)

**Frontend:**  
![React](https://img.shields.io/badge/React-61dafb?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178c6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646cff?logo=vite&logoColor=white)
![MapLibre](https://img.shields.io/badge/MapLibre_GL-396cb2)

**Data Science:**  
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c)

---

## 👤 Author

**Ved Vivek Talmaley**  
🔗 [GitHub](https://github.com/VED-VIVEK-TALMALEY)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built with ❤️ for Earth Observation & ISRO</b><br/>
  <i>TerraSight — See the Earth through AI</i>
</p>
