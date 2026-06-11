# Build Protocol: TerraSight System Implementation

## 1. Abstract
This document delineates the standard operating procedure for building the TerraSight system from source. The architecture consists of a multispectral Vision-Language Model (VLM) integrated with a full-stack geospatial research interface.

## 2. Infrastructure Setup

### 2.1 Virtual Environment Initialization
python -m venv venv
./venv/Scripts/activate

### 2.2 Dependency Resolution
pip install torch torchvision transformers peft numpy pillow matplotlib fastapi uvicorn strawberry-graphql[fastapi] python-multipart tqdm requests streamlit

## 3. Web Application Construction

### 3.1 Orchestration Layer (Node.js/Express)
cd earthaware/geo-research-assistant/backend
npm install
npm run build

### 3.2 Client Layer (React/TypeScript)
cd ../frontend
npm install
npm run build

## 4. Model Checkpoint Integration
Place the trained model weights (isro_eo_enhanced_best.pt) within the directory:
earthaware/checkpoints/

## 5. Execution
Refer to QUICK_START.md for runtime procedures.
