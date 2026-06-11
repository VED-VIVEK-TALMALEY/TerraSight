# Quick Start Guide: TerraSight Implementation

## 1. System Requirements

* OS: Windows 10+, Linux (Ubuntu 20.04+), or macOS 12+
* Python: 3.10.x or higher
* Node.js: 18.x or higher
* Hardware: NVIDIA GPU with 8GB+ VRAM (Recommended for local inference)

## 2. Environment Configuration

### 2.1 Clone Repository
git clone https://github.com/VED-VIVEK-TALMALEY/TerraSight.git
cd TerraSight/earthaware

### 2.2 Python Virtual Environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

### 2.3 Web Application Dependencies
# Backend Orchestration
cd geo-research-assistant/backend
npm install

# Frontend Interface
cd ../frontend
npm install

## 3. Deployment Protocol

### 3.1 Local Model Inference Server (FastAPI)
cd earthaware
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

### 3.2 Backend Service (Express.js)
cd earthaware/geo-research-assistant/backend
npm run dev

### 3.3 Frontend Research Portal (React/Vite)
cd earthaware/geo-research-assistant/frontend
npm run dev

## 4. Verification

Access the research portal at: http://localhost:5173
Model API health status available at: http://localhost:8000/health
