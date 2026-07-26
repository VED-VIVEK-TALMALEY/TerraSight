# Web Application Setup and Integration Guide

## 1. Abstract
The TerraSight platform includes a geospatial research portal consisting of a Node.js/TypeScript backend and a React/Vite frontend. This system orchestrates communication between the user interface and the Python-based AI model.

## 2. Environment Prerequisites
*   Node.js v18.x or higher
*   NPM v9.x or higher
*   Python v3.10.x (with FastAPI)

## 3. Backend Orchestration Setup

### 3.1 Installation
```bash
cd earthaware/geo-research-assistant/backend
npm install
```

### 3.2 Configuration
Create a `.env` file within the backend directory with the following parameters:
```text
PORT=8080
CORS_ORIGIN=http://localhost:5173
LOCAL_MODEL_API_BASE=http://localhost:8000
AUTH_USER=researcher
AUTH_PASSWORD=earthaware123
```

### 3.3 Execution
```bash
npm run dev
```

## 4. Frontend Client Setup

### 4.1 Installation
```bash
cd earthaware/geo-research-assistant/frontend
npm install
```

### 4.2 Configuration
Create a `.env` file within the frontend directory:
```text
VITE_API_BASE_URL=http://localhost:8080
VITE_MAPBOX_TOKEN=your_mapbox_access_token
```

### 4.3 Execution
```bash
npm run dev
```

## 5. Integration Protocol
The frontend interacts with the orchestration layer via RESTful API calls. The orchestration layer validates requests using Zod schemas and forwards multispectral data to the Python model server for inference.

