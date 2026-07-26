# TerraSight: Evaluation and Deployment Protocol

## 1. Evaluation Framework

### 1.1 Quantitative Metrics
The system is evaluated based on the following technical parameters:
*   **Visual Question Answering (VQA) Accuracy**: Correctness of model responses to geographic queries using keyword-based validation.
*   **BLEU Score**: Measurement of linguistic similarity between model-generated captions and expert-provided ground truth.
*   **EO Terminology Coverage**: Density of domain-specific Earth Observation terms (e.g., NDVI, backscatter, spectral reflectance) within generated reports.
*   **Satellite Identification Accuracy**: Correct classification of sensor types (RESOURCESAT, CARTOSAT, RISAT) from multispectral signatures.

### 1.2 Training Convergence
Performance is monitored across 7 training epochs with the best model checkpoint identified at Epoch 6 (Global Step 315) with a validation loss of 3.514.

## 2. Qualitative Assessment

### 2.1 Expert Review Protocol
The qualitative evaluation phase involves human-in-the-loop verification of model outputs. Researchers rate responses on a scale of 1-5 based on factuality, relevance, and technical accuracy.

## 3. Deployment Architecture

### 3.1 API Service (FastAPI)
The core model is served via a FastAPI interface, providing high-throughput endpoints for single-image analysis, batch processing, and comparative change detection.

### 3.2 Containerization (Docker)
The system is encapsulated within a Docker container based on NVIDIA CUDA runtimes to ensure environment parity across different compute clusters.

### 3.3 Public Cloud Deployment (Streamlit)
For public access and demonstration, a lightweight version of the system is deployed to Streamlit Cloud. Refer to DEPLOYMENT_STREAMLIT.md for configuration details.

## 4. Performance Optimization
*   **Quantization**: Utilization of BitsAndBytes (int8) quantization to reduce VRAM requirements during inference.
*   **Parameter-Efficient Fine-Tuning (PEFT)**: Implementation of LoRA (Low-Rank Adaptation) to minimize the number of trainable parameters while maintaining research-grade performance.

