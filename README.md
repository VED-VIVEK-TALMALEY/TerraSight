# TerraSight: Multimodal Vision-Language Platform for Earth Observation

## Abstract
TerraSight (developed as EarthAware) is an end-to-end geospatial research system implementing a Multispectral Vision-Language Model (VLM). The platform integrates a SpectralViT encoder with a GPT-2 decoder using LoRA adapters to provide natural language analysis of Earth Observation (EO) data.

## System Architecture

### Multi-Layer Stack
1. Client Layer: React 18 / TypeScript / MapLibre GL
2. Orchestration Layer: Node.js / Express.js / TypeScript
3. AI Inference Layer: Python 3.10 / FastAPI / PyTorch
4. Model Layer: SpectralViT + GPT-2 (PEFT/LoRA)

### Core Capabilities
* Multispectral Band Analysis (13-band Sentinel-2 support)
* Land Cover Classification (Automated scene grounded analysis)
* Visual Question Answering (VQA) for EO data
* Multimodal Change Detection
* Interactive 3D Geospatial Research Interface

## Performance Metrics

| Metric | Measured Value |
|--------|----------------|
| Validation Loss (Global) | 3.514 |
| NDVI Estimation Accuracy | 85.0% |
| Training Convergence | 41.3% Loss Reduction |
| Training Dataset Size | 70k-80k Instruction Pairs |

## Technical Implementation

### Vision Encoder (SpectralViT)
The system utilizes a custom Vision Transformer modified with a Spectral Attention Module. This architecture is designed to process non-RGB bands (NIR, SWIR, Red-Edge) natively, providing superior feature extraction compared to standard vision models.

### Language Decoder
A language model is integrated via a projection layer, enabling the system to generate research-grade reports and answer complex queries about specific geographic regions.

## Repository Structure
* earthaware/: Core implementation and AI services
* hf-spaces-demo/: Optimized production assets for cloud deployment

## Compliance and Standards
This project is designed in alignment with ISRO multimodal satellite imagery requirements.
