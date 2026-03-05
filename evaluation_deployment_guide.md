# ISRO MULTIMODAL GPT - EVALUATION & DEPLOYMENT GUIDE

## EVALUATION FRAMEWORK

### 1. Quantitative Metrics

```python
# evaluate_isro_model.py
"""
Comprehensive evaluation for ISRO Multimodal GPT
"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from collections import defaultdict

class ISROModelEvaluator:
    """
    Evaluate multimodal model on various EO tasks
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_vqa_accuracy(self, test_loader):
        """
        Visual Question Answering accuracy
        
        For questions with specific answers (land cover classification, etc.)
        """
        correct = 0
        total = 0
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="VQA Evaluation"):
                images = batch['image'].to(self.device)
                prompts = batch['prompt']
                true_responses = batch['response']
                
                # Generate predictions
                generated = self.model.generate(
                    images=images,
                    prompts=prompts,
                    max_new_tokens=100,
                    temperature=0.1  # Low temp for factual answers
                )
                
                # Compare
                for pred, truth in zip(generated, true_responses):
                    predictions.append(pred)
                    ground_truths.append(truth)
                    
                    # Simple accuracy (can be improved with semantic similarity)
                    if self.is_answer_correct(pred, truth):
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
    
    def is_answer_correct(self, prediction, ground_truth):
        """
        Check if prediction matches ground truth
        Can be improved with semantic similarity
        """
        # Normalize
        pred = prediction.lower().strip()
        truth = ground_truth.lower().strip()
        
        # Exact match
        if pred == truth:
            return True
        
        # Key phrase matching (for classifications)
        truth_keywords = set(truth.split())
        pred_keywords = set(pred.split())
        
        # If most keywords match
        overlap = len(truth_keywords & pred_keywords)
        if overlap / len(truth_keywords) > 0.7:
            return True
        
        return False
    
    def evaluate_bleu_score(self, test_loader):
        """
        BLEU score for caption generation
        """
        from nltk.translate.bleu_score import corpus_bleu
        
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="BLEU Evaluation"):
                images = batch['image'].to(self.device)
                prompts = batch['prompt']
                true_responses = batch['response']
                
                generated = self.model.generate(
                    images=images,
                    prompts=prompts,
                    max_new_tokens=100
                )
                
                for pred, truth in zip(generated, true_responses):
                    hypotheses.append(pred.split())
                    references.append([truth.split()])
        
        bleu_score = corpus_bleu(references, hypotheses)
        
        return bleu_score
    
    def evaluate_eo_terminology(self, test_loader):
        """
        Check if model uses correct EO terminology
        """
        eo_terms = {
            'ndvi', 'evi', 'ndwi', 'lst', 'sar', 'spectral',
            'backscatter', 'reflectance', 'vegetation', 'urban',
            'forest', 'water', 'agriculture', 'multispectral',
            'temporal', 'resolution', 'sentinel', 'landsat',
            'resourcesat', 'cartosat', 'risat'
        }
        
        term_counts = defaultdict(int)
        total_responses = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Terminology Check"):
                images = batch['image'].to(self.device)
                prompts = batch['prompt']
                
                generated = self.model.generate(
                    images=images,
                    prompts=prompts,
                    max_new_tokens=150
                )
                
                for response in generated:
                    response_lower = response.lower()
                    for term in eo_terms:
                        if term in response_lower:
                            term_counts[term] += 1
                    total_responses += 1
        
        # Average terms per response
        avg_terms = sum(term_counts.values()) / total_responses
        
        return {
            'avg_eo_terms_per_response': avg_terms,
            'term_distribution': dict(term_counts),
            'total_responses': total_responses
        }
    
    def evaluate_satellite_specific(self, test_loader, satellite_type):
        """
        Evaluate on specific satellite data (RESOURCESAT, CARTOSAT, etc.)
        """
        results = {
            'satellite': satellite_type,
            'total': 0,
            'correct_satellite_id': 0,
            'samples': []
        }
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"{satellite_type} Evaluation"):
                images = batch['image'].to(self.device)
                
                # Ask: "What satellite captured this image?"
                prompts = ["What satellite or sensor captured this image?"] * len(images)
                
                generated = self.model.generate(
                    images=images,
                    prompts=prompts,
                    max_new_tokens=50
                )
                
                for response in generated:
                    response_lower = response.lower()
                    
                    if satellite_type.lower() in response_lower:
                        results['correct_satellite_id'] += 1
                    
                    results['total'] += 1
                    results['samples'].append(response)
        
        results['accuracy'] = results['correct_satellite_id'] / results['total']
        
        return results
    
    def comprehensive_evaluation(self, test_loaders):
        """
        Run all evaluation metrics
        
        Args:
            test_loaders: Dict of test data loaders
                {'general': loader, 'resourcesat': loader, 'cartosat': loader}
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION")
        print("="*70)
        
        results = {}
        
        # 1. VQA Accuracy
        print("\n1. Visual Question Answering Accuracy...")
        vqa_results = self.evaluate_vqa_accuracy(test_loaders['general'])
        results['vqa_accuracy'] = vqa_results['accuracy']
        print(f"   VQA Accuracy: {vqa_results['accuracy']:.2%}")
        
        # 2. BLEU Score
        print("\n2. BLEU Score (Caption Quality)...")
        bleu = self.evaluate_bleu_score(test_loaders['general'])
        results['bleu_score'] = bleu
        print(f"   BLEU Score: {bleu:.4f}")
        
        # 3. EO Terminology Usage
        print("\n3. EO Terminology Usage...")
        term_results = self.evaluate_eo_terminology(test_loaders['general'])
        results['eo_terminology'] = term_results
        print(f"   Avg EO terms per response: {term_results['avg_eo_terms_per_response']:.2f}")
        
        # 4. Satellite-Specific
        for satellite in ['resourcesat', 'cartosat', 'risat']:
            if satellite in test_loaders:
                print(f"\n4. {satellite.upper()} Specific Evaluation...")
                sat_results = self.evaluate_satellite_specific(
                    test_loaders[satellite],
                    satellite
                )
                results[f'{satellite}_accuracy'] = sat_results['accuracy']
                print(f"   Satellite ID Accuracy: {sat_results['accuracy']:.2%}")
        
        # Save results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("Results saved to: evaluation_results.json")
        print("="*70)
        
        return results


# Example usage
if __name__ == "__main__":
    from architecture import ISROMultimodalGPT
    from isro_data_loader import create_isro_dataloader
    
    # Load model
    model = ISROMultimodalGPT.from_pretrained("models/final_model")
    model = model.to('cuda')
    
    # Create test loaders
    test_loaders = {
        'general': create_isro_dataloader(
            "data/test/general",
            "data/test/general.json",
            batch_size=8
        ),
        'resourcesat': create_isro_dataloader(
            "data/test/resourcesat",
            "data/test/resourcesat.json",
            satellite_type='RESOURCESAT',
            batch_size=8
        ),
        'cartosat': create_isro_dataloader(
            "data/test/cartosat",
            "data/test/cartosat.json",
            satellite_type='CARTOSAT',
            batch_size=8
        )
    }
    
    # Evaluate
    evaluator = ISROModelEvaluator(model, model.tokenizer)
    results = evaluator.comprehensive_evaluation(test_loaders)
```

---

## 2. QUALITATIVE EVALUATION

```python
# qualitative_eval.py
"""
Human-in-the-loop evaluation
Generate samples for expert review
"""

import torch
from PIL import Image
import json

class QualitativeEvaluator:
    """
    Generate samples for human expert evaluation
    """
    
    def __init__(self, model, output_dir='evaluation_samples'):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_evaluation_samples(self, test_images, num_samples=50):
        """
        Generate diverse samples for expert review
        
        Args:
            test_images: List of image paths
            num_samples: Number of samples to generate
        """
        
        # Different types of questions
        question_templates = [
            "What type of land cover is visible in this image?",
            "Identify the main features in this satellite image.",
            "What changes are visible compared to a previous observation?",
            "Estimate the vegetation health in this region.",
            "Describe the urban infrastructure visible.",
            "What water bodies are present?",
            "Classify the agricultural pattern.",
            "Identify any environmental concerns.",
        ]
        
        samples = []
        
        for i, image_path in enumerate(test_images[:num_samples]):
            # Load image
            image = Image.open(image_path)
            
            # Generate responses for different questions
            sample = {
                'image': image_path,
                'responses': {}
            }
            
            for question in question_templates:
                response = self.model.generate(
                    images=[image],
                    prompts=[question],
                    max_new_tokens=150
                )[0]
                
                sample['responses'][question] = response
            
            samples.append(sample)
            
            # Save individual sample
            with open(f"{self.output_dir}/sample_{i}.json", 'w') as f:
                json.dump({
                    'image': image_path,
                    'responses': sample['responses']
                }, f, indent=2)
        
        # Save all samples
        with open(f"{self.output_dir}/all_samples.json", 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"Generated {len(samples)} evaluation samples")
        print(f"Saved to: {self.output_dir}")
        
        return samples
    
    def create_evaluation_spreadsheet(self, samples):
        """
        Create spreadsheet for expert annotation
        
        Experts rate each response 1-5:
        1 - Completely wrong
        2 - Mostly wrong
        3 - Partially correct
        4 - Mostly correct
        5 - Perfect
        """
        import pandas as pd
        
        rows = []
        for sample in samples:
            for question, response in sample['responses'].items():
                rows.append({
                    'Image': sample['image'],
                    'Question': question,
                    'Model Response': response,
                    'Expert Rating (1-5)': '',
                    'Expert Feedback': '',
                    'Correct Response': ''
                })
        
        df = pd.DataFrame(rows)
        df.to_excel(f"{self.output_dir}/expert_evaluation.xlsx", index=False)
        
        print(f"Expert evaluation spreadsheet created")
        print(f"File: {self.output_dir}/expert_evaluation.xlsx")
```

---

## 3. DEPLOYMENT ARCHITECTURE

### 3.1 API Server

```python
# api_server.py
"""
FastAPI server for ISRO Multimodal GPT
Production-ready deployment
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import base64
from typing import Optional
import uvicorn

app = FastAPI(
    title="ISRO Multimodal GPT API",
    description="Visual Question Answering for Earth Observation",
    version="1.0.0"
)

# Global model (loaded once at startup)
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    """Load model at startup"""
    global model
    
    print("Loading ISRO Multimodal GPT...")
    from architecture import ISROMultimodalGPT
    
    model = ISROMultimodalGPT.from_pretrained("models/production")
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    max_length: int = Form(default=256),
    temperature: float = Form(default=0.7)
):
    """
    Analyze satellite image and answer question
    
    Args:
        image: Satellite image file
        question: Question about the image
        max_length: Maximum response length
        temperature: Sampling temperature
    
    Returns:
        JSON with analysis
    """
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Generate response
        with torch.no_grad():
            response = model.generate(
                images=[pil_image],
                prompts=[question],
                max_new_tokens=max_length,
                temperature=temperature
            )[0]
        
        return JSONResponse({
            "success": True,
            "question": question,
            "analysis": response,
            "metadata": {
                "image_size": pil_image.size,
                "model": "ISRO Multimodal GPT v1.0"
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_analyze")
async def batch_analyze(
    images: list[UploadFile] = File(...),
    questions: list[str] = Form(...)
):
    """
    Analyze multiple images
    """
    if len(images) != len(questions):
        raise HTTPException(400, "Number of images must match number of questions")
    
    results = []
    
    for img_file, question in zip(images, questions):
        image_data = await img_file.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        with torch.no_grad():
            response = model.generate(
                images=[pil_image],
                prompts=[question],
                max_new_tokens=256
            )[0]
        
        results.append({
            "image": img_file.filename,
            "question": question,
            "analysis": response
        })
    
    return JSONResponse({
        "success": True,
        "results": results
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # Single worker due to GPU
    )
```

### 3.2 Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download model (or mount as volume)
# RUN python3 download_model.py

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "api_server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  isro-multimodal-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3.3 Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isro-multimodal-gpt
spec:
  replicas: 1
  selector:
    matchLabels:
      app: isro-multimodal-gpt
  template:
    metadata:
      labels:
        app: isro-multimodal-gpt
    spec:
      containers:
      - name: api
        image: isro-multimodal-gpt:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: Service
metadata:
  name: isro-multimodal-service
spec:
  selector:
    app: isro-multimodal-gpt
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 4. WEB INTERFACE

```python
# streamlit_app.py
"""
Streamlit web interface for ISRO Multimodal GPT
User-friendly interface for non-technical users
"""

import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="ISRO Multimodal GPT",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ ISRO Earth Observation Analysis")
st.markdown("**AI-Powered Satellite Image Analysis**")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", "http://localhost:8000")
    max_length = st.slider("Max Response Length", 50, 500, 256)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    st.header("About")
    st.info("""
    This tool uses AI to analyze satellite imagery from:
    - RESOURCESAT (LISS-III, LISS-IV)
    - CARTOSAT (PAN, MX)
    - RISAT (SAR)
    
    Ask questions about land cover, features, changes, etc.
    """)

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Satellite Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.header("Analysis")
    
    # Question templates
    question_type = st.selectbox(
        "Select Question Type",
        [
            "Custom Question",
            "Land Cover Classification",
            "Feature Identification",
            "Change Detection",
            "Vegetation Health",
            "Water Bodies",
            "Urban Analysis"
        ]
    )
    
    # Question templates
    templates = {
        "Land Cover Classification": "What type of land cover is visible in this satellite image?",
        "Feature Identification": "Identify and describe the main features in this image.",
        "Change Detection": "What changes are visible in this temporal analysis?",
        "Vegetation Health": "Assess the vegetation health and coverage in this region.",
        "Water Bodies": "Identify and describe the water bodies in this image.",
        "Urban Analysis": "Analyze the urban infrastructure and development patterns."
    }
    
    if question_type == "Custom Question":
        question = st.text_area("Enter your question:", height=100)
    else:
        question = st.text_area(
            "Question:",
            value=templates.get(question_type, ""),
            height=100
        )
    
    if st.button("🚀 Analyze", type="primary"):
        if not uploaded_file:
            st.error("Please upload an image first")
        elif not question:
            st.error("Please enter a question")
        else:
            with st.spinner("Analyzing satellite image..."):
                try:
                    # Send to API
                    files = {"image": uploaded_file.getvalue()}
                    data = {
                        "question": question,
                        "max_length": max_length,
                        "temperature": temperature
                    }
                    
                    response = requests.post(
                        f"{api_url}/analyze",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("Analysis Complete!")
                        st.markdown("### Response:")
                        st.write(result['analysis'])
                        
                        with st.expander("Metadata"):
                            st.json(result['metadata'])
                    else:
                        st.error(f"Error: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Batch processing
st.markdown("---")
st.header("📊 Batch Processing")

batch_files = st.file_uploader(
    "Upload multiple images",
    type=['tif', 'tiff', 'png', 'jpg'],
    accept_multiple_files=True
)

if batch_files:
    st.write(f"Uploaded {len(batch_files)} images")
    
    batch_question = st.text_input("Question for all images")
    
    if st.button("Process Batch"):
        with st.spinner(f"Processing {len(batch_files)} images..."):
            # Process each image
            results = []
            for file in batch_files:
                # Call API
                pass
            
            st.success("Batch processing complete!")

# Run: streamlit run streamlit_app.py
```

---

## 5. PERFORMANCE OPTIMIZATION

```python
# optimize_model.py
"""
Model optimization for production deployment
"""

import torch
from transformers import AutoModelForCausalLM

def quantize_model(model_path, output_path):
    """
    Quantize model to int8 for faster inference
    """
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    model.save_pretrained(output_path)
    print(f"Quantized model saved to {output_path}")

def optimize_for_onnx(model_path, output_path):
    """
    Convert to ONNX for deployment on various platforms
    """
    import onnx
    from torch.onnx import export
    
    model = torch.load(model_path)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['embeddings'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'embeddings': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to {output_path}")
```

---

## DEPLOYMENT CHECKLIST

```markdown
PRE-DEPLOYMENT:
✅ Model trained and validated
✅ Evaluation metrics meet requirements
✅ Model quantized/optimized
✅ API tested locally
✅ Docker image built
✅ Documentation complete

DEPLOYMENT:
✅ Server infrastructure ready
✅ GPU resources allocated
✅ SSL certificates configured
✅ Monitoring setup (Prometheus/Grafana)
✅ Logging configured
✅ Backup strategy in place

POST-DEPLOYMENT:
✅ Health checks passing
✅ Load testing completed
✅ User training conducted
✅ Feedback mechanism in place
✅ Maintenance schedule defined
```

---

This completes the evaluation and deployment guide!
