# ISRO Problem Statement: Enhancing GPT-OSS with Multimodal Vision Capabilities for EO Data

## COMPLETE TECHNICAL PROPOSAL & IMPLEMENTATION GUIDE

---

## TABLE OF CONTENTS

1. [Problem Statement Analysis](#1-problem-statement-analysis)
2. [Proposed Solution Architecture](#2-proposed-solution-architecture)
3. [Technical Background](#3-technical-background)
4. [Multimodal Architecture Design](#4-multimodal-architecture-design)
5. [ISRO EO Data Integration](#5-isro-eo-data-integration)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Complete Code Implementation](#7-complete-code-implementation)
8. [Training & Fine-Tuning Strategy](#8-training--fine-tuning-strategy)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Challenges & Solutions](#11-challenges--solutions)
12. [Future Enhancements](#12-future-enhancements)

---

## 1. PROBLEM STATEMENT ANALYSIS

### 1.1 Understanding the Challenge

**ISRO's Requirements**:
```
┌─────────────────────────────────────────────────────────────┐
│ CURRENT STATE                                               │
│ - Open-source GPT models (text-only)                       │
│ - ISRO EO satellite imagery (visual data)                  │
│ - Disconnect between text models and image analysis        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ REQUIRED STATE                                              │
│ - Unified multimodal model                                  │
│ - Process both satellite images AND text                   │
│ - Answer questions about EO imagery                         │
│ - Generate reports from visual data                         │
│ - Domain-specific understanding of ISRO satellites          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 ISRO Satellite Context

**ISRO's Earth Observation Satellites**:
```
1. RESOURCESAT series (IRS)
   - High-resolution optical imagery
   - LISS-III, LISS-IV, AWiFS sensors
   - Agriculture, forestry, land use

2. CARTOSAT series
   - High-resolution stereo imaging
   - DEM generation, mapping
   - Urban planning, defense

3. RISAT (Radar Imaging Satellite)
   - All-weather SAR imaging
   - Day/night capability
   - Disaster monitoring

4. OCEANSAT series
   - Ocean monitoring
   - Coastal zone studies
   - Ocean color, SST

5. INSAT/GSAT series
   - Meteorological imaging
   - Weather forecasting
   - Climate monitoring
```

### 1.3 Use Cases

**Primary Applications**:
1. **Visual Question Answering (VQA)**
   - "What type of land cover is visible in this RESOURCESAT image?"
   - "Identify deforestation areas in this region"

2. **Image Captioning**
   - Generate descriptions of satellite scenes
   - Automated metadata generation

3. **Change Detection**
   - Compare temporal images
   - Report changes in infrastructure, vegetation

4. **Disaster Response**
   - Flood extent mapping
   - Damage assessment
   - Emergency report generation

5. **Report Generation**
   - Automated analysis reports
   - Multi-image synthesis
   - Technical documentation

---

## 2. PROPOSED SOLUTION ARCHITECTURE

### 2.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     MULTIMODAL GPT-OSS SYSTEM                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐         ┌──────────────┐                   │
│  │   VISION     │         │   LANGUAGE   │                   │
│  │   ENCODER    │────────▶│   MODEL      │                   │
│  │   (ViT)      │         │   (GPT-OSS)  │                   │
│  └──────────────┘         └──────────────┘                   │
│         ▲                         │                            │
│         │                         ▼                            │
│  ┌──────────────┐         ┌──────────────┐                   │
│  │ ISRO EO      │         │   TEXT       │                   │
│  │ SATELLITE    │         │   OUTPUT     │                   │
│  │ IMAGES       │         │   RESPONSE   │                   │
│  └──────────────┘         └──────────────┘                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘

COMPONENTS:
1. Vision Encoder: Processes satellite images → embeddings
2. Projection Layer: Maps image embeddings → text space
3. Language Model: Processes combined inputs → text output
4. EO-Specific Adapters: Domain knowledge for ISRO data
```

### 2.2 Model Selection Strategy

**Option 1: LLaVA-Style Architecture** (RECOMMENDED)
```
Vision: CLIP ViT-L/14 or ViT-H/14
Projection: Linear projection layer
Language: Llama-2-7B/13B or Mistral-7B
Training: Two-stage (pretrain + instruction tuning)
```

**Option 2: Flamingo-Style Architecture**
```
Vision: NFNet or Vision Transformer
Projection: Perceiver Resampler
Language: Any GPT-OSS model
Training: Interleaved image-text
```

**Option 3: BLIP-2 Architecture**
```
Vision: EVA-CLIP
Projection: Q-Former (Querying Transformer)
Language: OPT or Llama
Training: Bootstrap from frozen models
```

### 2.3 Technical Stack

```yaml
Core Components:
  - PyTorch 2.0+
  - Transformers (HuggingFace)
  - PEFT (LoRA for efficient training)
  - Vision Transformers (timm library)
  
Vision Processing:
  - CLIP (OpenAI)
  - SigLIP (Google)
  - DINOv2 (Meta)
  
Language Models:
  - Llama-2-7B/13B
  - Mistral-7B
  - Phi-2 (for lightweight deployment)
  
EO-Specific:
  - Rasterio (geospatial reading)
  - GDAL (format support)
  - Sentinelhub (if using Sentinel data)
  - Custom ISRO data loaders
```

---

## 3. TECHNICAL BACKGROUND

### 3.1 Multimodal Learning Fundamentals

**How Vision-Language Models Work**:

```python
# Conceptual flow
def multimodal_model(image, text_query):
    """
    Step 1: Encode image
    """
    image_features = vision_encoder(image)  # Shape: [N, 256, 1024]
    
    """
    Step 2: Project to language space
    """
    image_tokens = projection_layer(image_features)  # Shape: [N, 256, 4096]
    
    """
    Step 3: Combine with text
    """
    text_tokens = tokenizer(text_query)
    combined = concat(image_tokens, text_tokens)
    
    """
    Step 4: Generate response
    """
    response = language_model(combined)
    
    return response
```

**Key Concepts**:

1. **Vision Encoder**: Converts images to feature vectors
   - Uses self-attention (Vision Transformer)
   - Pre-trained on large datasets (ImageNet, LAION)
   - Output: Spatial feature maps

2. **Projection Layer**: Bridges vision and language
   - Linear projection (simple)
   - Q-Former (sophisticated)
   - Perceiver (flexible)

3. **Language Model**: Generates text from features
   - Auto-regressive decoder
   - Trained on text + image features
   - Output: Natural language

### 3.2 Vision Transformers (ViT)

```
IMAGE PROCESSING IN ViT:

Input Image (224x224x3)
    ↓
Split into patches (16x16)
    ↓
196 patches (14×14 grid)
    ↓
Flatten each patch → 768-dim vector
    ↓
Add positional embeddings
    ↓
Transformer encoder (12-24 layers)
    ↓
Output: 196 feature vectors + 1 CLS token
```

**For Satellite Imagery**:
```python
# Satellite images are often larger
Input: 512x512x3 (or multi-spectral)
Patch size: 32x32
Grid: 16x16 = 256 patches

# Multi-spectral adaptation
Input: 512x512x13 (Sentinel-2 all bands)
First layer: Conv to project 13→3 channels
Then standard ViT processing
```

### 3.3 Attention Mechanisms

```python
# Self-attention (simplified)
def attention(Q, K, V):
    """
    Q: Queries (what we're looking for)
    K: Keys (what's available)
    V: Values (actual content)
    """
    scores = Q @ K.T / sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output

# Cross-attention (vision-language)
def cross_attention(text_queries, image_keys, image_values):
    """
    Text asks questions about image
    """
    scores = text_queries @ image_keys.T / sqrt(d_k)
    weights = softmax(scores)
    output = weights @ image_values
    return output
```

---

## 4. MULTIMODAL ARCHITECTURE DESIGN

### 4.1 Complete Architecture (LLaVA-style)

```python
# architecture.py
"""
Complete multimodal architecture for ISRO EO data
Based on LLaVA with adaptations for satellite imagery
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    LlamaForCausalLM,
    LlamaTokenizer
)

class MultimodalProjector(nn.Module):
    """
    Projects vision features to language model space
    """
    def __init__(self, vision_hidden_size=1024, text_hidden_size=4096):
        super().__init__()
        
        # Simple linear projection
        self.linear = nn.Linear(vision_hidden_size, text_hidden_size)
        
        # Or use MLP for more capacity
        # self.mlp = nn.Sequential(
        #     nn.Linear(vision_hidden_size, text_hidden_size),
        #     nn.GELU(),
        #     nn.Linear(text_hidden_size, text_hidden_size)
        # )
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: [batch, num_patches, vision_dim]
        Returns:
            projected: [batch, num_patches, text_dim]
        """
        return self.linear(vision_features)


class ISROMultimodalGPT(nn.Module):
    """
    Multimodal GPT for ISRO Earth Observation data
    
    Architecture:
        Vision Encoder (CLIP ViT) → Projector → Language Model (Llama)
    """
    
    def __init__(
        self,
        vision_model_name="openai/clip-vit-large-patch14",
        language_model_name="meta-llama/Llama-2-7b-hf",
        freeze_vision=True,
        freeze_language=False
    ):
        super().__init__()
        
        print("Initializing ISRO Multimodal GPT...")
        
        # Vision encoder
        print(f"Loading vision model: {vision_model_name}")
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("Vision encoder frozen")
        
        # Language model
        print(f"Loading language model: {language_model_name}")
        self.language_model = LlamaForCausalLM.from_pretrained(
            language_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(language_model_name)
        
        if freeze_language:
            for param in self.language_model.parameters():
                param.requires_grad = False
            print("Language model frozen")
        
        # Projector (vision → language space)
        vision_hidden_size = self.vision_encoder.config.hidden_size
        language_hidden_size = self.language_model.config.hidden_size
        
        self.projector = MultimodalProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=language_hidden_size
        )
        
        print("✓ Model initialized")
        print(f"  Vision dim: {vision_hidden_size}")
        print(f"  Language dim: {language_hidden_size}")
    
    def encode_images(self, images):
        """
        Encode images to features
        
        Args:
            images: Tensor of shape [batch, 3, H, W]
        
        Returns:
            image_features: [batch, num_patches, hidden_dim]
        """
        # Process through vision encoder
        vision_outputs = self.vision_encoder(images, output_hidden_states=True)
        
        # Get patch features (exclude CLS token)
        image_features = vision_outputs.last_hidden_state[:, 1:, :]
        
        # Project to language space
        image_features = self.projector(image_features)
        
        return image_features
    
    def prepare_inputs(self, images, text_prompts):
        """
        Prepare multimodal inputs for the model
        
        Args:
            images: List of PIL Images or Tensors
            text_prompts: List of text strings
        
        Returns:
            input_ids: Combined image + text tokens
            attention_mask: Mask for the inputs
        """
        batch_size = len(images)
        
        # Encode images
        if isinstance(images[0], torch.Tensor):
            image_tensors = torch.stack(images)
        else:
            # Process PIL images
            image_tensors = self.vision_processor(
                images=images,
                return_tensors="pt"
            )['pixel_values']
        
        image_tensors = image_tensors.to(self.vision_encoder.device)
        image_features = self.encode_images(image_tensors)
        
        # Tokenize text
        text_inputs = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Combine: [image tokens] + [text tokens]
        # This is simplified - actual implementation needs careful token management
        
        return image_features, text_inputs
    
    def forward(self, images, text_prompts, labels=None):
        """
        Forward pass
        
        Args:
            images: Batch of images
            text_prompts: Batch of text prompts
            labels: Optional labels for training
        
        Returns:
            loss (if labels provided) or logits
        """
        # Get image and text features
        image_features, text_inputs = self.prepare_inputs(images, text_prompts)
        
        # Create embeddings for text
        text_embeds = self.language_model.get_input_embeddings()(
            text_inputs['input_ids']
        )
        
        # Concatenate image and text embeddings
        combined_embeds = torch.cat([image_features, text_embeds], dim=1)
        
        # Create attention mask
        batch_size = image_features.shape[0]
        num_image_tokens = image_features.shape[1]
        
        image_attention = torch.ones(
            batch_size, num_image_tokens,
            device=image_features.device
        )
        
        combined_attention = torch.cat([
            image_attention,
            text_inputs['attention_mask']
        ], dim=1)
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, images, prompts, max_new_tokens=256, **kwargs):
        """
        Generate text from images and prompts
        
        Args:
            images: Input images
            prompts: Text prompts
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            generated_texts: List of generated strings
        """
        # Prepare inputs
        image_features, text_inputs = self.prepare_inputs(images, prompts)
        
        # Get text embeddings
        text_embeds = self.language_model.get_input_embeddings()(
            text_inputs['input_ids']
        )
        
        # Combine
        combined_embeds = torch.cat([image_features, text_embeds], dim=1)
        
        # Attention mask
        batch_size = image_features.shape[0]
        num_image_tokens = image_features.shape[1]
        
        image_attention = torch.ones(
            batch_size, num_image_tokens,
            device=image_features.device
        )
        
        combined_attention = torch.cat([
            image_attention,
            text_inputs['attention_mask']
        ], dim=1)
        
        # Generate
        with torch.no_grad():
            outputs = self.language_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return generated_texts


class ISROEODataAugmentation(nn.Module):
    """
    EO-specific data augmentation and preprocessing
    Handles multi-spectral, SAR, and optical imagery
    """
    
    def __init__(self, num_input_channels=3):
        super().__init__()
        self.num_input_channels = num_input_channels
        
        # If multi-spectral, project to 3 channels for CLIP
        if num_input_channels > 3:
            self.channel_projection = nn.Conv2d(
                num_input_channels, 3,
                kernel_size=1, bias=False
            )
        else:
            self.channel_projection = None
    
    def normalize_eo_data(self, image, sensor_type='optical'):
        """
        Normalize EO data based on sensor type
        
        Args:
            image: Tensor [C, H, W]
            sensor_type: 'optical', 'sar', 'thermal'
        """
        if sensor_type == 'optical':
            # Standard normalization for optical
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        elif sensor_type == 'sar':
            # SAR-specific normalization
            # Log scale for backscatter
            image = torch.log1p(torch.clamp(image, min=0))
            mean = image.mean(dim=[1, 2], keepdim=True)
            std = image.std(dim=[1, 2], keepdim=True)
        
        elif sensor_type == 'thermal':
            # Thermal normalization
            mean = image.mean(dim=[1, 2], keepdim=True)
            std = image.std(dim=[1, 2], keepdim=True)
        
        normalized = (image - mean) / (std + 1e-6)
        return normalized
    
    def forward(self, image, sensor_type='optical'):
        """
        Process EO image
        """
        # Project multi-spectral to RGB if needed
        if self.channel_projection is not None:
            image = self.channel_projection(image)
        
        # Normalize based on sensor
        image = self.normalize_eo_data(image, sensor_type)
        
        return image


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = ISROMultimodalGPT(
        vision_model_name="openai/clip-vit-large-patch14",
        language_model_name="meta-llama/Llama-2-7b-hf"
    )
    
    # Test with dummy data
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_prompts = [
        "What type of land cover is visible in this satellite image?",
        "Identify the main features in this image."
    ]
    
    # Generate
    outputs = model.generate(dummy_images, dummy_prompts, max_new_tokens=100)
    
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i+1}: {dummy_prompts[i]}")
        print(f"Response: {output}")
```

---

## 5. ISRO EO DATA INTEGRATION

### 5.1 Data Loader for ISRO Satellites

```python
# isro_data_loader.py
"""
Custom data loader for ISRO satellite imagery
Supports RESOURCESAT, CARTOSAT, RISAT formats
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
import json

class ISROEODataset(Dataset):
    """
    Dataset for ISRO Earth Observation imagery
    
    Supports:
    - RESOURCESAT (LISS-III, LISS-IV, AWiFS)
    - CARTOSAT (PAN, MX)
    - RISAT (SAR)
    """
    
    def __init__(
        self,
        data_dir,
        annotation_file,
        satellite_type='RESOURCESAT',
        sensor='LISS-III',
        transform=None,
        max_samples=None
    ):
        """
        Args:
            data_dir: Directory with satellite images
            annotation_file: JSON with image-text pairs
            satellite_type: RESOURCESAT, CARTOSAT, RISAT
            sensor: Specific sensor
            transform: Image transformations
            max_samples: Limit dataset size
        """
        self.data_dir = data_dir
        self.satellite_type = satellite_type
        self.sensor = sensor
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        if max_samples:
            self.annotations = self.annotations[:max_samples]
        
        # Sensor specifications
        self.sensor_specs = self._get_sensor_specs()
        
        print(f"Loaded {len(self.annotations)} samples")
        print(f"Satellite: {satellite_type}, Sensor: {sensor}")
    
    def _get_sensor_specs(self):
        """Get sensor specifications"""
        specs = {
            'RESOURCESAT': {
                'LISS-III': {
                    'bands': 4,
                    'resolution': 23.5,  # meters
                    'wavelengths': {
                        'B2': '0.52-0.59',  # Green
                        'B3': '0.62-0.68',  # Red
                        'B4': '0.77-0.86',  # NIR
                        'B5': '1.55-1.70'   # SWIR
                    }
                },
                'LISS-IV': {
                    'bands': 3,
                    'resolution': 5.8,
                    'wavelengths': {
                        'B2': '0.52-0.59',
                        'B3': '0.62-0.68',
                        'B4': '0.77-0.86'
                    }
                },
                'AWiFS': {
                    'bands': 4,
                    'resolution': 56,
                    'wavelengths': {
                        'B2': '0.52-0.59',
                        'B3': '0.62-0.68',
                        'B4': '0.77-0.86',
                        'B5': '1.55-1.70'
                    }
                }
            },
            'CARTOSAT': {
                'PAN': {
                    'bands': 1,
                    'resolution': 2.5,
                    'wavelengths': {'PAN': '0.50-0.85'}
                },
                'MX': {
                    'bands': 4,
                    'resolution': 2.5,
                    'wavelengths': {
                        'B1': '0.45-0.52',
                        'B2': '0.52-0.59',
                        'B3': '0.62-0.68',
                        'B4': '0.77-0.86'
                    }
                }
            },
            'RISAT': {
                'SAR': {
                    'bands': 1,
                    'resolution': 1,  # varies
                    'type': 'C-band SAR'
                }
            }
        }
        
        return specs.get(self.satellite_type, {}).get(self.sensor, {})
    
    def __len__(self):
        return len(self.annotations)
    
    def load_geotiff(self, filepath):
        """
        Load GeoTIFF file
        
        Returns:
            image: numpy array [H, W, C]
            metadata: dict with georeferencing info
        """
        with rasterio.open(filepath) as src:
            # Read all bands
            image = src.read()  # Shape: [C, H, W]
            
            # Transpose to [H, W, C]
            image = np.transpose(image, (1, 2, 0))
            
            # Metadata
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'count': src.count,
                'dtype': src.dtypes[0]
            }
        
        return image, metadata
    
    def preprocess_multispectral(self, image):
        """
        Preprocess multi-spectral imagery
        
        Args:
            image: [H, W, C] numpy array
        
        Returns:
            processed: [3, H, W] tensor (RGB for CLIP)
        """
        # For RESOURCESAT LISS-III (4 bands: G, R, NIR, SWIR)
        if image.shape[-1] == 4:
            # Create false color composite: NIR-R-G
            rgb = np.stack([
                image[:, :, 2],  # NIR → R
                image[:, :, 1],  # Red → G
                image[:, :, 0]   # Green → B
            ], axis=-1)
        
        # For RESOURCESAT LISS-IV or CARTOSAT MX (3 bands: R, G, B)
        elif image.shape[-1] == 3:
            rgb = image
        
        # For CARTOSAT PAN or RISAT SAR (1 band)
        elif image.shape[-1] == 1:
            rgb = np.repeat(image, 3, axis=-1)
        
        else:
            # Default: use first 3 bands
            rgb = image[:, :, :3]
        
        # Normalize to 0-255
        rgb = self.normalize_to_uint8(rgb)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb.astype(np.uint8))
        
        return pil_image
    
    def normalize_to_uint8(self, image):
        """
        Normalize image to 0-255 range
        
        Handles different data types and value ranges
        """
        # Get min/max per channel
        for c in range(image.shape[-1]):
            channel = image[:, :, c]
            
            # Remove outliers (2nd and 98th percentile)
            p2, p98 = np.percentile(channel, (2, 98))
            
            # Clip and normalize
            channel = np.clip(channel, p2, p98)
            channel = (channel - p2) / (p98 - p2) * 255
            
            image[:, :, c] = channel
        
        return image
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            dict with 'image', 'prompt', 'response', 'metadata'
        """
        annotation = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, annotation['image_file'])
        
        if image_path.endswith('.tif') or image_path.endswith('.tiff'):
            image, geo_metadata = self.load_geotiff(image_path)
        else:
            image = np.array(Image.open(image_path))
            geo_metadata = {}
        
        # Preprocess based on sensor type
        if self.satellite_type in ['RESOURCESAT', 'CARTOSAT']:
            image = self.preprocess_multispectral(image)
        elif self.satellite_type == 'RISAT':
            # SAR-specific preprocessing
            image = self.preprocess_sar(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get text
        prompt = annotation.get('prompt', '')
        response = annotation.get('response', '')
        
        return {
            'image': image,
            'prompt': prompt,
            'response': response,
            'metadata': {
                'satellite': self.satellite_type,
                'sensor': self.sensor,
                'image_file': annotation['image_file'],
                **geo_metadata
            }
        }
    
    def preprocess_sar(self, image):
        """
        Preprocess SAR imagery
        
        SAR data requires special handling:
        - Log scaling for backscatter
        - Speckle filtering
        """
        # Convert to float
        image = image.astype(np.float32)
        
        # Log scale (common for SAR)
        image = np.log10(image + 1)
        
        # Normalize
        image = self.normalize_to_uint8(image)
        
        # Convert to PIL
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        return pil_image


def create_isro_dataloader(
    data_dir,
    annotation_file,
    batch_size=4,
    num_workers=4,
    **kwargs
):
    """
    Create DataLoader for ISRO data
    
    Args:
        data_dir: Image directory
        annotation_file: Annotations JSON
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        DataLoader
    """
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset
    dataset = ISROEODataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        transform=transform,
        **kwargs
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Example annotation format
"""
annotations.json:
[
    {
        "image_file": "RESOURCESAT_LISS3_20230615_scene1.tif",
        "prompt": "What type of land cover is visible in this image?",
        "response": "The image shows dense forest vegetation with high NDVI values, typical of tropical rainforest in the Western Ghats region."
    },
    {
        "image_file": "CARTOSAT_PAN_20230620_urban.tif",
        "prompt": "Identify the urban features in this image.",
        "response": "The image shows a densely populated urban area with organized road networks, residential blocks, and commercial buildings typical of a tier-2 Indian city."
    }
]
"""
```

### 5.2 Data Annotation Pipeline

```python
# annotation_tools.py
"""
Tools for creating annotations for ISRO EO data
Semi-automated annotation with human-in-the-loop
"""

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ISROAnnotationTool:
    """
    GUI tool for annotating ISRO satellite imagery
    """
    
    def __init__(self, image_dir, output_file="annotations.json"):
        self.image_dir = Path(image_dir)
        self.output_file = output_file
        self.annotations = []
        self.current_idx = 0
        
        # Get all images
        self.image_files = list(self.image_dir.glob("*.tif")) + \
                          list(self.image_dir.glob("*.tiff")) + \
                          list(self.image_dir.glob("*.png"))
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the annotation interface"""
        self.root = tk.Tk()
        self.root.title("ISRO EO Annotation Tool")
        
        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        
        # Prompt entry
        tk.Label(self.root, text="Question/Prompt:").pack()
        self.prompt_text = tk.Text(self.root, height=3, width=80)
        self.prompt_text.pack()
        
        # Response entry
        tk.Label(self.root, text="Answer/Response:").pack()
        self.response_text = tk.Text(self.root, height=5, width=80)
        self.response_text.pack()
        
        # Template buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        
        templates = [
            ("Land Cover", self.template_land_cover),
            ("Change Detection", self.template_change),
            ("Feature ID", self.template_feature),
        ]
        
        for text, command in templates:
            tk.Button(button_frame, text=text, command=command).pack(side=tk.LEFT)
        
        # Save button
        tk.Button(self.root, text="Save & Next", command=self.save_annotation).pack()
        
        # Load first image
        self.load_image()
    
    def load_image(self):
        """Load and display current image"""
        if self.current_idx >= len(self.image_files):
            self.finish()
            return
        
        image_path = self.image_files[self.current_idx]
        
        # Load and resize
        image = Image.open(image_path)
        image.thumbnail((800, 800))
        
        # Display
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        
        self.root.title(f"Annotating: {image_path.name} ({self.current_idx+1}/{len(self.image_files)})")
    
    def template_land_cover(self):
        """Insert land cover template"""
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", "What type of land cover is visible in this satellite image?")
    
    def template_change(self):
        """Insert change detection template"""
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", "What changes are visible between the two time periods?")
    
    def template_feature(self):
        """Insert feature identification template"""
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", "Identify and describe the main features in this image.")
    
    def save_annotation(self):
        """Save current annotation and move to next"""
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        response = self.response_text.get("1.0", tk.END).strip()
        
        if prompt and response:
            annotation = {
                "image_file": self.image_files[self.current_idx].name,
                "prompt": prompt,
                "response": response
            }
            self.annotations.append(annotation)
            
            # Clear fields
            self.prompt_text.delete("1.0", tk.END)
            self.response_text.delete("1.0", tk.END)
            
            # Next image
            self.current_idx += 1
            self.load_image()
    
    def finish(self):
        """Save all annotations and close"""
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"Saved {len(self.annotations)} annotations to {self.output_file}")
        self.root.destroy()
    
    def run(self):
        """Start the annotation tool"""
        self.root.mainloop()


# Automatic annotation generation (using existing models)
class AutoAnnotationGenerator:
    """
    Generate initial annotations using existing models
    These can then be refined by humans
    """
    
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate_caption(self, image_path):
        """Generate caption for an image"""
        image = Image.open(image_path)
        
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return caption
    
    def generate_vqa(self, image_path, question):
        """Answer a question about an image"""
        image = Image.open(image_path)
        
        inputs = self.processor(
            image,
            text=question,
            return_tensors="pt"
        ).to("cuda")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return answer
    
    def batch_annotate(self, image_dir, questions, output_file):
        """
        Generate annotations for all images
        
        Args:
            image_dir: Directory with images
            questions: List of questions to ask
            output_file: Output JSON file
        """
        image_files = list(Path(image_dir).glob("*.tif")) + \
                     list(Path(image_dir).glob("*.png"))
        
        annotations = []
        
        for image_file in image_files:
            print(f"Processing {image_file.name}...")
            
            for question in questions:
                answer = self.generate_vqa(image_file, question)
                
                annotations.append({
                    "image_file": image_file.name,
                    "prompt": question,
                    "response": answer
                })
        
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Generated {len(annotations)} annotations")


# Usage
if __name__ == "__main__":
    # Option 1: Manual annotation
    tool = ISROAnnotationTool(
        image_dir="data/isro_images",
        output_file="annotations.json"
    )
    tool.run()
    
    # Option 2: Auto-generate then refine
    # generator = AutoAnnotationGenerator()
    # generator.batch_annotate(
    #     image_dir="data/isro_images",
    #     questions=[
    #         "What type of land cover is visible?",
    #         "What are the main features?",
    #         "Is this urban or rural?"
    #     ],
    #     output_file="auto_annotations.json"
    # )
```

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Weeks 1-2)

```
Week 1: Environment & Data
├── Setup development environment
├── Install dependencies
├── Collect ISRO sample data
└── Create initial annotations (100-500 samples)

Week 2: Base Model
├── Implement multimodal architecture
├── Load pretrained components (CLIP + Llama)
├── Create data pipeline
└── Test forward pass
```

### Phase 2: Training (Weeks 3-5)

```
Week 3: Pretraining
├── Implement pretraining on image-text pairs
├── Train projection layer
├── Monitor convergence
└── Evaluate basic capabilities

Week 4: Instruction Tuning
├── Create instruction-following dataset
├── Fine-tune on VQA tasks
├── Implement LoRA for efficiency
└── Validate on held-out set

Week 5: EO-Specific Tuning
├── Fine-tune on ISRO-specific data
├── Add domain knowledge
├── Optimize for satellite imagery
└── Comprehensive evaluation
```

### Phase 3: Enhancement (Weeks 6-7)

```
Week 6: Advanced Features
├── Multi-image support (temporal analysis)
├── Specialized tasks (change detection)
├── Report generation
└── Integration with ISRO workflows

Week 7: Optimization
├── Model quantization
├── Inference optimization
├── API development
└── Documentation
```

### Phase 4: Deployment (Week 8)

```
Week 8: Production
├── Deploy model
├── Create user interface
├── Integration testing
└── Handoff to ISRO
```

---

## 7. COMPLETE CODE IMPLEMENTATION

Due to length constraints, I've provided the core architecture above. Let me create the complete training script:

```python
# train_multimodal.py - in next section
```

---

## 8. TRAINING & FINE-TUNING STRATEGY

### 8.1 Two-Stage Training (LLaVA Approach)

**Stage 1: Pretraining** (Align vision and language)
- Dataset: Large image-caption pairs (CC3M, LAION)
- Freeze: Vision encoder + Language model
- Train: Only projection layer
- Duration: 1 epoch on large dataset
- Goal: Learn to map images to language space

**Stage 2: Instruction Tuning** (Task-specific)
- Dataset: Instruction-following VQA
- Freeze: Vision encoder
- Train: Projection + Language model (LoRA)
- Duration: 3-5 epochs
- Goal: Follow instructions, answer questions

**Stage 3: ISRO-Specific** (Domain adaptation)
- Dataset: ISRO satellite imagery + annotations
- Freeze: Vision encoder
- Train: Projection + Language model (LoRA)
- Duration: 5-10 epochs
- Goal: EO domain expertise

This is a comprehensive foundation. Would you like me to continue with:
1. Complete training scripts
2. Evaluation framework
3. Deployment guide
4. Specific satellite sensor adaptations

The implementation is tailored for ISRO's needs with multi-spectral support, SAR handling, and domain-specific preprocessing.
