# DAY 3 - MULTISPECTRAL VISION-LANGUAGE ARCHITECTURE
## Design Document & Implementation Specification

**Date:** February 12, 2026  
**Objective:** Design novel architecture preserving 13-band multispectral information

---

## 1. PROBLEM SUMMARY (From Day 2)

### Empirically Validated Issues:

✗ **RGB-only models lose 88.5% of spectral information** (13 bands → 3 bands)  
✗ **62.5% contradiction rate** when baseline model sees different band combinations  
✗ **Agriculture misclassified** in RGB, correctly identified in NIR  
✗ **Water detection less confident** in RGB vs SWIR  
✗ **No access to NDVI, NDWI, NDBI** (require non-RGB bands)

### Requirements for Solution:

✓ Process 13-band input natively  
✓ Learn which bands matter for different tasks  
✓ Preserve spectral information through the pipeline  
✓ Integrate with existing language models  
✓ Train efficiently on 4GB GPU

---

## 2. ARCHITECTURE OPTIONS ANALYSIS

### Option A: Spectral Projection Layer

**Concept:** Learn a projection from 13 bands to 3-channel equivalent that ViT can process

```
Input: 13 bands (64×64×13)
         ↓
Learnable Conv2D: 13→3 channels
         ↓
Standard ViT (pretrained on RGB)
         ↓
Language Model
```

**Pros:**
- Simple to implement
- Can use pretrained ViT unchanged
- Minimal training required

**Cons:**
- Still bottlenecks to 3 channels
- May not preserve all spectral info
- Projection is task-agnostic (same for all questions)

**Verdict:** ❌ Too similar to RGB conversion - defeats the purpose

---

### Option B: Multi-Path Architecture

**Concept:** Separate encoders for different spectral groups, then fuse

```
Input: 13 bands
         ↓
Split into groups:
├── RGB (B02, B03, B04) → ViT-RGB
├── NIR (B05-B08, B8A) → ViT-NIR  
└── SWIR (B11, B12)    → ViT-SWIR
         ↓
Fusion Layer (Attention)
         ↓
Language Model
```

**Pros:**
- Preserves spectral groups separately
- Can use pretrained ViT for RGB path
- Explicit handling of different wavelength ranges

**Cons:**
- 3× memory requirement (3 ViT models)
- Complex fusion mechanism needed
- Doesn't fit in 4GB VRAM

**Verdict:** ❌ Too memory-intensive for our hardware

---

### Option C: Spectral Attention Architecture (RECOMMENDED)

**Concept:** Single ViT with modified input layer + spectral attention

```
Input: 13 bands (64×64×13)
         ↓
Modified ViT Input Layer: 13→768 dims per patch
         ↓
Spectral Attention Module (learns band importance)
         ↓
Standard ViT Transformer Blocks (frozen or LoRA)
         ↓
Vision-Language Projection
         ↓
Language Model (Llama-2-7B with LoRA)
```

**Pros:**
- Preserves all 13 bands
- Learns task-specific band importance
- Memory efficient (single model)
- Can leverage pretrained ViT (except first layer)

**Cons:**
- Need to modify ViT input layer
- Spectral attention adds parameters
- Requires careful initialization

**Verdict:** ✓ **SELECTED** - Best balance of performance and efficiency

---

## 3. DETAILED ARCHITECTURE SPECIFICATION

### 3.1 Overall Pipeline

```python
class MultispectralVLM:
    """
    Multispectral Vision-Language Model
    
    Components:
    1. Spectral Vision Encoder (modified ViT)
    2. Vision-Language Projection
    3. Language Model (Llama-2-7B)
    """
    
    def __init__(self):
        # Vision encoder for 13-band input
        self.vision_encoder = SpectralViT(
            in_channels=13,        # 13 Sentinel-2 bands
            patch_size=8,          # 8×8 patches (64×64 image = 8×8 grid)
            embed_dim=768,         # Standard ViT dimension
            depth=12,              # 12 transformer blocks
            num_heads=12,          # 12 attention heads
            use_spectral_attention=True
        )
        
        # Projection from vision to language space
        self.vision_projection = nn.Linear(768, 4096)  # To Llama embedding size
        
        # Language model (frozen initially)
        self.language_model = Llama2_7B(
            use_lora=True,         # Parameter-efficient fine-tuning
            lora_rank=8,           # LoRA rank
            lora_alpha=16
        )
    
    def forward(self, multispectral_image, text_prompt):
        # Encode 13-band image
        visual_features = self.vision_encoder(multispectral_image)
        # Shape: (batch, num_patches, 768)
        
        # Project to language space
        visual_embeddings = self.vision_projection(visual_features)
        # Shape: (batch, num_patches, 4096)
        
        # Tokenize text
        text_tokens = self.tokenizer(text_prompt)
        text_embeddings = self.language_model.embed_tokens(text_tokens)
        
        # Concatenate visual + text embeddings
        combined = torch.cat([visual_embeddings, text_embeddings], dim=1)
        
        # Generate response
        output = self.language_model.generate(combined)
        
        return output
```

### 3.2 Spectral Vision Transformer (SpectralViT)

**Key Innovation:** Modified input layer + spectral attention

```python
class SpectralViT(nn.Module):
    """
    Vision Transformer adapted for multispectral input
    """
    
    def __init__(self, in_channels=13, patch_size=8, embed_dim=768):
        super().__init__()
        
        # Modified patch embedding for 13 channels
        self.patch_embed = SpectralPatchEmbedding(
            in_channels=13,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Positional embeddings (standard)
        num_patches = (64 // patch_size) ** 2  # 8×8 = 64 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Spectral attention module (our innovation)
        self.spectral_attention = SpectralAttentionModule(
            num_bands=13,
            embed_dim=embed_dim
        )
        
        # Standard transformer blocks (can use pretrained)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=12)
            for _ in range(12)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x shape: (batch, 13, 64, 64)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Apply spectral attention (our innovation)
        x = self.spectral_attention(x)
        
        # Transformer blocks
        for block in self.transformer:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        return x
```

### 3.3 Spectral Patch Embedding (Novel Component)

**Challenge:** Standard ViT expects 3 channels, we have 13

**Solution:** Learn separate embeddings per band, then combine

```python
class SpectralPatchEmbedding(nn.Module):
    """
    Convert 13-band patches to embeddings
    
    Approach: Per-band convolution → concatenate → project to embed_dim
    """
    
    def __init__(self, in_channels=13, patch_size=8, embed_dim=768):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_bands = in_channels
        
        # Option 1: Shared convolution across all bands
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Option 2: Per-band convolution (more expressive)
        # self.band_convs = nn.ModuleList([
        #     nn.Conv2d(1, embed_dim // num_bands, patch_size, patch_size)
        #     for _ in range(num_bands)
        # ])
        # self.fusion = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x shape: (batch, 13, 64, 64)
        
        # Option 1: Direct projection
        x = self.projection(x)  # (batch, embed_dim, 8, 8)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (batch, embed_dim, 64)
        x = x.transpose(1, 2)  # (batch, 64, embed_dim)
        
        return x
```

### 3.4 Spectral Attention Module (Core Innovation)

**Purpose:** Learn which spectral bands are important for different image regions

**Mechanism:** Attention over spectral dimension

```python
class SpectralAttentionModule(nn.Module):
    """
    Learn to attend to relevant spectral bands
    
    Idea: Different regions need different bands
    - Vegetation: High weight on NIR, Red edge
    - Water: High weight on SWIR
    - Urban: Balanced weights
    """
    
    def __init__(self, num_bands=13, embed_dim=768):
        super().__init__()
        
        # Query: What do I need to know?
        self.query = nn.Linear(embed_dim, embed_dim)
        
        # Key: What band information is available?
        self.key = nn.Linear(embed_dim, embed_dim)
        
        # Value: The actual band information
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Band-specific embeddings
        self.band_embeddings = nn.Parameter(
            torch.randn(num_bands, embed_dim)
        )
        
        self.scale = embed_dim ** -0.5
    
    def forward(self, x):
        # x shape: (batch, num_patches+1, embed_dim)
        
        batch_size, num_patches, embed_dim = x.shape
        
        # Generate queries from current representations
        Q = self.query(x)  # What information do I need?
        
        # Generate keys from band embeddings
        K = self.key(self.band_embeddings.unsqueeze(0))  # What's available?
        K = K.expand(batch_size, -1, -1)
        
        # Compute attention weights: which bands matter?
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        # Shape: (batch, num_patches, num_bands)
        
        # Apply attention to refine features
        # This allows the model to emphasize relevant spectral information
        V = self.value(x)
        attended = x + 0.1 * torch.matmul(attn_weights, K)  # Residual connection
        
        return attended
```

**Intuition:**
- For vegetation patches: Model learns to weight NIR (B08) and Red Edge (B05-B07) highly
- For water patches: Model learns to weight SWIR (B11, B12) highly  
- For urban patches: Model learns balanced weights or emphasizes SWIR

---

## 4. TRAINING STRATEGY

### 4.1 Three-Stage Training (LLaVA-inspired)

**Stage 1: Spectral Projection Pre-training** (Day 4 morning)
- **Freeze:** Language model, ViT transformer blocks
- **Train:** Spectral patch embedding, spectral attention, projection layer
- **Data:** Image-caption pairs (100-150 samples)
- **Objective:** Learn to map 13-band images to language-compatible features
- **Duration:** 5-10 epochs, ~1-2 hours
- **Hardware:** RTX 3050 4GB (fits with frozen LLM)

**Stage 2: Vision-Language Alignment** (Day 4 afternoon)
- **Freeze:** Language model (but use LoRA)
- **Train:** Entire vision encoder + projection
- **Data:** VQA pairs with spectral-specific questions
- **Objective:** Align visual understanding with language generation
- **Duration:** 10-15 epochs, ~2-3 hours
- **Hardware:** RTX 3050 4GB with gradient checkpointing

**Stage 3: End-to-End Fine-tuning** (Day 5)
- **Freeze:** Nothing (but use LoRA on LLM)
- **Train:** All components with LoRA
- **Data:** Full dataset with diverse tasks
- **Objective:** Joint optimization for best performance
- **Duration:** 15-20 epochs, ~3-4 hours
- **Hardware:** RTX 3050 4GB with mixed precision + LoRA

### 4.2 Memory Optimization for 4GB GPU

**Techniques:**

1. **LoRA (Low-Rank Adaptation):**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                    # Rank (lower = less memory, faster)
    lora_alpha=16,          # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, lora_config)
```

**Memory Savings:** 
- 7B model with LoRA: ~4GB (vs ~14GB full fine-tuning)
- Only trains ~0.5% of parameters

2. **Gradient Checkpointing:**
```python
model.gradient_checkpointing_enable()
```
**Trade-off:** 20% slower, 40% less memory

3. **Mixed Precision (FP16):**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
```

4. **Batch Size = 1:**
- Process one image at a time
- Accumulate gradients over 4-8 steps (effective batch size 4-8)

**Total Memory Budget:**
```
Vision Encoder (modified ViT):     ~800 MB
Projection Layer:                  ~50 MB
Language Model (7B with LoRA):     ~3.2 GB
Optimizer States:                  ~200 MB
Activations (batch=1, checkpt):    ~500 MB
────────────────────────────────────────────
Total:                             ~4.75 GB

With cache clearing:               ~3.8 GB peak ✓
```

### 4.3 Dataset Requirements

**Minimum for Proof-of-Concept:**
- 150-200 image-text pairs
- Coverage: All 5 land cover types
- Annotations mentioning spectral features

**Data Sources:**

1. **Our Synthetic Data (50 samples):**
   - Expand with augmentation (rotation, flipping) → 100 samples
   
2. **Manual Annotation (50 samples):**
   - Write spectral-aware captions
   - Example: "Dense forest with high NIR reflectance (NDVI ~0.8)"
   
3. **Semi-Automatic (50 samples):**
   - Use BLIP-2 for initial caption
   - Manually add spectral information
   - Example: "Green fields" → "Agricultural cropland showing strong red-edge response"

**Annotation Template:**
```json
{
  "image_id": "sample_0001",
  "land_cover": "forest",
  "bands": ["B01", "B02", ..., "B12"],
  "captions": [
    "Dense forest with high vegetation index (NDVI: 0.76)",
    "Healthy vegetation showing strong NIR reflection",
    "Forest area with low SWIR values indicating high moisture"
  ],
  "qa_pairs": [
    {
      "question": "What type of land cover is this?",
      "answer": "Dense forest with healthy vegetation"
    },
    {
      "question": "What is the NDVI value?",
      "answer": "High NDVI around 0.76, indicating dense vegetation"
    }
  ],
  "spectral_indices": {
    "NDVI": 0.756,
    "NDWI": -0.364,
    "NDBI": -0.287
  }
}
```

---

## 5. IMPLEMENTATION PLAN

### Day 3 Tasks:

**Morning (4 hours):**
- ✅ Complete architecture design (this document)
- ✅ Implement core components (SpectralViT, SpectralAttention)
- ✅ Test forward pass with dummy data

**Afternoon (4 hours):**
- ✅ Integrate with language model
- ✅ Implement training loop skeleton
- ✅ Verify memory fits in 4GB GPU
- ✅ Start dataset annotation

### Day 4 Tasks:

**Morning:**
- Stage 1 training (spectral projection)
- Monitor convergence

**Afternoon:**
- Stage 2 training (vision-language alignment)
- Initial evaluation

### Day 5 Tasks:

**Morning:**
- Stage 3 training (end-to-end)
- Final evaluation

**Afternoon:**
- Compare with baseline
- Document improvements

---

## 6. EVALUATION METRICS

### Quantitative Metrics:

1. **Land Cover Accuracy:**
   - Baseline (RGB): ~60%
   - Target (Multispectral): >80%

2. **Consistency Across Composites:**
   - Baseline: 37.5% (contradicts 62.5% of the time)
   - Target: >90% (should give same answer regardless of visualization)

3. **Spectral Terminology Usage:**
   - Baseline: 0% (never mentions NIR, SWIR, NDVI)
   - Target: >50% (mentions spectral features when relevant)

4. **NDVI/NDWI/NDBI Value Accuracy:**
   - Can the model estimate these values from the image?
   - Target: Within ±0.1 of ground truth

### Qualitative Metrics:

1. **Spectral Reasoning:**
   - Does model explain WHY it classified something?
   - Example: "Forest because high NIR reflection" ✓

2. **Band-Specific Insights:**
   - Does model mention specific bands?
   - Example: "Strong absorption in Red band (B04), high reflectance in NIR (B08)"

3. **Vegetation Health Assessment:**
   - Can model distinguish healthy vs stressed vegetation?
   - Requires NIR and Red Edge information

---

## 7. EXPECTED CHALLENGES

### Challenge 1: Spectral Attention Learning

**Risk:** Model might ignore spectral attention, collapse to using only RGB-like bands

**Mitigation:**
- Add regularization encouraging diverse band usage
- Use spectral-specific questions during training
- Monitor attention weights during training

### Challenge 2: Language Model Hallucination

**Risk:** Model might generate plausible-sounding but incorrect spectral information

**Mitigation:**
- Use temperature=0.1 for factual questions
- Validate outputs against computed spectral indices
- Include "I don't know" as valid response

### Challenge 3: Limited Training Data

**Risk:** 150-200 samples may not be enough for robust learning

**Mitigation:**
- Heavy data augmentation (rotation, flip, crops)
- Transfer learning from pretrained components
- Focus on key land cover types

### Challenge 4: Memory Constraints

**Risk:** Model + data + gradients might exceed 4GB

**Mitigation:**
- Gradient checkpointing
- Batch size = 1
- Gradient accumulation
- Monitor with nvidia-smi during training

---

## 8. SUCCESS CRITERIA

**Minimum Viable Product (MVP):**
- ✓ Model accepts 13-band input
- ✓ Generates captions mentioning spectral features
- ✓ >70% land cover accuracy
- ✓ <30% contradiction rate (vs 62.5% baseline)

**Target Performance:**
- ✓ >80% land cover accuracy
- ✓ <10% contradiction rate
- ✓ >50% spectral terminology usage
- ✓ Accurate NDVI/NDWI estimation (±0.1)

**Stretch Goals:**
- ✓ >90% land cover accuracy
- ✓ Generalizes to unseen satellite sensors
- ✓ Can explain reasoning with spectral references

---

## 9. ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT (64×64×13)                          │
│              13 Sentinel-2 Spectral Bands                    │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│            SPECTRAL PATCH EMBEDDING                          │
│    Conv2D: 13 channels → 768 dims per 8×8 patch            │
│             Output: 64 patches × 768                         │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│          SPECTRAL ATTENTION MODULE ⭐ INNOVATION             │
│   Learn band importance per patch (vegetation→NIR,          │
│   water→SWIR, urban→balanced)                               │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│       VISION TRANSFORMER (12 layers, pretrained)             │
│          Standard self-attention blocks                      │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│         VISION-LANGUAGE PROJECTION                           │
│           Linear: 768 → 4096 (Llama space)                  │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│      LANGUAGE MODEL (Llama-2-7B with LoRA)                  │
│   Frozen base, trainable LoRA adapters (rank=8)            │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                OUTPUT TEXT GENERATION                        │
│  "Dense forest (NDVI: 0.76) with high NIR reflectance"     │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. CODE STRUCTURE

```
day3_architecture/
├── models/
│   ├── spectral_vit.py          # SpectralViT implementation
│   ├── spectral_attention.py    # SpectralAttentionModule
│   ├── patch_embedding.py       # SpectralPatchEmbedding
│   └── multimodal_model.py      # Complete MultispectralVLM
├── training/
│   ├── train_stage1.py          # Projection pre-training
│   ├── train_stage2.py          # Vision-language alignment
│   ├── train_stage3.py          # End-to-end fine-tuning
│   └── utils.py                 # Training utilities
├── data/
│   ├── dataset.py               # MultispectralDataset class
│   ├── augmentation.py          # Data augmentation
│   └── annotation_tool.py       # Manual annotation interface
├── evaluation/
│   ├── evaluate.py              # Evaluation metrics
│   └── visualize_attention.py   # Visualize spectral attention
└── config/
    ├── model_config.yaml        # Architecture parameters
    └── training_config.yaml     # Training hyperparameters
```

---

**END OF ARCHITECTURE DESIGN**

Next: Implementation!
