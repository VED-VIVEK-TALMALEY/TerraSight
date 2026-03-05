"""
GraphQL Backend for Multispectral VLM Web App
FastAPI + Strawberry GraphQL

Endpoints:
- Query: getSatelliteAnalysis(lat, lon)
- Returns: Both baseline and trained model results
"""

import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from typing import Optional, List
import asyncio

# Import your model
from day4_multimodal_model import MultispectralVLM
from transformers import BlipProcessor, BlipForConditionalGeneration

# GraphQL Types
@strawberry.type
class SpectralIndices:
    ndvi: float
    ndwi: float
    ndbi: float

@strawberry.type
class ModelResult:
    caption: str
    keywords: List[str]
    ndvi_predicted: Optional[float]
    processing_time: float

@strawberry.type
class SatelliteAnalysis:
    latitude: float
    longitude: float
    location_name: str
    spectral_indices: SpectralIndices
    baseline_result: ModelResult
    trained_result: ModelResult
    improvement_keywords: int

# Model Manager (Singleton)
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading models...")
        
        # Load trained model
        self.trained_model = MultispectralVLM(use_lora=True, lora_rank=8)
        checkpoint = torch.load('checkpoints/best_model.pt', map_location=self.device)
        self.trained_model.load_state_dict(checkpoint['model_state_dict'])
        self.trained_model = self.trained_model.to(self.device)
        self.trained_model.eval()
        
        # Load baseline
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()
        
        print("✓ Models loaded")
        self._initialized = True
    
    def generate_synthetic_multispectral(self, lat, lon):
        """
        Generate synthetic multispectral data based on coordinates
        
        In production, this would fetch real Sentinel-2 data from API
        For demo, we generate realistic synthetic data
        """
        # Simple heuristic based on lat/lon
        # You can replace this with actual API calls
        
        # Determine land cover type based on location
        # This is simplified - in production use actual land cover maps
        if abs(lat) < 10 and abs(lon) < 10:  # Equatorial (forest)
            land_cover = 'forest'
            ndvi = 0.85
            ndwi = -0.35
            ndbi = -0.28
        elif abs(lat) > 60:  # Polar (bare/ice)
            land_cover = 'bare_soil'
            ndvi = 0.15
            ndwi = -0.20
            ndbi = 0.10
        elif abs(lon) > 100:  # Asia (agriculture)
            land_cover = 'agriculture'
            ndvi = 0.78
            ndwi = -0.42
            ndbi = -0.30
        else:  # Default urban
            land_cover = 'urban'
            ndvi = 0.25
            ndwi = -0.30
            ndbi = 0.08
        
        # Generate 13-band data based on land cover
        spectral_signatures = {
            'forest': {
                'B02': 0.03, 'B03': 0.04, 'B04': 0.03,
                'B05': 0.10, 'B06': 0.15, 'B07': 0.20,
                'B08': 0.45, 'B8A': 0.45,
                'B11': 0.25, 'B12': 0.15
            },
            'urban': {
                'B02': 0.12, 'B03': 0.15, 'B04': 0.18,
                'B05': 0.20, 'B06': 0.22, 'B07': 0.24,
                'B08': 0.28, 'B8A': 0.28,
                'B11': 0.32, 'B12': 0.35
            },
            'agriculture': {
                'B02': 0.05, 'B03': 0.06, 'B04': 0.05,
                'B05': 0.12, 'B06': 0.18, 'B07': 0.25,
                'B08': 0.40, 'B8A': 0.40,
                'B11': 0.22, 'B12': 0.12
            },
            'bare_soil': {
                'B02': 0.10, 'B03': 0.13, 'B04': 0.16,
                'B05': 0.18, 'B06': 0.20, 'B07': 0.22,
                'B08': 0.24, 'B8A': 0.24,
                'B11': 0.28, 'B12': 0.30
            }
        }
        
        signature = spectral_signatures.get(land_cover, spectral_signatures['urban'])
        
        # Create 64x64 image with spatial variation
        band_ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                    'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        
        bands = []
        for band_id in band_ids:
            if band_id in signature:
                base_value = signature[band_id]
            else:
                base_value = 0.15
            
            # Add spatial noise
            band_array = np.random.normal(base_value, 0.05, size=(64, 64))
            band_array = np.clip(band_array, 0, 1)
            bands.append(band_array)
        
        multi_image = np.stack(bands, axis=0).astype(np.float32)
        
        return torch.from_numpy(multi_image), ndvi, ndwi, ndbi, land_cover
    
    def create_rgb_from_multispectral(self, multi_image):
        """Create RGB PIL image for baseline"""
        from PIL import Image
        
        red = multi_image[3]
        green = multi_image[2]
        blue = multi_image[1]
        
        def normalize(band):
            band = np.clip(band, 0, 1)
            return (band * 255).astype(np.uint8)
        
        rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=2)
        return Image.fromarray(rgb)
    
    def generate_trained(self, multi_image):
        """Generate with trained model"""
        import time
        import re
        
        start = time.time()
        
        with torch.no_grad():
            visual_embeddings = self.trained_model.encode_image(
                multi_image.unsqueeze(0).to(self.device)
            )
            
            prompt = "This satellite image shows"
            prompt_tokens = self.trained_model.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.device)
            
            generated_ids = prompt_tokens.input_ids.clone()
            
            for _ in range(50):
                text_emb = self.trained_model.language_model.get_input_embeddings()(generated_ids)
                combined = torch.cat([visual_embeddings, text_emb], dim=1)
                
                outputs = self.trained_model.language_model(
                    inputs_embeds=combined,
                    attention_mask=torch.ones(combined.shape[:2], device=self.device)
                )
                
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                if next_token.item() == self.trained_model.tokenizer.eos_token_id:
                    break
            
            caption = self.trained_model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        processing_time = time.time() - start
        
        # Extract keywords
        keywords = []
        if 'NDVI' in caption.upper():
            keywords.append('NDVI')
        if 'NIR' in caption.upper() or 'infrared' in caption.lower():
            keywords.append('NIR')
        if 'SWIR' in caption.upper():
            keywords.append('SWIR')
        if 'reflectance' in caption.lower():
            keywords.append('REFLECTANCE')
        
        # Extract NDVI value
        ndvi_predicted = None
        match = re.search(r'NDVI[:\s]+([0-9.]+)', caption, re.IGNORECASE)
        if match:
            try:
                ndvi_predicted = float(match.group(1))
            except:
                pass
        
        return caption, keywords, ndvi_predicted, processing_time
    
    def generate_baseline(self, rgb_image):
        """Generate with baseline"""
        import time
        
        start = time.time()
        
        inputs = self.blip_processor(rgb_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=50)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        
        processing_time = time.time() - start
        
        # Extract keywords (usually none for baseline)
        keywords = []
        if 'NDVI' in caption.upper():
            keywords.append('NDVI')
        if 'NIR' in caption.upper():
            keywords.append('NIR')
        
        return caption, keywords, None, processing_time

# Initialize model manager
model_manager = ModelManager()

# GraphQL Resolvers
@strawberry.type
class Query:
    @strawberry.field
    async def get_satellite_analysis(
        self, 
        latitude: float, 
        longitude: float
    ) -> SatelliteAnalysis:
        """
        Analyze satellite imagery at given coordinates
        """
        # Get location name (simple approximation)
        if abs(latitude) < 10 and abs(longitude) < 10:
            location_name = "Equatorial Region"
        elif abs(latitude) > 60:
            location_name = "Polar Region"
        elif abs(longitude) > 100:
            location_name = "Asia-Pacific"
        else:
            location_name = f"Location ({latitude:.2f}°, {longitude:.2f}°)"
        
        # Generate/fetch multispectral data
        multi_image, ndvi, ndwi, ndbi, land_cover = model_manager.generate_synthetic_multispectral(
            latitude, longitude
        )
        
        # Create RGB for baseline
        rgb_image = model_manager.create_rgb_from_multispectral(multi_image.numpy())
        
        # Run both models
        trained_caption, trained_kw, trained_ndvi, trained_time = model_manager.generate_trained(multi_image)
        baseline_caption, baseline_kw, baseline_ndvi, baseline_time = model_manager.generate_baseline(rgb_image)
        
        # Create results
        return SatelliteAnalysis(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            spectral_indices=SpectralIndices(
                ndvi=ndvi,
                ndwi=ndwi,
                ndbi=ndbi
            ),
            baseline_result=ModelResult(
                caption=baseline_caption,
                keywords=baseline_kw,
                ndvi_predicted=baseline_ndvi,
                processing_time=baseline_time
            ),
            trained_result=ModelResult(
                caption=trained_caption,
                keywords=trained_kw,
                ndvi_predicted=trained_ndvi,
                processing_time=trained_time
            ),
            improvement_keywords=len(trained_kw) - len(baseline_kw)
        )

# Create GraphQL schema
schema = strawberry.Schema(query=Query)

# Create FastAPI app
app = FastAPI(
    title="Multispectral VLM API",
    description="GraphQL API for Multispectral Vision-Language Model",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GraphQL route
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

# Health check
@app.get("/")
async def root():
    return {
        "message": "Multispectral VLM API",
        "status": "running",
        "graphql_endpoint": "/graphql"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Multispectral VLM API...")
    print("GraphQL endpoint: http://localhost:8000/graphql")
    uvicorn.run(app, host="0.0.0.0", port=8000)
