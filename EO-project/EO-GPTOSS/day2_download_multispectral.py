"""
Download Sentinel-2 Multispectral Satellite Imagery
Day 2 - Problem Demonstration

This script downloads actual 13-band Sentinel-2 data for analysis
"""

import numpy as np
from pathlib import Path
import requests
from PIL import Image
import json
from tqdm import tqdm

def download_sentinel2_sample():
    """
    Download sample Sentinel-2 multispectral imagery
    
    We'll use the Sentinel-2 Cloud-Free Mosaic dataset from HuggingFace
    which includes all 13 spectral bands
    """
    print("="*60)
    print("DOWNLOADING SENTINEL-2 MULTISPECTRAL DATA")
    print("="*60)
    
    # Try to use datasets library
    try:
        from datasets import load_dataset
        
        print("\nAttempting to download BigEarthNet subset...")
        print("(This dataset has all 13 Sentinel-2 bands)")
        
        # Load a small subset
        dataset = load_dataset(
            "blanchon/BigEarthNet",
            split="train[:50]"  # Just 50 images for Day 2
        )
        
        print(f"\n✓ Downloaded {len(dataset)} multispectral images")
        
        # Save to organized directory
        output_dir = Path('data/raw/sentinel2_multispectral')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing multispectral data")):
            # BigEarthNet provides bands separately
            sample_info = {
                'id': idx,
                'bands': {}
            }
            
            # Save each band
            for band_name, band_data in item.items():
                if isinstance(band_data, Image.Image):
                    band_path = output_dir / f"sample_{idx:04d}_{band_name}.tif"
                    band_data.save(band_path)
                    sample_info['bands'][band_name] = str(band_path)
            
            metadata.append(sample_info)
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved {len(metadata)} multispectral samples to {output_dir}")
        return True
        
    except Exception as e:
        print(f"\nBigEarthNet download failed: {e}")
        print("Falling back to synthetic multispectral generation...")
        return False

def create_synthetic_multispectral():
    """
    Create synthetic multispectral data for demonstration
    
    This simulates what real Sentinel-2 data looks like:
    - 13 spectral bands
    - Different reflectance patterns for different land covers
    """
    print("\n" + "="*60)
    print("CREATING SYNTHETIC MULTISPECTRAL DATA")
    print("="*60)
    
    output_dir = Path('data/raw/sentinel2_multispectral')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sentinel-2 band information
    bands_info = {
        'B01': {'name': 'Coastal Aerosol', 'wavelength': 443, 'resolution': 60},
        'B02': {'name': 'Blue', 'wavelength': 490, 'resolution': 10},
        'B03': {'name': 'Green', 'wavelength': 560, 'resolution': 10},
        'B04': {'name': 'Red', 'wavelength': 665, 'resolution': 10},
        'B05': {'name': 'Red Edge 1', 'wavelength': 705, 'resolution': 20},
        'B06': {'name': 'Red Edge 2', 'wavelength': 740, 'resolution': 20},
        'B07': {'name': 'Red Edge 3', 'wavelength': 783, 'resolution': 20},
        'B08': {'name': 'NIR', 'wavelength': 842, 'resolution': 10},
        'B8A': {'name': 'NIR Narrow', 'wavelength': 865, 'resolution': 20},
        'B09': {'name': 'Water Vapor', 'wavelength': 945, 'resolution': 60},
        'B10': {'name': 'SWIR Cirrus', 'wavelength': 1375, 'resolution': 60},
        'B11': {'name': 'SWIR 1', 'wavelength': 1610, 'resolution': 20},
        'B12': {'name': 'SWIR 2', 'wavelength': 2190, 'resolution': 20},
    }
    
    # Land cover spectral signatures (reflectance values 0-1)
    spectral_signatures = {
        'forest': {
            'B02': 0.03, 'B03': 0.04, 'B04': 0.03,  # Low visible (absorbed by chlorophyll)
            'B05': 0.10, 'B06': 0.15, 'B07': 0.20,  # Red edge transition
            'B08': 0.45, 'B8A': 0.45,               # High NIR (plant cells reflect)
            'B11': 0.25, 'B12': 0.15                # Medium SWIR
        },
        'water': {
            'B02': 0.15, 'B03': 0.12, 'B04': 0.08,  # Some visible reflection
            'B05': 0.04, 'B06': 0.02, 'B07': 0.01,  # Absorbs red edge
            'B08': 0.01, 'B8A': 0.01,               # Very low NIR (water absorbs)
            'B11': 0.00, 'B12': 0.00                # No SWIR (complete absorption)
        },
        'urban': {
            'B02': 0.12, 'B03': 0.15, 'B04': 0.18,  # Moderate visible
            'B05': 0.20, 'B06': 0.22, 'B07': 0.24,  # Increasing with wavelength
            'B08': 0.28, 'B8A': 0.28,               # Moderate NIR
            'B11': 0.32, 'B12': 0.35                # High SWIR (concrete/asphalt)
        },
        'agriculture': {
            'B02': 0.05, 'B03': 0.06, 'B04': 0.05,  # Low visible (crops)
            'B05': 0.12, 'B06': 0.18, 'B07': 0.25,  # Red edge (healthy vegetation)
            'B08': 0.40, 'B8A': 0.40,               # High NIR (vegetation)
            'B11': 0.22, 'B12': 0.12                # Medium-low SWIR
        },
        'bare_soil': {
            'B02': 0.10, 'B03': 0.13, 'B04': 0.16,  # Moderate visible
            'B05': 0.18, 'B06': 0.20, 'B07': 0.22,  # Gradual increase
            'B08': 0.24, 'B8A': 0.24,               # Moderate NIR
            'B11': 0.28, 'B12': 0.30                # High SWIR
        }
    }
    
    print("\nCreating synthetic multispectral images...")
    print("Land cover types: forest, water, urban, agriculture, bare_soil")
    
    metadata = []
    image_size = 64  # 64x64 pixels
    
    for land_cover, signature in spectral_signatures.items():
        for sample_num in range(10):  # 10 samples per land cover
            
            sample_id = len(metadata)
            sample_info = {
                'id': sample_id,
                'land_cover': land_cover,
                'bands': {},
                'spectral_indices': {}
            }
            
            # Create each band with the land cover's spectral signature
            for band_id, band_spec in bands_info.items():
                
                # Base reflectance from signature
                if band_id in signature:
                    base_value = signature[band_id]
                else:
                    # Interpolate for missing bands
                    base_value = 0.15
                
                # Add some spatial variation (noise)
                band_array = np.random.normal(
                    base_value, 
                    0.05,  # Standard deviation
                    size=(image_size, image_size)
                )
                
                # Clip to valid reflectance range [0, 1]
                band_array = np.clip(band_array, 0, 1)
                
                # Convert to uint16 (common for satellite imagery)
                # Scale 0-1 to 0-10000 (typical satellite data range)
                band_uint16 = (band_array * 10000).astype(np.uint16)
                
                # Save as numpy array
                band_path = output_dir / f"sample_{sample_id:04d}_{band_id}.npy"
                np.save(band_path, band_uint16)
                
                sample_info['bands'][band_id] = str(band_path)
            
            # Compute spectral indices from the synthetic data
            # Load the bands we just created
            red = np.load(sample_info['bands']['B04']) / 10000.0
            green = np.load(sample_info['bands']['B03']) / 10000.0
            nir = np.load(sample_info['bands']['B08']) / 10000.0
            swir1 = np.load(sample_info['bands']['B11']) / 10000.0
            
            # NDVI = (NIR - Red) / (NIR + Red)
            ndvi = (nir - red) / (nir + red + 1e-8)
            sample_info['spectral_indices']['NDVI'] = float(np.mean(ndvi))
            
            # NDWI = (Green - NIR) / (Green + NIR)
            ndwi = (green - nir) / (green + nir + 1e-8)
            sample_info['spectral_indices']['NDWI'] = float(np.mean(ndwi))
            
            # NDBI = (SWIR - NIR) / (SWIR + NIR)
            ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)
            sample_info['spectral_indices']['NDBI'] = float(np.mean(ndbi))
            
            metadata.append(sample_info)
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created {len(metadata)} synthetic multispectral samples")
    print(f"  Saved to: {output_dir}")
    
    # Print example spectral indices
    print("\n" + "="*60)
    print("EXAMPLE SPECTRAL INDICES")
    print("="*60)
    
    for land_cover in spectral_signatures.keys():
        samples = [m for m in metadata if m['land_cover'] == land_cover]
        avg_ndvi = np.mean([s['spectral_indices']['NDVI'] for s in samples])
        avg_ndwi = np.mean([s['spectral_indices']['NDWI'] for s in samples])
        avg_ndbi = np.mean([s['spectral_indices']['NDBI'] for s in samples])
        
        print(f"\n{land_cover.upper()}:")
        print(f"  NDVI (vegetation): {avg_ndvi:+.3f}")
        print(f"  NDWI (water):      {avg_ndwi:+.3f}")
        print(f"  NDBI (built-up):   {avg_ndbi:+.3f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("NDVI: >0.3 = vegetation, <0.2 = bare soil/urban")
    print("NDWI: >0.0 = water, <0.0 = land")
    print("NDBI: >0.0 = urban/built-up, <0.0 = vegetation")
    
    return True

def main():
    """Main data acquisition function for Day 2"""
    
    print("="*60)
    print("DAY 2: MULTISPECTRAL DATA ACQUISITION")
    print("="*60)
    
    # Try real data first
    success = download_sentinel2_sample()
    
    # If real data fails, use synthetic
    if not success:
        create_synthetic_multispectral()
    
    print("\n" + "="*60)
    print("DATA ACQUISITION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Create RGB and false-color composites")
    print("2. Test baseline model on different band combinations")
    print("3. Compare results to demonstrate information loss")

if __name__ == "__main__":
    main()