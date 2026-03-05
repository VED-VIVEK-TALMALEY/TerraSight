"""
Generate Band Composites from Multispectral Data
Day 2 - Problem Demonstration

This script creates:
1. RGB composites (standard true-color)
2. False-color composites (NIR, SWIR emphasis)
3. Spectral index visualizations
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

class CompositeGenerator:
    """Generate various band composites from multispectral data"""
    
    def __init__(self, data_dir='data/raw/sentinel2_multispectral'):
        self.data_dir = Path(data_dir)
        
        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded metadata for {len(self.metadata)} samples")
    
    def normalize_band(self, band_array, percentile_clip=True):
        """
        Normalize band values to 0-255 range for visualization
        
        Args:
            band_array: Raw band data
            percentile_clip: If True, clip to 2nd and 98th percentile
        """
        if percentile_clip:
            # Clip outliers for better visualization
            p2, p98 = np.percentile(band_array, (2, 98))
            band_array = np.clip(band_array, p2, p98)
        
        # Normalize to 0-1
        band_min = band_array.min()
        band_max = band_array.max()
        
        if band_max > band_min:
            normalized = (band_array - band_min) / (band_max - band_min)
        else:
            normalized = np.zeros_like(band_array)
        
        # Scale to 0-255
        return (normalized * 255).astype(np.uint8)
    
    def load_band(self, sample_info, band_id):
        """Load a specific band from disk"""
        band_path = sample_info['bands'][band_id]
        
        # Load numpy array
        band_data = np.load(band_path)
        
        return band_data
    
    def create_rgb_composite(self, sample_info):
        """
        Create standard RGB composite (True Color)
        Uses: Red (B04), Green (B03), Blue (B02)
        """
        red = self.load_band(sample_info, 'B04')
        green = self.load_band(sample_info, 'B03')
        blue = self.load_band(sample_info, 'B02')
        
        # Normalize each band
        r = self.normalize_band(red)
        g = self.normalize_band(green)
        b = self.normalize_band(blue)
        
        # Stack into RGB image
        rgb = np.stack([r, g, b], axis=2)
        
        return Image.fromarray(rgb)
    
    def create_false_color_nir(self, sample_info):
        """
        Create NIR false-color composite (Vegetation emphasis)
        Uses: NIR (B08), Red (B04), Green (B03)
        
        Interpretation:
        - Bright red = Healthy vegetation (high NIR)
        - Dark red/brown = Stressed vegetation
        - Blue/cyan = Water, urban
        """
        nir = self.load_band(sample_info, 'B08')
        red = self.load_band(sample_info, 'B04')
        green = self.load_band(sample_info, 'B03')
        
        # Map NIR to Red channel, Red to Green, Green to Blue
        r = self.normalize_band(nir)
        g = self.normalize_band(red)
        b = self.normalize_band(green)
        
        rgb = np.stack([r, g, b], axis=2)
        
        return Image.fromarray(rgb)
    
    def create_false_color_swir(self, sample_info):
        """
        Create SWIR false-color composite (Water/Moisture emphasis)
        Uses: SWIR (B11), NIR (B08), Red (B04)
        
        Interpretation:
        - Dark blue = Water (absorbs SWIR)
        - Bright areas = Dry soil, urban
        - Green = Vegetation
        """
        swir = self.load_band(sample_info, 'B11')
        nir = self.load_band(sample_info, 'B08')
        red = self.load_band(sample_info, 'B04')
        
        r = self.normalize_band(swir)
        g = self.normalize_band(nir)
        b = self.normalize_band(red)
        
        rgb = np.stack([r, g, b], axis=2)
        
        return Image.fromarray(rgb)
    
    def create_false_color_agriculture(self, sample_info):
        """
        Create Agriculture false-color composite
        Uses: SWIR (B11), NIR (B08), Blue (B02)
        
        Interpretation:
        - Pink/magenta = Crops, vegetation
        - Blue = Urban, bare soil
        - Dark = Water
        """
        swir = self.load_band(sample_info, 'B11')
        nir = self.load_band(sample_info, 'B08')
        blue = self.load_band(sample_info, 'B02')
        
        r = self.normalize_band(swir)
        g = self.normalize_band(nir)
        b = self.normalize_band(blue)
        
        rgb = np.stack([r, g, b], axis=2)
        
        return Image.fromarray(rgb)
    
    def create_spectral_index_visualization(self, sample_info, index_name='NDVI'):
        """
        Create visualization of spectral index
        
        Args:
            sample_info: Sample metadata
            index_name: 'NDVI', 'NDWI', or 'NDBI'
        """
        # Load required bands
        red = self.load_band(sample_info, 'B04') / 10000.0
        green = self.load_band(sample_info, 'B03') / 10000.0
        nir = self.load_band(sample_info, 'B08') / 10000.0
        swir = self.load_band(sample_info, 'B11') / 10000.0
        
        if index_name == 'NDVI':
            # NDVI = (NIR - Red) / (NIR + Red)
            index = (nir - red) / (nir + red + 1e-8)
            cmap = 'RdYlGn'  # Red to Yellow to Green
            vmin, vmax = -1, 1
            
        elif index_name == 'NDWI':
            # NDWI = (Green - NIR) / (Green + NIR)
            index = (green - nir) / (green + nir + 1e-8)
            cmap = 'BrBG'  # Brown to Blue to Green
            vmin, vmax = -1, 1
            
        elif index_name == 'NDBI':
            # NDBI = (SWIR - NIR) / (SWIR + NIR)
            index = (swir - nir) / (swir + nir + 1e-8)
            cmap = 'RdGy'  # Red to Gray
            vmin, vmax = -1, 1
        
        # Create colored visualization
        plt.figure(figsize=(6, 6))
        plt.imshow(index, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label=index_name)
        plt.title(f"{index_name} - {sample_info.get('land_cover', 'Unknown')}")
        plt.axis('off')
        
        # Convert to PIL Image
        plt.tight_layout()
        
        # Save to buffer and convert to PIL
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    def generate_all_composites(self, num_samples=20):
        """
        Generate all composite types for samples
        
        Args:
            num_samples: Number of samples to process (default 20)
        """
        print("="*60)
        print("GENERATING BAND COMPOSITES")
        print("="*60)
        
        # Create output directories
        output_base = Path('data/processed/composites')
        
        composite_types = [
            'rgb', 'false_color_nir', 'false_color_swir', 
            'false_color_agriculture', 'ndvi', 'ndwi', 'ndbi'
        ]
        
        for comp_type in composite_types:
            (output_base / comp_type).mkdir(parents=True, exist_ok=True)
        
        # Process samples
        samples_to_process = self.metadata[:num_samples]
        
        composite_info = []
        
        for sample_info in tqdm(samples_to_process, desc="Creating composites"):
            sample_id = sample_info['id']
            land_cover = sample_info.get('land_cover', 'unknown')
            
            info = {
                'sample_id': sample_id,
                'land_cover': land_cover,
                'composites': {}
            }
            
            # RGB (True Color)
            rgb_img = self.create_rgb_composite(sample_info)
            rgb_path = output_base / 'rgb' / f'sample_{sample_id:04d}_rgb.png'
            rgb_img.save(rgb_path)
            info['composites']['rgb'] = str(rgb_path)
            
            # False Color NIR
            fc_nir_img = self.create_false_color_nir(sample_info)
            fc_nir_path = output_base / 'false_color_nir' / f'sample_{sample_id:04d}_fc_nir.png'
            fc_nir_img.save(fc_nir_path)
            info['composites']['false_color_nir'] = str(fc_nir_path)
            
            # False Color SWIR
            fc_swir_img = self.create_false_color_swir(sample_info)
            fc_swir_path = output_base / 'false_color_swir' / f'sample_{sample_id:04d}_fc_swir.png'
            fc_swir_img.save(fc_swir_path)
            info['composites']['false_color_swir'] = str(fc_swir_path)
            
            # False Color Agriculture
            fc_ag_img = self.create_false_color_agriculture(sample_info)
            fc_ag_path = output_base / 'false_color_agriculture' / f'sample_{sample_id:04d}_fc_ag.png'
            fc_ag_img.save(fc_ag_path)
            info['composites']['false_color_agriculture'] = str(fc_ag_path)
            
            # NDVI visualization
            ndvi_img = self.create_spectral_index_visualization(sample_info, 'NDVI')
            ndvi_path = output_base / 'ndvi' / f'sample_{sample_id:04d}_ndvi.png'
            ndvi_img.save(ndvi_path)
            info['composites']['ndvi'] = str(ndvi_path)
            
            # NDWI visualization
            ndwi_img = self.create_spectral_index_visualization(sample_info, 'NDWI')
            ndwi_path = output_base / 'ndwi' / f'sample_{sample_id:04d}_ndwi.png'
            ndwi_img.save(ndwi_path)
            info['composites']['ndwi'] = str(ndwi_path)
            
            # NDBI visualization
            ndbi_img = self.create_spectral_index_visualization(sample_info, 'NDBI')
            ndbi_path = output_base / 'ndbi' / f'sample_{sample_id:04d}_ndbi.png'
            ndbi_img.save(ndbi_path)
            info['composites']['ndbi'] = str(ndbi_path)
            
            composite_info.append(info)
        
        # Save composite metadata
        with open(output_base / 'composite_metadata.json', 'w') as f:
            json.dump(composite_info, f, indent=2)
        
        print(f"\n✓ Generated {len(composite_types)} composite types for {len(composite_info)} samples")
        print(f"  Saved to: {output_base}")
        
        # Print summary
        print("\n" + "="*60)
        print("COMPOSITE TYPES GENERATED")
        print("="*60)
        print("\n1. RGB (True Color)")
        print("   - Standard red-green-blue")
        print("   - What human eyes see")
        
        print("\n2. False Color NIR")
        print("   - Vegetation appears RED")
        print("   - Shows vegetation health")
        
        print("\n3. False Color SWIR")
        print("   - Water appears DARK")
        print("   - Shows moisture content")
        
        print("\n4. False Color Agriculture")
        print("   - Crops appear PINK/MAGENTA")
        print("   - Optimized for farmland")
        
        print("\n5. NDVI (Vegetation Index)")
        print("   - Green = healthy vegetation")
        print("   - Red = bare soil/urban")
        
        print("\n6. NDWI (Water Index)")
        print("   - Blue = water")
        print("   - Brown = land")
        
        print("\n7. NDBI (Built-up Index)")
        print("   - Red = urban/built-up")
        print("   - Gray = vegetation")
        
        return composite_info

def main():
    """Main composite generation function"""
    
    print("="*60)
    print("DAY 2: COMPOSITE GENERATION")
    print("="*60)
    
    # Initialize generator
    generator = CompositeGenerator()
    
    # Generate all composites
    composite_info = generator.generate_all_composites(num_samples=20)
    
    print("\n" + "="*60)
    print("COMPOSITE GENERATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Test baseline model on RGB composites")
    print("2. Test baseline model on false-color composites")
    print("3. Compare results to demonstrate information loss")
    print("\nFiles saved in: data/processed/composites/")

if __name__ == "__main__":
    main()