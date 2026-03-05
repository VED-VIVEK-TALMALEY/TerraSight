"""
Download Sample Satellite Imagery for Baseline Testing
This script downloads pre-processed satellite images from public sources
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def setup_directories():
    """Create necessary directories"""
    base_dir = Path('.')
    
    dirs = [
        'data/raw',
        'data/processed',
        'data/annotations',
        'outputs/visualizations',
        'results/baseline',
        'models'
    ]
    
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")
    return base_dir

def download_eurosat_samples():
    """
    Download EuroSAT sample images
    EuroSAT is a Sentinel-2 based land cover classification dataset
    """
    print("\n" + "="*60)
    print("DOWNLOADING EUROSAT SAMPLE IMAGES")
    print("="*60)
    
    # We'll download a small sample from HuggingFace
    from datasets import load_dataset
    
    try:
        print("\nDownloading EuroSAT dataset (this may take a few minutes)...")
        dataset = load_dataset("tanganke/eurosat", split="train[:100]")
        
        # Save images locally
        data_dir = Path('data/raw/eurosat')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(dataset)} images to {data_dir}...")
        
        for idx, item in enumerate(dataset):
            # Get image and label
            image = item['image']
            label = item['label']
            
            # Label names
            label_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 
                          'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
                          'Residential', 'River', 'SeaLake']
            
            label_name = label_names[label]
            
            # Save image
            image_path = data_dir / f"{idx:04d}_{label_name}.jpg"
            image.save(image_path)
            
            if (idx + 1) % 10 == 0:
                print(f"  Saved {idx + 1}/100 images...")
        
        print(f"\n✓ Downloaded 100 EuroSAT images to {data_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading EuroSAT: {e}")
        print("Will try alternative source...")
        return False

def download_sample_images_manual():
    """
    Download individual sample satellite images
    Backup method if HuggingFace fails
    """
    print("\n" + "="*60)
    print("DOWNLOADING SAMPLE SATELLITE IMAGES")
    print("="*60)
    
    # Sample images from various sources (placeholder URLs)
    samples = [
        {
            'url': 'https://raw.githubusercontent.com/EuroSAT/EuroSAT/master/samples/Forest_1.jpg',
            'filename': 'sample_forest.jpg',
            'category': 'forest'
        },
        # Add more sample URLs here
    ]
    
    data_dir = Path('data/raw/samples')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading sample images to {data_dir}...")
    
    for sample in samples:
        try:
            destination = data_dir / sample['filename']
            print(f"\nDownloading {sample['filename']}...")
            download_file(sample['url'], destination)
            print(f"✓ Saved to {destination}")
        except Exception as e:
            print(f"✗ Failed to download {sample['filename']}: {e}")
    
    return True

def create_sample_images():
    """
    Create synthetic sample images for testing
    Use this as last resort if downloads fail
    """
    print("\n" + "="*60)
    print("CREATING SYNTHETIC TEST IMAGES")
    print("="*60)
    
    from PIL import Image
    import numpy as np
    
    data_dir = Path('data/raw/synthetic')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create different types of synthetic images
    image_types = {
        'forest': (34, 139, 34),      # Forest green
        'urban': (128, 128, 128),     # Gray
        'water': (0, 119, 190),       # Blue
        'agriculture': (255, 215, 0), # Golden
        'desert': (237, 201, 175),    # Sandy
    }
    
    print(f"\nCreating synthetic images in {data_dir}...")
    
    for name, base_color in image_types.items():
        # Create 224x224 image with some noise
        img_array = np.ones((224, 224, 3), dtype=np.uint8)
        
        for i in range(3):
            img_array[:, :, i] = base_color[i]
        
        # Add some random noise
        noise = np.random.randint(-30, 30, (224, 224, 3), dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_array)
        img.save(data_dir / f'synthetic_{name}.jpg')
        print(f"  ✓ Created {name} image")
    
    print(f"\n✓ Created 5 synthetic test images")
    return True

def main():
    """Main download function"""
    print("="*60)
    print("SATELLITE IMAGE DOWNLOAD SCRIPT")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Try downloading EuroSAT first
    print("\nAttempting to download real satellite imagery...")
    
    try:
        success = download_eurosat_samples()
        if success:
            print("\n✓ Successfully downloaded satellite imagery!")
            print("\nYou can find images in:")
            print("  data/raw/eurosat/")
            return
    except:
        print("EuroSAT download failed, trying alternatives...")
    
    # Fallback: Create synthetic images
    print("\nCreating synthetic test images as backup...")
    create_sample_images()
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check data/raw/ for downloaded images")
    print("2. Run baseline model testing")
    print("3. Evaluate results")

if __name__ == "__main__":
    # Install datasets library if needed
    try:
        import datasets
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'datasets'])
    
    main()