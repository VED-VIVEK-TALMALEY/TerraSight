#!/usr/bin/env python3
"""
ISRO Multimodal Project - Setup Verification Script
Run this after installation to ensure everything is working correctly.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version is 3.8+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_libraries():
    """Check all required libraries are installed"""
    print("\nChecking required libraries...")
    
    libraries = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'rasterio': 'Rasterio',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'tqdm': 'tqdm',
        'peft': 'PEFT',
        'accelerate': 'Accelerate',
    }
    
    all_good = True
    for module, name in libraries.items():
        try:
            imported = __import__(module)
            version = getattr(imported, '__version__', 'unknown')
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: Not installed")
            all_good = False
    
    return all_good

def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU availability...")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"    GPU Memory: {gpu_memory:.2f} GB")
            
            if gpu_memory < 8:
                print(f"    ⚠️ WARNING: GPU has less than 8GB memory")
                print(f"    You may need to use smaller models or batch sizes")
            
            # Test GPU computation
            test_tensor = torch.randn(100, 100).cuda()
            result = test_tensor @ test_tensor.T
            print(f"  ✓ GPU computation test passed")
            del test_tensor, result
            torch.cuda.empty_cache()
            
            return True
        else:
            print(f"  ✗ No GPU detected")
            print(f"    Training will be very slow on CPU")
            return False
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False

def check_directories():
    """Check project directory structure"""
    print("\nChecking directory structure...")
    
    base_dir = Path.cwd()
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/annotations',
        'outputs/visualizations',
        'outputs/predictions',
        'results/baseline',
        'results/multispectral',
        'results/comparisons',
        'models',
        'logs'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            all_exist = False
            # Create missing directory
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"    → Created {dir_path}")
    
    return all_exist

def test_model_loading():
    """Test loading a small transformer model"""
    print("\nTesting model loading...")
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Load a tiny model to test
        print("  Loading test model (distilbert-base-uncased)...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Quick inference test
        inputs = tokenizer("Testing model loading", return_tensors="pt")
        outputs = model(**inputs)
        
        print("  ✓ Model loading successful")
        print("  ✓ Inference test passed")
        
        del tokenizer, model
        return True
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False

def test_image_processing():
    """Test basic image processing"""
    print("\nTesting image processing...")
    try:
        import numpy as np
        from PIL import Image
        
        # Create a test image
        test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        
        # Test basic operations
        resized = test_image.resize((112, 112))
        array = np.array(resized)
        
        print("  ✓ PIL image operations")
        
        # Test with OpenCV
        import cv2
        cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        print("  ✓ OpenCV operations")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Image processing failed: {e}")
        return False

def generate_report(results):
    """Generate final report"""
    print("\n" + "="*60)
    print("SETUP VERIFICATION REPORT")
    print("="*60)
    
    all_passed = all(results.values())
    
    print("\nResults:")
    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {check}: {status}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to start!")
        print("="*60)
        print("\nNext steps:")
        print("1. Download Sentinel-2 imagery to data/raw/")
        print("2. Run baseline model testing")
        print("3. Begin Day 1 tasks")
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues above")
        print("="*60)
        print("\nCommon solutions:")
        print("- No GPU: Use Google Colab or cloud GPU")
        print("- Missing libraries: pip install -r requirements.txt")
        print("- Model loading fails: Check internet connection")
    print("")
    
    return all_passed

def main():
    """Run all verification checks"""
    print("="*60)
    print("ISRO MULTIMODAL PROJECT - SETUP VERIFICATION")
    print("="*60)
    print()
    
    results = {
        'Python Version': check_python_version(),
        'Required Libraries': check_libraries(),
        'GPU Availability': check_gpu(),
        'Directory Structure': check_directories(),
        'Model Loading': test_model_loading(),
        'Image Processing': test_image_processing(),
    }
    
    all_passed = generate_report(results)
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
