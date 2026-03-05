import torch
import sys

print("="*60)
print("SETUP VERIFICATION")
print("="*60)

print(f"\n✓ Python: {sys.version}")
print(f"✓ PyTorch: {torch.__version__}")

print("\n--- GPU CHECK ---")
if torch.cuda.is_available():
    print(f"✓ CUDA Available: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = gpu_props.total_memory / (1024**3)
    print(f"✓ GPU Memory: {gpu_memory_gb:.2f} GB")
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = x @ x.T
    print(f"✓ GPU computation test PASSED")
    
    del x, y
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("🎉 GPU SETUP SUCCESSFUL!")
    print("="*60)
else:
    print("❌ NO GPU DETECTED")
    print("\nPossible issues:")
    print("1. NVIDIA drivers not installed")
    print("2. CUDA toolkit not installed")
    print("3. PyTorch installed without CUDA support")
    print("\nRun 'nvidia-smi' to check GPU")