"""
Empty Folder Checker
Checks all project folders and reports which are empty or missing
"""

from pathlib import Path
import json

def check_folder_health():
    """Check all important folders in the project"""
    
    print("="*70)
    print(" "*20 + "PROJECT FOLDER HEALTH CHECK")
    print("="*70)
    
    # Define all important folders and what they should contain
    folders_to_check = {
        "Critical Files": [
            ("checkpoints/", "*.pt", "Trained model checkpoint"),
            ("data/training/", "*.json", "Training dataset"),
        ],
        "Multispectral Data": [
            ("data/raw/sentinel2_multispectral/", "*.npy", "13-band .npy files"),
            ("data/raw/sentinel2_multispectral/", "metadata.json", "Metadata JSON"),
        ],
        "Composites": [
            ("data/processed/composites/rgb/", "*.png", "RGB composites"),
            ("data/processed/composites/false_color_nir/", "*.png", "NIR false-color"),
            ("data/processed/composites/false_color_swir/", "*.png", "SWIR false-color"),
            ("data/processed/composites/ndvi/", "*.png", "NDVI visualizations"),
            ("data/processed/composites/ndwi/", "*.png", "NDWI visualizations"),
            ("data/processed/composites/ndbi/", "*.png", "NDBI visualizations"),
        ],
        "Results": [
            ("results/", "*.json", "Evaluation results"),
            ("results/final_presentation/", "*.png", "Visualization charts"),
        ],
        "Scripts": [
            ("./", "day*.py", "Python scripts"),
            ("./", "backend_graphql.py", "GraphQL backend"),
        ],
    }
    
    total_issues = 0
    total_checks = 0
    
    for category, checks in folders_to_check.items():
        print(f"\n📁 {category}")
        print("-" * 70)
        
        for folder_path, pattern, description in checks:
            total_checks += 1
            folder = Path(folder_path)
            
            # Check if folder exists
            if not folder.exists():
                print(f"  ✗ {folder_path:<45} FOLDER MISSING")
                total_issues += 1
                continue
            
            # Check if folder has files matching pattern
            if pattern.startswith("*."):
                # Wildcard pattern
                files = list(folder.glob(pattern))
            else:
                # Specific file
                files = [folder / pattern] if (folder / pattern).exists() else []
            
            if not files:
                print(f"  ✗ {folder_path:<45} EMPTY (no {pattern})")
                total_issues += 1
            else:
                # Calculate total size
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                print(f"  ✓ {folder_path:<45} {len(files):>3} files ({size_mb:>6.1f} MB)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total checks: {total_checks}")
    print(f"Issues found: {total_issues}")
    
    if total_issues == 0:
        print("\n✅ ALL FOLDERS HEALTHY - Ready to proceed!")
    else:
        print(f"\n⚠️  {total_issues} ISSUES FOUND - See details above")
        print("\nRECOMMENDED FIXES:")
        
        # Specific recommendations
        if not Path("checkpoints/best_model.pt").exists():
            print("  🔴 CRITICAL: Trained model missing!")
            print("     → Run: python day4_train.py")
        
        if not Path("data/raw/sentinel2_multispectral").exists() or \
           not list(Path("data/raw/sentinel2_multispectral").glob("*.npy")):
            print("  ⚠️  Multispectral data missing")
            print("     → Run: python day2_download_multispectral.py")
        
        if not Path("data/processed/composites/rgb").exists() or \
           not list(Path("data/processed/composites/rgb").glob("*.png")):
            print("  ⚠️  Composites missing")
            print("     → Run: python day2_create_composites.py")
        
        if not Path("data/training/training_data.json").exists():
            print("  ⚠️  Training data missing")
            print("     → Run: python day3_create_dataset.py")
    
    print("\n" + "="*70)
    
    # Create detailed report
    report = {
        "total_checks": total_checks,
        "issues_found": total_issues,
        "status": "HEALTHY" if total_issues == 0 else "ISSUES_FOUND"
    }
    
    with open("folder_health_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("📄 Detailed report saved to: folder_health_report.json")
    print("="*70)
    
    return total_issues == 0


if __name__ == "__main__":
    check_folder_health()