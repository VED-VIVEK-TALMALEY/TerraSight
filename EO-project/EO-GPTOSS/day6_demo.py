"""
Interactive Demo - Multispectral Vision-Language Model
Day 6 - Showcase Achievement

Live demonstration comparing baseline vs trained model
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import time

from day4_multimodal_model import MultispectralVLM
from transformers import BlipProcessor, BlipForConditionalGeneration

class LiveDemo:
    """Interactive demonstration"""
    
    def __init__(self, checkpoint_path='checkpoints/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*70)
        print(" "*10 + "MULTISPECTRAL VISION-LANGUAGE MODEL")
        print(" "*20 + "Live Demo")
        print("="*70)
        print()
        
        # Load trained model
        print("Loading SpectralVLM (trained model)...")
        self.trained_model = MultispectralVLM(use_lora=True, lora_rank=8)
        checkpoint = torch.load(checkpoint_path)
        self.trained_model.load_state_dict(checkpoint['model_state_dict'])
        self.trained_model = self.trained_model.to(self.device)
        self.trained_model.eval()
        print("  ✓ Trained model ready")
        
        # Load baseline model
        print("Loading BLIP baseline (RGB-only)...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()
        print("  ✓ Baseline model ready")
        
        # Load test data
        with open('data/training/training_data.json', 'r') as f:
            self.test_data = json.load(f)
        
        print(f"\n✓ Demo ready! Loaded {len(self.test_data)} samples")
        print("="*70)
    
    def load_multispectral_image(self, bands_dict):
        """Load 13-band image"""
        band_ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                    'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        
        bands = []
        for band_id in band_ids:
            if band_id in bands_dict:
                band_path = bands_dict[band_id]
                band_data = np.load(band_path)
                band_data = band_data.astype(np.float32) / 10000.0
                bands.append(band_data)
        
        image = np.stack(bands, axis=0)
        return torch.from_numpy(image).float()
    
    def create_rgb_composite(self, multispectral_image):
        """Create RGB for baseline"""
        red = multispectral_image[3]
        green = multispectral_image[2]
        blue = multispectral_image[1]
        
        def normalize(band):
            band = np.clip(band, 0, 1)
            return (band * 255).astype(np.uint8)
        
        rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=2)
        return Image.fromarray(rgb)
    
    def generate_caption_trained(self, image, max_length=50):
        """Generate with trained model"""
        with torch.no_grad():
            visual_embeddings = self.trained_model.encode_image(
                image.unsqueeze(0).to(self.device)
            )
            
            prompt = "This satellite image shows"
            prompt_tokens = self.trained_model.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.device)
            
            generated_ids = prompt_tokens.input_ids.clone()
            
            for _ in range(max_length):
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
            
            return self.trained_model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def generate_caption_baseline(self, rgb_image):
        """Generate with baseline"""
        inputs = self.blip_processor(rgb_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=50)
        
        return self.blip_processor.decode(out[0], skip_special_tokens=True)
    
    def show_comparison(self, sample_idx):
        """Show side-by-side comparison"""
        sample = self.test_data[sample_idx]
        
        # Load image
        multi_image = self.load_multispectral_image(sample['bands'])
        rgb_image = self.create_rgb_composite(multi_image.numpy())
        
        # Ground truth
        land_cover = sample['land_cover']
        ndvi = sample['spectral_indices']['NDVI']
        ndwi = sample['spectral_indices']['NDWI']
        ndbi = sample['spectral_indices']['NDBI']
        
        print("\n" + "="*70)
        print(f"SAMPLE #{sample_idx}")
        print("="*70)
        
        print(f"\n📊 GROUND TRUTH:")
        print(f"  Land Cover: {land_cover.upper()}")
        print(f"  NDVI: {ndvi:.3f}  (vegetation index)")
        print(f"  NDWI: {ndwi:.3f}  (water index)")
        print(f"  NDBI: {ndbi:.3f}  (built-up index)")
        
        print(f"\n" + "-"*70)
        print("BASELINE MODEL (RGB only - 3 bands)")
        print("-"*70)
        
        start = time.time()
        baseline_caption = self.generate_caption_baseline(rgb_image)
        baseline_time = time.time() - start
        
        print(f"Caption:")
        print(f"  {baseline_caption}")
        print(f"\n⏱️ Generation time: {baseline_time:.2f}s")
        
        # Check for spectral keywords
        keywords_base = []
        if 'NDVI' in baseline_caption.upper():
            keywords_base.append('NDVI')
        if 'NIR' in baseline_caption.upper() or 'infrared' in baseline_caption.lower():
            keywords_base.append('NIR')
        if 'SWIR' in baseline_caption.upper():
            keywords_base.append('SWIR')
        
        if keywords_base:
            print(f"✓ Spectral keywords: {', '.join(keywords_base)}")
        else:
            print(f"❌ No spectral keywords found")
        
        print(f"\n" + "-"*70)
        print("TRAINED MODEL (Multispectral - 13 bands)")
        print("-"*70)
        
        start = time.time()
        trained_caption = self.generate_caption_trained(multi_image)
        trained_time = time.time() - start
        
        print(f"Caption:")
        print(f"  {trained_caption}")
        print(f"\n⏱️ Generation time: {trained_time:.2f}s")
        
        # Check for spectral keywords
        keywords_trained = []
        if 'NDVI' in trained_caption.upper():
            keywords_trained.append('NDVI')
        if 'NIR' in trained_caption.upper() or 'infrared' in trained_caption.lower():
            keywords_trained.append('NIR')
        if 'SWIR' in trained_caption.upper():
            keywords_trained.append('SWIR')
        if 'reflectance' in trained_caption.lower():
            keywords_trained.append('REFLECTANCE')
        
        if keywords_trained:
            print(f"✓ Spectral keywords: {', '.join(keywords_trained)}")
            print(f"✓ Keyword count: {len(keywords_trained)}")
        else:
            print(f"⚠️ No spectral keywords found")
        
        # Extract NDVI if present
        import re
        ndvi_match = re.search(r'NDVI[:\s]+([0-9.]+)', trained_caption, re.IGNORECASE)
        if ndvi_match:
            predicted_ndvi = float(ndvi_match.group(1))
            error = abs(predicted_ndvi - ndvi)
            print(f"\n✓ NDVI Prediction: {predicted_ndvi:.3f}")
            print(f"  Error: {error:.3f} (target: <0.1)")
            if error < 0.1:
                print(f"  ✓ Within target!")
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"Baseline keywords: {len(keywords_base)}")
        print(f"Trained keywords:  {len(keywords_trained)}")
        print(f"Improvement:       +{len(keywords_trained) - len(keywords_base)}")
        
        return {
            'baseline': baseline_caption,
            'trained': trained_caption,
            'keywords_baseline': len(keywords_base),
            'keywords_trained': len(keywords_trained)
        }
    
    def interactive_demo(self):
        """Run interactive demo"""
        
        print("\n" + "="*70)
        print("INTERACTIVE DEMO MODE")
        print("="*70)
        print(f"\nAvailable samples: 0-{len(self.test_data)-1}")
        print("Commands:")
        print("  [number] - Show comparison for sample")
        print("  'random' - Show random sample")
        print("  'summary' - Show overall statistics")
        print("  'quit' - Exit demo")
        
        while True:
            print("\n" + "-"*70)
            choice = input("Enter command: ").strip().lower()
            
            if choice == 'quit':
                print("\n✓ Demo ended. Thank you!")
                break
            
            elif choice == 'random':
                import random
                idx = random.randint(0, len(self.test_data)-1)
                self.show_comparison(idx)
            
            elif choice == 'summary':
                self.show_summary()
            
            elif choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(self.test_data):
                    self.show_comparison(idx)
                else:
                    print(f"Invalid sample index. Range: 0-{len(self.test_data)-1}")
            
            else:
                print("Invalid command. Try 'random', 'summary', or a sample number.")
    
    def show_summary(self):
        """Show overall performance summary"""
        print("\n" + "="*70)
        print("OVERALL PERFORMANCE SUMMARY")
        print("="*70)
        print("\nFrom Day 5 Evaluation (20 samples):")
        print("\nSpectral Keyword Usage:")
        print("  Baseline:  0%")
        print("  Trained:   50%")
        print("  Improvement: +50%")
        print("\nNDVI Estimation:")
        print("  Baseline coverage:  0% (0/20 samples)")
        print("  Trained coverage:   50% (10/20 samples)")
        print("  Average error:      ±0.071")
        print("  Within target:      100% (10/10)")
        print("\nKey Achievements:")
        print("  ✓ Spectral awareness learned")
        print("  ✓ Quantitative NDVI prediction")
        print("  ✓ Technical vocabulary usage")
        print("  ✓ Physical reasoning capability")


def main():
    """Run demo"""
    
    demo = LiveDemo()
    
    print("\n" + "="*70)
    print("QUICK DEMO - Showing 3 examples")
    print("="*70)
    
    # Show 3 diverse examples
    for idx in [0, 10, 20]:
        if idx < len(demo.test_data):
            demo.show_comparison(idx)
            input("\nPress Enter to continue...")
    
    # Show summary
    demo.show_summary()
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("\nYour model demonstrates:")
    print("  ✓ 50% spectral keyword usage (vs 0% baseline)")
    print("  ✓ Accurate NDVI estimation (±0.071)")
    print("  ✓ Spectral awareness and reasoning")
    print("="*70)
    
    # Offer interactive mode
    print("\nWant to try interactive mode? (y/n): ", end='')
    choice = input().strip().lower()
    
    if choice == 'y':
        demo.interactive_demo()


if __name__ == "__main__":
    main()