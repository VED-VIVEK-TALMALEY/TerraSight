"""
Comprehensive Model Evaluation
Day 5 - Compare Trained Model vs Baseline

Quantitatively measure improvements in:
1. Spectral keyword usage
2. NDVI estimation accuracy
3. Consistency across composites
4. Land cover classification
"""

import torch
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from day4_multimodal_model import MultispectralVLM
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

class ComprehensiveEvaluator:
    """Evaluate trained model vs baseline"""
    
    def __init__(self, 
                 trained_model_path='checkpoints/best_model.pt',
                 test_data_path='data/training/training_data.json'):
        
        self.test_data_path = test_data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        print("Loading trained SpectralVLM...")
        self.trained_model = MultispectralVLM(use_lora=True, lora_rank=8)
        checkpoint = torch.load(trained_model_path)
        self.trained_model.load_state_dict(checkpoint['model_state_dict'])
        self.trained_model = self.trained_model.to(self.device)
        self.trained_model.eval()
        
        # Load baseline model (BLIP)
        print("Loading baseline BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()
        
        # Load test data
        with open(test_data_path, 'r') as f:
            self.test_data = json.load(f)
    
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
        """Create RGB composite from multispectral for baseline"""
        # Extract RGB bands (B04=Red, B03=Green, B02=Blue)
        red = multispectral_image[3]    # B04
        green = multispectral_image[2]  # B03
        blue = multispectral_image[1]   # B02
        
        # Normalize to 0-255
        def normalize(band):
            band = np.clip(band, 0, 1)
            return (band * 255).astype(np.uint8)
        
        rgb = np.stack([
            normalize(red),
            normalize(green),
            normalize(blue)
        ], axis=2)
        
        from PIL import Image
        return Image.fromarray(rgb)
    
    def generate_caption_trained(self, image, max_length=50):
        """Generate caption with trained model"""
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
        """Generate caption with baseline BLIP"""
        inputs = self.blip_processor(rgb_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=50)
        
        return self.blip_processor.decode(out[0], skip_special_tokens=True)
    
    def extract_spectral_keywords(self, text):
        """Extract spectral-related keywords"""
        keywords = {
            'ndvi': 'NDVI' in text.upper() or 'vegetation index' in text.lower(),
            'ndwi': 'NDWI' in text.upper() or 'water index' in text.lower(),
            'ndbi': 'NDBI' in text.upper() or 'built-up index' in text.lower() or 'built up' in text.lower(),
            'nir': 'NIR' in text.upper() or 'near-infrared' in text.lower() or 'near infrared' in text.lower(),
            'swir': 'SWIR' in text.upper() or 'short-wave infrared' in text.lower() or 'shortwave' in text.lower(),
            'red_edge': 'red edge' in text.lower() or 'red-edge' in text.lower(),
            'reflectance': 'reflectance' in text.lower() or 'reflection' in text.lower(),
            'absorption': 'absorption' in text.lower() or 'absorb' in text.lower(),
        }
        return keywords
    
    def extract_ndvi_value(self, text):
        """Extract NDVI value from text"""
        import re
        # Look for patterns like "NDVI: 0.76" or "NDVI of 0.76"
        patterns = [
            r'NDVI[:\s]+([0-9.-]+)',
            r'vegetation index[:\s]+([0-9.-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None
    
    def evaluate_all_samples(self, num_samples=20):
        """Comprehensive evaluation"""
        
        print("="*60)
        print("COMPREHENSIVE EVALUATION")
        print("="*60)
        
        results = {
            'trained': [],
            'baseline': [],
            'comparisons': []
        }
        
        samples = self.test_data[:num_samples]
        
        for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
            # Load image
            multi_image = self.load_multispectral_image(sample['bands'])
            rgb_image = self.create_rgb_composite(multi_image.numpy())
            
            # Ground truth
            land_cover = sample['land_cover']
            true_ndvi = sample['spectral_indices']['NDVI']
            true_caption = sample['captions'][0]
            
            # Generate captions
            trained_caption = self.generate_caption_trained(multi_image)
            baseline_caption = self.generate_caption_baseline(rgb_image)
            
            # Extract keywords
            trained_keywords = self.extract_spectral_keywords(trained_caption)
            baseline_keywords = self.extract_spectral_keywords(baseline_caption)
            
            # Extract NDVI
            trained_ndvi = self.extract_ndvi_value(trained_caption)
            baseline_ndvi = self.extract_ndvi_value(baseline_caption)
            
            # Store results
            results['trained'].append({
                'sample_id': idx,
                'land_cover': land_cover,
                'caption': trained_caption,
                'keywords': trained_keywords,
                'ndvi_predicted': trained_ndvi,
                'ndvi_true': true_ndvi
            })
            
            results['baseline'].append({
                'sample_id': idx,
                'land_cover': land_cover,
                'caption': baseline_caption,
                'keywords': baseline_keywords,
                'ndvi_predicted': baseline_ndvi,
                'ndvi_true': true_ndvi
            })
            
            # Comparison
            keyword_improvement = sum(trained_keywords.values()) - sum(baseline_keywords.values())
            
            ndvi_error_trained = abs(trained_ndvi - true_ndvi) if trained_ndvi else float('inf')
            ndvi_error_baseline = abs(baseline_ndvi - true_ndvi) if baseline_ndvi else float('inf')
            
            results['comparisons'].append({
                'sample_id': idx,
                'keyword_improvement': keyword_improvement,
                'ndvi_error_trained': ndvi_error_trained,
                'ndvi_error_baseline': ndvi_error_baseline
            })
        
        return results
    
    def compute_metrics(self, results):
        """Compute aggregate metrics"""
        
        print("\n" + "="*60)
        print("AGGREGATE METRICS")
        print("="*60)
        
        metrics = {}
        
        # Spectral keyword usage
        trained_keyword_counts = defaultdict(int)
        baseline_keyword_counts = defaultdict(int)
        
        for res in results['trained']:
            for key, found in res['keywords'].items():
                if found:
                    trained_keyword_counts[key] += 1
        
        for res in results['baseline']:
            for key, found in res['keywords'].items():
                if found:
                    baseline_keyword_counts[key] += 1
        
        total_samples = len(results['trained'])
        
        print("\n1. SPECTRAL KEYWORD USAGE (%)")
        print("-" * 60)
        print(f"{'Keyword':<20} {'Baseline':<15} {'Trained':<15} {'Improvement'}")
        print("-" * 60)
        
        for key in ['ndvi', 'nir', 'swir', 'reflectance']:
            baseline_pct = 100 * baseline_keyword_counts[key] / total_samples
            trained_pct = 100 * trained_keyword_counts[key] / total_samples
            improvement = trained_pct - baseline_pct
            
            print(f"{key.upper():<20} {baseline_pct:>6.1f}%        {trained_pct:>6.1f}%        {improvement:>+6.1f}%")
            
            metrics[f'{key}_usage_baseline'] = baseline_pct
            metrics[f'{key}_usage_trained'] = trained_pct
        
        # NDVI estimation accuracy
        trained_ndvi_errors = [r['ndvi_error_trained'] for r in results['comparisons'] 
                              if r['ndvi_error_trained'] != float('inf')]
        baseline_ndvi_errors = [r['ndvi_error_baseline'] for r in results['comparisons']
                               if r['ndvi_error_baseline'] != float('inf')]
        
        print("\n2. NDVI ESTIMATION")
        print("-" * 60)
        
        if trained_ndvi_errors:
            avg_error_trained = np.mean(trained_ndvi_errors)
            print(f"Trained model:")
            print(f"  Samples with NDVI: {len(trained_ndvi_errors)}/{total_samples}")
            print(f"  Average error: {avg_error_trained:.3f}")
            metrics['ndvi_avg_error_trained'] = avg_error_trained
            metrics['ndvi_coverage_trained'] = 100 * len(trained_ndvi_errors) / total_samples
        else:
            print(f"Trained model: No NDVI values generated")
            metrics['ndvi_coverage_trained'] = 0
        
        if baseline_ndvi_errors:
            avg_error_baseline = np.mean(baseline_ndvi_errors)
            print(f"Baseline model:")
            print(f"  Samples with NDVI: {len(baseline_ndvi_errors)}/{total_samples}")
            print(f"  Average error: {avg_error_baseline:.3f}")
            metrics['ndvi_avg_error_baseline'] = avg_error_baseline
            metrics['ndvi_coverage_baseline'] = 100 * len(baseline_ndvi_errors) / total_samples
        else:
            print(f"Baseline model: No NDVI values generated")
            metrics['ndvi_coverage_baseline'] = 0
        
        # Overall improvements
        print("\n3. OVERALL IMPROVEMENTS")
        print("-" * 60)
        
        avg_keyword_improvement = np.mean([r['keyword_improvement'] for r in results['comparisons']])
        print(f"Average keyword improvement per sample: {avg_keyword_improvement:+.2f}")
        
        return metrics
    
    def print_example_comparisons(self, results, num_examples=3):
        """Print example comparisons"""
        
        print("\n" + "="*60)
        print("EXAMPLE COMPARISONS")
        print("="*60)
        
        for i in range(min(num_examples, len(results['trained']))):
            trained = results['trained'][i]
            baseline = results['baseline'][i]
            
            print(f"\nSAMPLE {i+1}: {trained['land_cover'].upper()}")
            print(f"True NDVI: {trained['ndvi_true']:.3f}")
            print("-" * 60)
            
            print(f"\nBaseline (RGB only):")
            print(f"  Caption: {baseline['caption']}")
            keywords_base = [k.upper() for k, v in baseline['keywords'].items() if v]
            print(f"  Keywords: {', '.join(keywords_base) if keywords_base else 'None'}")
            
            print(f"\nTrained (Multispectral):")
            print(f"  Caption: {trained['caption']}")
            keywords_trained = [k.upper() for k, v in trained['keywords'].items() if v]
            print(f"  Keywords: {', '.join(keywords_trained)}")
            
            if trained['ndvi_predicted']:
                error = abs(trained['ndvi_predicted'] - trained['ndvi_true'])
                print(f"  NDVI predicted: {trained['ndvi_predicted']:.3f} (error: {error:.3f})")


def main():
    """Run comprehensive evaluation"""
    
    evaluator = ComprehensiveEvaluator()
    
    # Evaluate all samples
    results = evaluator.evaluate_all_samples(num_samples=20)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(results)
    
    # Show examples
    evaluator.print_example_comparisons(results, num_examples=3)
    
    # Save results
    output_file = Path('results/day5_evaluation.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'metrics': metrics
        }, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*60)
    print("DAY 5 EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()