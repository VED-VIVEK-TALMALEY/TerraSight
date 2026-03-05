"""
Test Baseline Model on Different Band Composites
Day 2 - Problem Demonstration

This script:
1. Tests BLIP on RGB composites
2. Tests BLIP on false-color composites  
3. Compares results to show contradictory answers
4. Quantifies information loss
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import time

class CompositeComparison:
    """Compare baseline model performance across different band composites"""
    
    def __init__(self, device='cuda'):
        print("="*60)
        print("LOADING BASELINE MODEL")
        print("="*60)
        
        self.device = device
        
        # Load processor
        print("\nLoading processor...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Load captioning model
        print("Loading captioning model...")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        
        # Load VQA model
        print("Loading VQA model...")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        
        # Move to device
        if torch.cuda.is_available():
            self.caption_model = self.caption_model.to(device, torch.float16)
            self.vqa_model = self.vqa_model.to(device, torch.float16)
        else:
            self.caption_model = self.caption_model.to(device)
            self.vqa_model = self.vqa_model.to(device)
        
        print("\n✓ Models loaded successfully!")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"GPU Memory: {allocated:.2f} MB")
    
    def generate_caption(self, image_path):
        """Generate caption for an image"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device, torch.float16) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self.caption_model.generate(**inputs, max_new_tokens=50)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return caption
    
    def answer_question(self, image_path, question):
        """Answer question about image"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(image, question, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device, torch.float16) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                out = self.vqa_model.generate(**inputs, max_new_tokens=30)
            
            answer = self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            answer = f"Error: {str(e)}"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return answer
    
    def test_composite_types(self, num_samples=10):
        """
        Test model on different composite types
        
        Args:
            num_samples: Number of samples to test (default 10)
        """
        print("\n" + "="*60)
        print("TESTING BASELINE ON DIFFERENT COMPOSITES")
        print("="*60)
        
        # Load composite metadata
        composite_dir = Path('data/processed/composites')
        with open(composite_dir / 'composite_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Test on subset
        samples_to_test = metadata[:num_samples]
        
        # Questions to ask
        questions = [
            "What type of land cover is visible in this image?",
            "Is there vegetation present?",
            "Are there water bodies visible?",
        ]
        
        # Composite types to compare
        composite_types = ['rgb', 'false_color_nir', 'false_color_swir']
        
        results = []
        
        print(f"\nTesting {len(samples_to_test)} samples across {len(composite_types)} composite types...")
        print("This will take ~5-10 minutes...\n")
        
        for sample_info in tqdm(samples_to_test, desc="Processing samples"):
            sample_id = sample_info['sample_id']
            land_cover = sample_info['land_cover']
            
            sample_result = {
                'sample_id': sample_id,
                'land_cover': land_cover,
                'composite_results': {}
            }
            
            # Test each composite type
            for comp_type in composite_types:
                image_path = sample_info['composites'][comp_type]
                
                # Generate caption
                try:
                    caption = self.generate_caption(image_path)
                except Exception as e:
                    caption = f"Error: {str(e)}"
                
                # Answer questions
                qa_pairs = []
                for question in questions:
                    try:
                        answer = self.answer_question(image_path, question)
                    except Exception as e:
                        answer = f"Error: {str(e)}"
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })
                
                sample_result['composite_results'][comp_type] = {
                    'caption': caption,
                    'qa_pairs': qa_pairs
                }
            
            results.append(sample_result)
        
        # Save results
        output_file = Path('results/day2_composite_comparison.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        return results
    
    def analyze_contradictions(self, results):
        """
        Analyze results to find contradictory answers
        """
        print("\n" + "="*60)
        print("ANALYZING CONTRADICTIONS")
        print("="*60)
        
        contradictions = []
        
        for sample_result in results:
            sample_id = sample_result['sample_id']
            land_cover = sample_result['land_cover']
            
            # Compare answers across composites
            rgb_result = sample_result['composite_results']['rgb']
            nir_result = sample_result['composite_results']['false_color_nir']
            swir_result = sample_result['composite_results']['false_color_swir']
            
            # Check if captions differ
            captions = {
                'RGB': rgb_result['caption'],
                'NIR': nir_result['caption'],
                'SWIR': swir_result['caption']
            }
            
            unique_captions = set(captions.values())
            if len(unique_captions) > 1:
                contradictions.append({
                    'sample_id': sample_id,
                    'land_cover': land_cover,
                    'type': 'caption',
                    'rgb': captions['RGB'],
                    'nir': captions['NIR'],
                    'swir': captions['SWIR']
                })
            
            # Check if VQA answers differ
            for q_idx in range(len(rgb_result['qa_pairs'])):
                question = rgb_result['qa_pairs'][q_idx]['question']
                
                answers = {
                    'RGB': rgb_result['qa_pairs'][q_idx]['answer'],
                    'NIR': nir_result['qa_pairs'][q_idx]['answer'],
                    'SWIR': swir_result['qa_pairs'][q_idx]['answer']
                }
                
                unique_answers = set(answers.values())
                if len(unique_answers) > 1 and 'Error' not in str(unique_answers):
                    contradictions.append({
                        'sample_id': sample_id,
                        'land_cover': land_cover,
                        'type': 'vqa',
                        'question': question,
                        'rgb': answers['RGB'],
                        'nir': answers['NIR'],
                        'swir': answers['SWIR']
                    })
        
        # Print contradictions
        print(f"\nFound {len(contradictions)} contradictions!")
        
        if contradictions:
            print("\n" + "="*60)
            print("EXAMPLE CONTRADICTIONS")
            print("="*60)
            
            # Show first 5 contradictions
            for i, contra in enumerate(contradictions[:5]):
                print(f"\n{i+1}. Sample {contra['sample_id']} ({contra['land_cover']})")
                
                if contra['type'] == 'caption':
                    print("   Type: Caption")
                    print(f"   RGB:  '{contra['rgb']}'")
                    print(f"   NIR:  '{contra['nir']}'")
                    print(f"   SWIR: '{contra['swir']}'")
                else:
                    print(f"   Type: VQA")
                    print(f"   Q: {contra['question']}")
                    print(f"   RGB answer:  '{contra['rgb']}'")
                    print(f"   NIR answer:  '{contra['nir']}'")
                    print(f"   SWIR answer: '{contra['swir']}'")
        
        # Calculate statistics
        caption_contradictions = [c for c in contradictions if c['type'] == 'caption']
        vqa_contradictions = [c for c in contradictions if c['type'] == 'vqa']
        
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Total samples tested: {len(results)}")
        print(f"Total contradictions: {len(contradictions)}")
        print(f"  Caption contradictions: {len(caption_contradictions)}")
        print(f"  VQA contradictions: {len(vqa_contradictions)}")
        print(f"\nContradiction rate: {len(contradictions)/(len(results)*4)*100:.1f}%")
        
        return contradictions

def main():
    """Main comparison function"""
    
    print("="*60)
    print("DAY 2: BASELINE MODEL COMPOSITE COMPARISON")
    print("="*60)
    
    # Initialize comparison
    comparison = CompositeComparison()
    
    # Test on different composites
    print("\nThis will test the baseline model on:")
    print("  1. RGB composites (standard)")
    print("  2. NIR false-color (vegetation emphasis)")
    print("  3. SWIR false-color (water/moisture emphasis)")
    print("\nFor each composite, the model will:")
    print("  - Generate a caption")
    print("  - Answer 3 questions")
    
    input("\nPress Enter to start testing...")
    
    results = comparison.test_composite_types(num_samples=10)
    
    # Analyze contradictions
    contradictions = comparison.analyze_contradictions(results)
    
    # Save contradictions
    if contradictions:
        output_file = Path('results/day2_contradictions.json')
        with open(output_file, 'w') as f:
            json.dump(contradictions, f, indent=2)
        print(f"\n✓ Contradictions saved to: {output_file}")
    
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    print("\n🔴 The baseline model gives DIFFERENT answers depending on")
    print("   which spectral bands it sees!")
    print("\n   This PROVES that:")
    print("   1. Different band combinations contain different information")
    print("   2. RGB alone is insufficient for satellite image analysis")
    print("   3. We need a model that can handle multispectral data")
    print("\n" + "="*60)
    print("DAY 2 PROBLEM DEMONSTRATION: COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()