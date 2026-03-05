"""
Baseline Model Testing - BLIP (Base) on Satellite Imagery
Using smaller BLIP model - faster download, less memory

This is a good baseline for Day 1 testing
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import time

class BaselineEvaluator:
    """Evaluate baseline BLIP model on satellite imagery"""
    
    def __init__(self, device='cuda'):
        """
        Initialize BLIP base model (much smaller and faster to download)
        """
        print("="*60)
        print("INITIALIZING BASELINE MODEL (BLIP-BASE)")
        print("="*60)
        
        self.device = device
        
        print("\nModel: Salesforce/blip-image-captioning-base")
        print("Model size: ~1GB (much faster download!)")
        print("Mode: FP16 (Windows compatible)")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.2f} GB)")
        else:
            print("No GPU - using CPU (slow)")
            device = 'cpu'
        
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
        
        # Check memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"\nGPU Memory: {allocated:.2f} MB")
            torch.cuda.empty_cache()
    
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
        
        with torch.no_grad():
            out = self.vqa_model.generate(**inputs, max_new_tokens=30)
        
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return answer
    
    def evaluate_dataset(self, image_dir, questions, max_images=20):
        """Evaluate on dataset"""
        print("\n" + "="*60)
        print("BASELINE EVALUATION")
        print("="*60)
        
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_files:
            print(f"No images in {image_dir}")
            return {}
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\nProcessing {len(image_files)} images")
        print(f"Questions per image: {len(questions)}")
        
        results = {
            'model': 'BLIP-base',
            'num_images': len(image_files),
            'evaluations': []
        }
        
        start_time = time.time()
        
        for img_path in tqdm(image_files, desc="Processing"):
            
            img_result = {
                'image_name': img_path.name,
                'qa_pairs': []
            }
            
            # Caption
            try:
                img_result['caption'] = self.generate_caption(img_path)
            except Exception as e:
                print(f"\nError: {e}")
                img_result['caption'] = "Error"
            
            # Questions
            for q in questions:
                try:
                    answer = self.answer_question(img_path, q)
                    img_result['qa_pairs'].append({'question': q, 'answer': answer})
                except Exception as e:
                    print(f"\nError: {e}")
                    img_result['qa_pairs'].append({'question': q, 'answer': "Error"})
            
            results['evaluations'].append(img_result)
        
        results['total_time'] = time.time() - start_time
        
        print(f"\n✓ Done! Time: {results['total_time']:.1f}s")
        
        return results
    
    def save_results(self, results, path):
        """Save to JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved: {path}")


def main():
    print("="*60)
    print("BASELINE EVALUATION - BLIP BASE MODEL")
    print("="*60)
    
    questions = [
        "What type of land cover is visible?",
        "Describe the main features.",
        "Is there vegetation present?",
        "Are there water bodies visible?",
        "Is this urban or rural?",
    ]
    
    try:
        evaluator = BaselineEvaluator()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have internet connection")
        return
    
    # Find images
    for dir_path in ['data/raw/eurosat', 'data/raw/synthetic', 'data/raw']:
        if Path(dir_path).exists():
            images = list(Path(dir_path).glob("*.jpg"))
            if images:
                image_dir = dir_path
                break
    else:
        print("\n✗ No images found!")
        print("Run: python download_satellite_data.py")
        return
    
    print(f"\nUsing images from: {image_dir}")
    
    # Evaluate
    results = evaluator.evaluate_dataset(image_dir, questions, max_images=20)
    
    # Save
    output = 'results/baseline/baseline_results.json'
    evaluator.save_results(results, output)
    
    # Show samples
    print("\n" + "="*60)
    print("SAMPLE RESULTS")
    print("="*60)
    
    if results['evaluations']:
        sample = results['evaluations'][0]
        print(f"\nImage: {sample['image_name']}")
        print(f"Caption: {sample['caption']}")
        print("\nQ&A:")
        for qa in sample['qa_pairs'][:3]:
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}")
    
    print("\n" + "="*60)
    print("✓ DAY 1 COMPLETE!")
    print("="*60)
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()