"""
Evaluate Trained Multispectral VLM
Day 4 - Model Evaluation

Test the trained model's generation capabilities
"""

import torch
import json
import numpy as np
from pathlib import Path
from day4_multimodal_model import MultispectralVLM

def load_model(checkpoint_path='checkpoints/best_model.pt'):
    """Load trained model"""
    print("Loading trained model...")
    
    # Create model
    model = MultispectralVLM(use_lora=True, lora_rank=8)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Training loss: {checkpoint['loss']:.4f}")
    
    return model

def load_multispectral_image(bands_dict):
    """Load 13-band image from .npy files"""
    band_ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    
    bands = []
    for band_id in band_ids:
        if band_id in bands_dict:
            band_path = bands_dict[band_id]
            band_data = np.load(band_path)
            # Normalize to 0-1
            band_data = band_data.astype(np.float32) / 10000.0
            bands.append(band_data)
    
    # Stack to (13, H, W)
    image = np.stack(bands, axis=0)
    return torch.from_numpy(image).float()

def generate_caption(model, image, device='cuda', max_length=100):
    """
    Generate caption using beam search or greedy decoding
    """
    model.eval()
    
    with torch.no_grad():
        # Encode image
        visual_embeddings = model.encode_image(image.unsqueeze(0).to(device))
        
        # Create prompt
        prompt = "This satellite image shows"
        prompt_tokens = model.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(device)
        
        # Get text embeddings
        text_embeddings = model.language_model.get_input_embeddings()(
            prompt_tokens.input_ids
        )
        
        # Combine visual + text embeddings
        combined_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones(
            combined_embeddings.shape[:2],
            device=device,
            dtype=torch.long
        )
        
        # Generate tokens autoregressively
        generated_ids = prompt_tokens.input_ids.clone()
        
        for _ in range(max_length):
            # Get current embeddings
            current_text_emb = model.language_model.get_input_embeddings()(generated_ids)
            current_combined = torch.cat([visual_embeddings, current_text_emb], dim=1)
            
            # Create mask
            current_mask = torch.ones(
                current_combined.shape[:2],
                device=device,
                dtype=torch.long
            )
            
            # Forward pass
            outputs = model.language_model(
                inputs_embeds=current_combined,
                attention_mask=current_mask
            )
            
            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == model.tokenizer.eos_token_id:
                break
        
        # Decode
        generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text

def evaluate_samples(model, data_path='data/training/training_data.json', num_samples=10):
    """Evaluate model on samples"""
    
    print("="*60)
    print("EVALUATING TRAINED MODEL")
    print("="*60)
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Test on first num_samples
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}/{num_samples}")
        print(f"{'='*60}")
        
        # Load image
        image = load_multispectral_image(sample['bands'])
        
        # Get ground truth
        land_cover = sample['land_cover']
        ndvi = sample['spectral_indices']['NDVI']
        ground_truth = sample['captions'][0]
        
        print(f"\nLand Cover: {land_cover}")
        print(f"NDVI: {ndvi:.3f}")
        print(f"\nGround Truth:")
        print(f"  {ground_truth}")
        
        # Generate caption
        print(f"\nGenerating caption...")
        generated = generate_caption(model, image, device, max_length=50)
        
        print(f"\nGenerated:")
        print(f"  {generated}")
        
        # Check for spectral keywords
        keywords = ['NDVI', 'NIR', 'SWIR', 'infrared', 'vegetation', 'reflectance']
        found_keywords = [kw for kw in keywords if kw.lower() in generated.lower()]
        
        if found_keywords:
            print(f"\n✓ Spectral keywords found: {', '.join(found_keywords)}")
        else:
            print(f"\n⚠ No spectral keywords in generation")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")

def main():
    # Load trained model
    model = load_model()
    
    # Evaluate on samples
    evaluate_samples(model, num_samples=5)

if __name__ == "__main__":
    main()