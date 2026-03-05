"""
Training Script for Multispectral VLM
Day 4 - Model Training

Train the SpectralViT + GPT-2 model on spectral-aware captions
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

from day4_multimodal_model import MultispectralVLM

class MultispectralDataset(Dataset):
    """
    Dataset for multispectral image-caption pairs
    """
    
    def __init__(self, data_path='data/training/training_data.json',
                 multispectral_dir='data/raw/sentinel2_multispectral'):
        """
        Args:
            data_path: Path to training_data.json
            multispectral_dir: Directory with .npy band files
        """
        # Load training data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.multispectral_dir = Path(multispectral_dir)
        
        # Flatten captions (one sample per caption)
        self.samples = []
        for item in self.data:
            for caption in item['captions']:
                self.samples.append({
                    'sample_id': item['sample_id'],
                    'caption': caption,
                    'bands': item['bands']
                })
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def load_multispectral_image(self, bands_dict):
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
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load multispectral image
        image = self.load_multispectral_image(sample['bands'])
        
        # Get caption
        caption = sample['caption']
        
        return {
            'image': image,
            'caption': caption
        }


def collate_fn(batch, tokenizer, max_length=128):
    """Collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    # Tokenize captions
    encodings = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        'images': images,
        'input_ids': encodings.input_ids,
        'attention_mask': encodings.attention_mask,
        'labels': encodings.input_ids.clone()  # For language modeling loss
    }


def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return avg_loss


def main():
    """Main training function"""
    
    print("="*60)
    print("MULTISPECTRAL VLM TRAINING")
    print("="*60)
    
    # Configuration
    batch_size = 1  # Small batch for 4GB GPU
    num_epochs = 10
    learning_rate = 5e-5
    gradient_accumulation_steps = 4  # Effective batch size = 4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = MultispectralVLM(use_lora=True, lora_rank=8)
    model = model.to(device)
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = MultispectralDataset()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer),
        num_workers=0  # 0 for Windows compatibility
    )
    
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, scaler, device, epoch)
        
        print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            checkpoint_dir = Path('checkpoints')
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            print(f"  ✓ Saved best model (loss: {avg_loss:.4f})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: checkpoints/best_model.pt")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    
    # Get first sample
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)
    
    print(f"Ground truth: {sample['caption']}")
    
    with torch.no_grad():
        generated = model.generate(
            image,
            prompt_text="Describe this satellite image:",
            max_new_tokens=30
        )
        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()