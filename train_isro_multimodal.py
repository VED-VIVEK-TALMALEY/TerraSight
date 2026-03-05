# train_isro_multimodal.py
"""
COMPLETE TRAINING SCRIPT FOR ISRO MULTIMODAL GPT
Implements two-stage training: Pretraining + Instruction Tuning
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
from datetime import datetime

# Import our custom modules (from previous sections)
# from architecture import ISROMultimodalGPT
# from isro_data_loader import create_isro_dataloader

class Config:
    """Training configuration"""
    
    # Model
    VISION_MODEL = "openai/clip-vit-large-patch14"
    LANGUAGE_MODEL = "meta-llama/Llama-2-7b-hf"
    
    # Data
    TRAIN_DATA_DIR = "data/isro_images/train"
    TRAIN_ANNOTATIONS = "data/annotations/train.json"
    VAL_DATA_DIR = "data/isro_images/val"
    VAL_ANNOTATIONS = "data/annotations/val.json"
    
    # Training
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    NUM_EPOCHS_STAGE1 = 1  # Pretraining
    NUM_EPOCHS_STAGE2 = 3  # Instruction tuning
    NUM_EPOCHS_STAGE3 = 5  # ISRO-specific
    
    LEARNING_RATE_STAGE1 = 1e-3  # Higher for projection layer only
    LEARNING_RATE_STAGE2 = 2e-5  # Lower for full model
    LEARNING_RATE_STAGE3 = 1e-5  # Even lower for domain adaptation
    
    WARMUP_STEPS = 100
    MAX_SEQ_LENGTH = 512
    
    # LoRA (for Stage 2 & 3)
    USE_LORA = True
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # Logging
    LOG_EVERY = 10
    EVAL_EVERY = 100
    SAVE_EVERY = 500
    
    # Output
    OUTPUT_DIR = f"models/isro_multimodal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True


class MultimodalTrainer:
    """
    Trainer for ISRO Multimodal GPT
    Handles both pretraining and fine-tuning stages
    """
    
    def __init__(self, config, model, train_loader, val_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # Save config
        with open(os.path.join(config.OUTPUT_DIR, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2, default=str)
        
        # Initialize wandb (optional)
        self.use_wandb = False
        try:
            wandb.init(
                project="isro-multimodal-gpt",
                config=vars(config)
            )
            self.use_wandb = True
        except:
            print("wandb not available, skipping logging")
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def setup_optimizer_stage1(self):
        """
        Stage 1: Train only projection layer
        Freeze vision encoder and language model
        """
        print("\n=== STAGE 1 SETUP: Pretraining Projection Layer ===")
        
        # Freeze everything
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False
        
        for param in self.model.language_model.parameters():
            param.requires_grad = False
        
        # Only train projection
        for param in self.model.projector.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.LEARNING_RATE_STAGE1,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        return optimizer
    
    def setup_optimizer_stage2(self):
        """
        Stage 2: Fine-tune with LoRA
        Keep vision encoder frozen
        """
        print("\n=== STAGE 2 SETUP: Instruction Tuning with LoRA ===")
        
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Apply LoRA to language model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        
        self.model.language_model = get_peft_model(
            self.model.language_model,
            lora_config
        )
        
        # Vision encoder still frozen
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False
        
        # Projection trainable
        for param in self.model.projector.parameters():
            param.requires_grad = True
        
        # Print trainable parameters
        self.model.language_model.print_trainable_parameters()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.LEARNING_RATE_STAGE2,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        return optimizer
    
    def train_epoch(self, optimizer, scheduler, epoch, stage_name):
        """
        Train for one epoch
        
        Args:
            optimizer: Optimizer
            scheduler: LR scheduler
            epoch: Current epoch number
            stage_name: "Stage1", "Stage2", or "Stage3"
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"{stage_name} Epoch {epoch}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(self.config.DEVICE)
            prompts = batch['prompt']
            responses = batch['response']
            
            # Prepare labels (tokenize responses)
            labels = self.model.tokenizer(
                responses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.MAX_SEQ_LENGTH
            )['input_ids'].to(self.config.DEVICE)
            
            # Forward pass
            outputs = self.model(
                images=images,
                text_prompts=prompts,
                labels=labels
            )
            
            loss = outputs.loss / self.config.GRADIENT_ACCUMULATION
            
            # Backward pass
            loss.backward()
            
            # Update weights (gradient accumulation)
            if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Logging
            if self.global_step % self.config.LOG_EVERY == 0:
                if self.use_wandb:
                    wandb.log({
                        f'{stage_name}/loss': avg_loss,
                        f'{stage_name}/lr': scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
            
            # Validation
            if self.global_step % self.config.EVAL_EVERY == 0 and self.val_loader:
                val_loss = self.validate()
                print(f"\nValidation Loss: {val_loss:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        f'{stage_name}/val_loss': val_loss,
                        'global_step': self.global_step
                    })
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model')
                
                self.model.train()
            
            # Save checkpoint
            if self.global_step % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(f'checkpoint_{self.global_step}')
        
        return total_loss / num_batches
    
    def validate(self):
        """
        Validate on validation set
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.config.DEVICE)
                prompts = batch['prompt']
                responses = batch['response']
                
                labels = self.model.tokenizer(
                    responses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.MAX_SEQ_LENGTH
                )['input_ids'].to(self.config.DEVICE)
                
                outputs = self.model(
                    images=images,
                    text_prompts=prompts,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_checkpoint(self, name):
        """Save model checkpoint"""
        save_path = os.path.join(self.config.OUTPUT_DIR, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.language_model.save_pretrained(save_path)
        torch.save(self.model.projector.state_dict(), 
                  os.path.join(save_path, 'projector.pt'))
        
        # Save tokenizer
        self.model.tokenizer.save_pretrained(save_path)
        
        print(f"\n✓ Saved checkpoint: {save_path}")
    
    def train_stage1(self):
        """
        Stage 1: Pretrain projection layer
        """
        print("\n" + "="*70)
        print("STAGE 1: PRETRAINING PROJECTION LAYER")
        print("="*70)
        
        optimizer = self.setup_optimizer_stage1()
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.NUM_EPOCHS_STAGE1
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Train
        for epoch in range(self.config.NUM_EPOCHS_STAGE1):
            avg_loss = self.train_epoch(
                optimizer, scheduler, epoch + 1, "Stage1"
            )
            print(f"Stage 1 Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Save
        self.save_checkpoint('stage1_final')
    
    def train_stage2(self):
        """
        Stage 2: Instruction tuning
        """
        print("\n" + "="*70)
        print("STAGE 2: INSTRUCTION TUNING")
        print("="*70)
        
        optimizer = self.setup_optimizer_stage2()
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.NUM_EPOCHS_STAGE2
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Train
        for epoch in range(self.config.NUM_EPOCHS_STAGE2):
            avg_loss = self.train_epoch(
                optimizer, scheduler, epoch + 1, "Stage2"
            )
            print(f"Stage 2 Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Save
        self.save_checkpoint('stage2_final')
    
    def train_stage3(self, isro_train_loader, isro_val_loader):
        """
        Stage 3: ISRO-specific fine-tuning
        
        Args:
            isro_train_loader: ISRO satellite data loader
            isro_val_loader: ISRO validation data loader
        """
        print("\n" + "="*70)
        print("STAGE 3: ISRO-SPECIFIC FINE-TUNING")
        print("="*70)
        
        # Update data loaders
        self.train_loader = isro_train_loader
        self.val_loader = isro_val_loader
        
        # Use same setup as Stage 2 but lower LR
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.LEARNING_RATE_STAGE3,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.NUM_EPOCHS_STAGE3
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Train
        for epoch in range(self.config.NUM_EPOCHS_STAGE3):
            avg_loss = self.train_epoch(
                optimizer, scheduler, epoch + 1, "Stage3"
            )
            print(f"Stage 3 Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Save final model
        self.save_checkpoint('final_model')


def create_sample_data():
    """
    Create sample data for testing
    (Replace with actual ISRO data in production)
    """
    import json
    from pathlib import Path
    
    # Create directories
    Path("data/isro_images/train").mkdir(parents=True, exist_ok=True)
    Path("data/isro_images/val").mkdir(parents=True, exist_ok=True)
    Path("data/annotations").mkdir(parents=True, exist_ok=True)
    
    # Sample annotations
    train_annotations = [
        {
            "image_file": "sample1.tif",
            "prompt": "What type of land cover is visible in this RESOURCESAT image?",
            "response": "The image shows dense agricultural fields with high NDVI values, indicating healthy crop growth. The regular patterns suggest irrigated farmland typical of the Indo-Gangetic plains."
        },
        {
            "image_file": "sample2.tif",
            "prompt": "Identify the main features in this CARTOSAT image.",
            "response": "This high-resolution CARTOSAT image shows an urban area with distinct road networks, residential blocks, and commercial buildings. The organized grid pattern suggests planned urban development."
        }
    ]
    
    val_annotations = [
        {
            "image_file": "val1.tif",
            "prompt": "Analyze the water bodies in this image.",
            "response": "The image shows a reservoir or lake with clear water indicated by low NDVI and high NDWI values. Surrounding vegetation appears healthy."
        }
    ]
    
    # Save annotations
    with open("data/annotations/train.json", 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open("data/annotations/val.json", 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    print("Sample data created. Replace with actual ISRO data.")


def main():
    """Main training pipeline"""
    
    print("="*70)
    print("ISRO MULTIMODAL GPT - TRAINING PIPELINE")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠ No GPU detected - training will be slow!")
    
    # Create sample data (for testing)
    # create_sample_data()
    
    # Configuration
    config = Config()
    
    # Initialize model
    print("\nInitializing model...")
    from architecture import ISROMultimodalGPT
    
    model = ISROMultimodalGPT(
        vision_model_name=config.VISION_MODEL,
        language_model_name=config.LANGUAGE_MODEL,
        freeze_vision=True,
        freeze_language=True  # Will be unfrozen with LoRA in Stage 2
    )
    model = model.to(config.DEVICE)
    
    # Create data loaders
    print("\nLoading data...")
    from isro_data_loader import create_isro_dataloader
    
    # General pretraining data (Stage 1 & 2)
    train_loader = create_isro_dataloader(
        data_dir=config.TRAIN_DATA_DIR,
        annotation_file=config.TRAIN_ANNOTATIONS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = create_isro_dataloader(
        data_dir=config.VAL_DATA_DIR,
        annotation_file=config.VAL_ANNOTATIONS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # ISRO-specific data (Stage 3)
    isro_train_loader = create_isro_dataloader(
        data_dir="data/isro_specific/train",
        annotation_file="data/isro_specific/train.json",
        satellite_type='RESOURCESAT',
        sensor='LISS-III',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    isro_val_loader = create_isro_dataloader(
        data_dir="data/isro_specific/val",
        annotation_file="data/isro_specific/val.json",
        satellite_type='RESOURCESAT',
        sensor='LISS-III',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Initialize trainer
    trainer = MultimodalTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # STAGE 1: Pretrain projection layer
    trainer.train_stage1()
    
    # STAGE 2: Instruction tuning
    trainer.train_stage2()
    
    # STAGE 3: ISRO-specific fine-tuning
    trainer.train_stage3(isro_train_loader, isro_val_loader)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Final model saved to: {config.OUTPUT_DIR}/final_model")
    print("="*70)


if __name__ == "__main__":
    main()
