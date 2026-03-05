"""
Spectral Vision Transformer (SpectralViT)
Day 3 - Complete Architecture

Vision Transformer adapted for 13-band multispectral input
"""

import torch
import torch.nn as nn
from day3_patch_embedding import SpectralPatchEmbedding
from day3_spectral_attention import SpectralAttentionModule

class SpectralViT(nn.Module):
    """
    Vision Transformer for Multispectral Satellite Imagery
    
    Architecture:
    1. Spectral Patch Embedding (13 bands → patches)
    2. Positional Encoding
    3. Spectral Attention Module (our innovation)
    4. Standard ViT Transformer Blocks
    5. Output embeddings for language model
    """
    
    def __init__(self,
                 in_channels=13,
                 image_size=64,
                 patch_size=8,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 use_spectral_attention=True,
                 dropout=0.1):
        """
        Args:
            in_channels: Number of spectral bands (default: 13)
            image_size: Input image size (default: 64×64)
            patch_size: Patch size (default: 8×8)
            embed_dim: Embedding dimension (default: 768)
            depth: Number of transformer blocks (default: 12)
            num_heads: Number of attention heads (default: 12)
            mlp_ratio: MLP hidden dim ratio (default: 4.0)
            use_spectral_attention: Use spectral attention module
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = SpectralPatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=image_size
        )
        
        # CLS token (for classification tasks)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Spectral attention module (our innovation)
        self.use_spectral_attention = use_spectral_attention
        if use_spectral_attention:
            self.spectral_attention = SpectralAttentionModule(
                num_bands=in_channels,
                embed_dim=embed_dim,
                num_heads=8  # Can be different from transformer heads
            )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor (batch, 13, 64, 64)
            return_attention: If True, return spectral attention weights
        
        Returns:
            Output features (batch, num_patches+1, embed_dim)
            [Optional] Spectral attention weights
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # (batch, num_patches+1, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply spectral attention
        spectral_attn_weights = None
        if self.use_spectral_attention:
            if return_attention:
                x, spectral_attn_weights = self.spectral_attention(
                    x, 
                    return_attention_weights=True
                )
            else:
                x = self.spectral_attention(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        if return_attention:
            return x, spectral_attn_weights
        else:
            return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer Block
    
    - Multi-head self-attention
    - MLP with GELU activation
    - Residual connections
    - Layer normalization
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=dropout
        )
    
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )[0]
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron
    
    - Linear → GELU → Dropout → Linear → Dropout
    """
    
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


if __name__ == "__main__":
    """Test SpectralViT"""
    
    print("="*60)
    print("TESTING SPECTRAL VISION TRANSFORMER")
    print("="*60)
    
    # Create model
    model = SpectralViT(
        in_channels=13,
        image_size=64,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        use_spectral_attention=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB (FP32)")
    print(f"  Model size: ~{total_params * 2 / (1024**2):.1f} MB (FP16)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 13, 64, 64)
    
    print(f"  Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output, attn_weights = model(dummy_input, return_attention=True)
    
    print(f"  Output shape: {output.shape}")
    print(f"    Expected: (2, 65, 768)")
    print(f"    - 65 tokens = 1 CLS + 64 patches")
    print(f"    - 768 = embedding dimension")
    
    if attn_weights is not None:
        print(f"  Spectral attention shape: {attn_weights.shape}")
    
    # Test GPU compatibility
    if torch.cuda.is_available():
        print("\nTesting GPU transfer...")
        model_gpu = model.cuda()
        input_gpu = dummy_input.cuda()
        
        with torch.no_grad():
            output_gpu = model_gpu(input_gpu)
        
        print(f"  ✓ GPU forward pass successful")
        print(f"  Output shape: {output_gpu.shape}")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated() / (1024**2)
        print(f"  GPU memory: {allocated:.1f} MB")
    
    print("\n" + "="*60)
    print("✓ SPECTRAL VIT TEST COMPLETE")
    print("="*60)
    
    # Comparison with standard ViT
    print("\nComparison with Standard ViT:")
    print(f"  Standard ViT-Base:  ~86M parameters (3 input channels)")
    print(f"  Our SpectralViT:    ~{total_params/1e6:.1f}M parameters (13 input channels)")
    print(f"  Overhead:           ~{(total_params - 86e6)/1e6:.1f}M parameters")
    print(f"    (mainly from spectral attention + modified patch embedding)")
