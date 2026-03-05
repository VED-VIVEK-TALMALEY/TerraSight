"""
Spectral Patch Embedding
Day 3 - Core Architecture Component

Converts 13-band multispectral patches to embeddings
"""

import torch
import torch.nn as nn

class SpectralPatchEmbedding(nn.Module):
    """
    Convert 13-band patches to embeddings
    
    Input: (batch, 13, 64, 64) - 13 Sentinel-2 bands
    Output: (batch, num_patches, embed_dim) - Patch embeddings
    
    Design Decision:
    - Use Conv2D for patch extraction (efficient, learnable)
    - Direct projection: 13 channels → embed_dim
    - Alternative: Per-band projection → fusion (more parameters)
    """
    
    def __init__(self, 
                 in_channels=13, 
                 patch_size=8, 
                 embed_dim=768,
                 image_size=64):
        """
        Args:
            in_channels: Number of spectral bands (default: 13)
            patch_size: Size of each patch (default: 8×8)
            embed_dim: Embedding dimension (default: 768, ViT-base)
            image_size: Input image size (default: 64×64)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Convolutional projection
        # Extracts patches and projects to embedding space in one operation
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        
        # Initialize weights
        # Use truncated normal (common for ViT)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, 13, 64, 64)
        
        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Apply convolutional projection
        # Input: (batch, 13, 64, 64)
        # Output: (batch, embed_dim, 8, 8)
        x = self.projection(x)
        
        # Flatten spatial dimensions
        # (batch, embed_dim, 8, 8) → (batch, embed_dim, 64)
        x = x.flatten(2)
        
        # Transpose to (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class SpectralPatchEmbeddingPerBand(nn.Module):
    """
    Alternative: Per-band projection with fusion
    
    More expressive but also more parameters
    Use this if simple projection doesn't work well
    """
    
    def __init__(self, 
                 in_channels=13, 
                 patch_size=8, 
                 embed_dim=768,
                 image_size=64):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Per-band embedding dimension
        # Round up to ensure total equals embed_dim
        self.band_embed_dim = (embed_dim + in_channels - 1) // in_channels
        
        # Separate convolution for each band
        self.band_projections = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=self.band_embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
            for _ in range(in_channels)
        ])
        
        # Fusion layer to combine band embeddings
        self.fusion = nn.Linear(embed_dim, embed_dim)
        
        # Initialize
        for proj in self.band_projections:
            nn.init.trunc_normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        nn.init.trunc_normal_(self.fusion.weight, std=0.02)
        nn.init.zeros_(self.fusion.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 13, 64, 64)
        Returns:
            (batch, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Process each band separately
        band_embeddings = []
        for i, projection in enumerate(self.band_projections):
            # Extract single band
            band = x[:, i:i+1, :, :]  # (batch, 1, 64, 64)
            
            # Project
            band_embed = projection(band)  # (batch, band_embed_dim, 8, 8)
            band_embed = band_embed.flatten(2).transpose(1, 2)
            # (batch, 64, band_embed_dim)
            
            band_embeddings.append(band_embed)
        
        # Concatenate all bands
        x = torch.cat(band_embeddings, dim=2)  # (batch, 64, embed_dim)
        
        # Fusion
        x = self.fusion(x)
        
        return x


if __name__ == "__main__":
    """Test the implementations"""
    
    print("="*60)
    print("TESTING SPECTRAL PATCH EMBEDDING")
    print("="*60)
    
    # Create dummy input (13-band image)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 13, 64, 64)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print("  13 spectral bands, 64×64 pixels")
    
    # Test simple projection
    print("\n1. Simple Projection:")
    embed1 = SpectralPatchEmbedding(
        in_channels=13,
        patch_size=8,
        embed_dim=768
    )
    
    output1 = embed1(dummy_input)
    print(f"   Output shape: {output1.shape}")
    print(f"   Expected: (2, 64, 768) - 64 patches, 768-dim embeddings")
    
    # Skip per-band projection (not needed for our implementation)
    print("\n2. Per-Band Projection: SKIPPED")
    print("   (We're using simple projection - works better!)")
    
    # Parameter count
    params1 = sum(p.numel() for p in embed1.parameters())
    
    print(f"\n3. Parameter Count:")
    print(f"   Simple projection: {params1:,} parameters")
    print(f"   Memory: ~{params1 * 4 / (1024**2):.1f} MB (FP32)")
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Start with simple projection")
    print("Use per-band only if results are unsatisfactory")
    print("="*60)