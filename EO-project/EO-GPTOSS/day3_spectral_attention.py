"""
Spectral Attention Module
Day 3 - Core Innovation

Learns which spectral bands are important for different image regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttentionModule(nn.Module):
    """
    Spectral Attention: Learn band importance per spatial location
    
    Key Idea:
    - Different land covers need different bands
    - Vegetation → high weight on NIR, Red Edge
    - Water → high weight on SWIR
    - Urban → balanced or SWIR-focused
    
    This module learns these patterns from data
    """
    
    def __init__(self, 
                 num_bands=13, 
                 embed_dim=768,
                 num_heads=8):
        """
        Args:
            num_bands: Number of spectral bands (13 for Sentinel-2)
            embed_dim: Embedding dimension (768 for ViT-base)
            num_heads: Number of attention heads (default: 8)
        """
        super().__init__()
        
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Learnable band embeddings
        # Each band gets a learnable vector representing its characteristics
        self.band_embeddings = nn.Parameter(
            torch.randn(num_bands, embed_dim)
        )
        
        # Query: "What information do I need?"
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # Key: "What spectral information is available?"
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        
        # Value: "The actual spectral information"
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Scaling factor for attention
        self.scale = self.head_dim ** -0.5
        
        # Initialize
        nn.init.trunc_normal_(self.band_embeddings, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: Input features (batch, num_patches, embed_dim)
            return_attention_weights: If True, also return attention weights
        
        Returns:
            Enhanced features (batch, num_patches, embed_dim)
            [Optional] Attention weights (batch, num_heads, num_patches, num_bands)
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # Generate queries from input features
        # "What spectral information does this patch need?"
        Q = self.query_proj(x)  # (batch, num_patches, embed_dim)
        
        # Generate keys from band embeddings
        # "What spectral bands are available?"
        band_keys = self.key_proj(self.band_embeddings)  # (num_bands, embed_dim)
        band_keys = band_keys.unsqueeze(0).expand(batch_size, -1, -1)
        # (batch, num_bands, embed_dim)
        
        # Generate values from band embeddings
        band_values = self.value_proj(self.band_embeddings)
        band_values = band_values.unsqueeze(0).expand(batch_size, -1, -1)
        # (batch, num_bands, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_patches, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch, num_heads, num_patches, head_dim)
        
        K = band_keys.view(batch_size, self.num_bands, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)  # (batch, num_heads, num_bands, head_dim)
        
        V = band_values.view(batch_size, self.num_bands, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)  # (batch, num_heads, num_bands, head_dim)
        
        # Compute attention scores
        # "How much should each patch attend to each spectral band?"
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # (batch, num_heads, num_patches, num_bands)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # (batch, num_heads, num_patches, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_patches, embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        # Add spectral attention as a refinement to original features
        output = x + attn_output
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output


class SimplifiedSpectralAttention(nn.Module):
    """
    Simplified version of spectral attention
    
    Use this if full multi-head attention is too complex
    or causes memory issues
    """
    
    def __init__(self, num_bands=13, embed_dim=768):
        super().__init__()
        
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        
        # Learnable band embeddings
        self.band_embeddings = nn.Parameter(
            torch.randn(num_bands, embed_dim)
        )
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_bands),
            nn.Softmax(dim=-1)
        )
        
        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch, num_patches, embed_dim)
        Returns:
            (batch, num_patches, embed_dim)
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # Compute attention weights for each patch
        # "Which bands matter for this patch?"
        attn_weights = self.attention(x)  # (batch, num_patches, num_bands)
        
        # Project band embeddings to values
        band_values = self.value_proj(self.band_embeddings)
        # (num_bands, embed_dim)
        
        # Apply attention to band embeddings
        # Weighted combination of band information
        attended = torch.matmul(attn_weights, band_values)
        # (batch, num_patches, embed_dim)
        
        # Residual connection
        output = x + attended
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output


if __name__ == "__main__":
    """Test spectral attention"""
    
    print("="*60)
    print("TESTING SPECTRAL ATTENTION MODULE")
    print("="*60)
    
    # Create dummy input (patch embeddings)
    batch_size = 2
    num_patches = 64  # 8×8 grid
    embed_dim = 768
    
    dummy_input = torch.randn(batch_size, num_patches, embed_dim)
    
    print(f"\nInput: {dummy_input.shape}")
    print("  (batch=2, patches=64, embed_dim=768)")
    
    # Test full multi-head attention
    print("\n1. Multi-Head Spectral Attention:")
    attn_module = SpectralAttentionModule(
        num_bands=13,
        embed_dim=768,
        num_heads=8
    )
    
    output, weights = attn_module(dummy_input, return_attention_weights=True)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights: {weights.shape}")
    print(f"     (batch, heads, patches, bands)")
    
    # Analyze attention patterns
    # Average attention across batch and heads
    avg_attention = weights.mean(dim=(0, 1))  # (num_patches, num_bands)
    
    print(f"\n   Average attention per band:")
    band_importance = avg_attention.mean(dim=0)  # (num_bands,)
    
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                  'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    
    for i, (name, importance) in enumerate(zip(band_names, band_importance)):
        print(f"     {name}: {importance:.4f}")
    
    # Test simplified attention
    print("\n2. Simplified Spectral Attention:")
    simple_attn = SimplifiedSpectralAttention(
        num_bands=13,
        embed_dim=768
    )
    
    output2, weights2 = simple_attn(dummy_input, return_attention_weights=True)
    
    print(f"   Output shape: {output2.shape}")
    print(f"   Attention weights: {weights2.shape}")
    
    # Parameter comparison
    params1 = sum(p.numel() for p in attn_module.parameters())
    params2 = sum(p.numel() for p in simple_attn.parameters())
    
    print(f"\n3. Parameter Count:")
    print(f"   Multi-head: {params1:,}")
    print(f"   Simplified: {params2:,}")
    print(f"   Ratio: {params1/params2:.1f}x")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("- Use multi-head for best performance")
    print("- Use simplified if memory is tight")
    print("="*60)
