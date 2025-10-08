"""
TGVA (Texture-Guided Visual Attention) Module
Implements single-head cross-attention mechanism for RGB-Texture fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TGVAAttention(nn.Module):
    """
    Texture-Guided Visual Attention (TGVA) - Single-head Cross-Attention
    
    Architecture Details:
    - d_model = 128, h = 1 (single-head), d_h = 128, N = 1024 tokens
    - W_Q, W_K, W_V, W_O ∈ R^{128×128} with bias
    - Pre-MHA LayerNorm (ε=1e-5) and Post-add LayerNorm
    - Attention dropout: p_attn = 0.10
    - FFN dropout: p_ff = 0.10
    - FFN with GELU activation, d_ff = 512
    - Gated add-fusion with W_f ∈ R^{256×128}, learnable γ
    """
    
    def __init__(self, d_model=128, d_ff=512, p_attn=0.10, p_ff=0.10):
        super(TGVAAttention, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Pre-MHA Layer Normalization
        self.ln_pre = nn.LayerNorm(d_model, eps=1e-5)
        
        # Single-head attention projections with bias
        self.query_proj = nn.Linear(d_model, d_model, bias=True)
        self.key_proj = nn.Linear(d_model, d_model, bias=True)
        self.value_proj = nn.Linear(d_model, d_model, bias=True)
        self.output_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(p_attn)
        
        # Feed-Forward Network with GELU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p_ff),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p_ff)
        )
        
        # Post-add Layer Normalization
        self.ln_post = nn.LayerNorm(d_model, eps=1e-5)
        
        # Gated Add-Fusion
        self.fusion_gate = nn.Linear(d_model * 2, d_model, bias=True)
        self.gamma = nn.Parameter(torch.ones(1))
        
    def forward(self, rgb_features, texture_features):
        """
        Args:
            rgb_features: [B, N, C] - RGB query features
            texture_features: [B, N, C] - Texture key-value features
        
        Returns:
            fused_features: [B, N, 2C] - Fused features with doubled channels
            attention_weights: [B, N, N] - Attention weights
        """
        B, N, C = rgb_features.shape
        
        # Pre-MHA Layer Normalization
        rgb_norm = self.ln_pre(rgb_features)
        texture_norm = self.ln_pre(texture_features)
        
        # Project to Q, K, V
        Q = self.query_proj(rgb_norm)  # [B, N, C]
        K = self.key_proj(texture_norm)  # [B, N, C]
        V = self.value_proj(texture_norm)  # [B, N, C]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # [B, N, N]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [B, N, C]
        
        # Output projection
        context = self.output_proj(context)
        
        # Residual connection + FFN
        attended = rgb_features + context
        attended = attended + self.ffn(attended)
        
        # Post-add Layer Normalization
        attended = self.ln_post(attended)
        
        # Gated Add-Fusion
        concatenated = torch.cat([rgb_features, attended], dim=-1)  # [B, N, 2C]
        gate = torch.sigmoid(self.fusion_gate(concatenated))
        fused = self.gamma * (gate * attended) + (1 - gate) * rgb_features
        
        # Output doubled channels for fusion
        fused_output = torch.cat([fused, attended], dim=-1)  # [B, N, 2C]
        
        return fused_output, attention_weights


class TGVABlock(nn.Module):
    """
    TGVA Block for integration at ResBlock-2
    Handles spatial-to-sequence conversion and back
    """
    
    def __init__(self, in_channels=128, d_model=128, d_ff=512, p_attn=0.10, p_ff=0.10):
        super(TGVABlock, self).__init__()
        
        self.in_channels = in_channels
        self.d_model = d_model
        
        # TGVA Attention module
        self.tgva_attention = TGVAAttention(d_model, d_ff, p_attn, p_ff)
        
        # Projection back to spatial features (256 channels after fusion)
        self.spatial_projection = nn.Conv2d(d_model * 2, d_model * 2, kernel_size=1)
        
    def forward(self, rgb_spatial, texture_spatial):
        """
        Args:
            rgb_spatial: [B, C, H, W] - RGB features from ResBlock-2
            texture_spatial: [B, C, H, W] - Texture features from ResBlock-2
        
        Returns:
            fused_spatial: [B, 2C, H, W] - Fused features with doubled channels
            attention_weights: [B, N, N] - Attention weights
        """
        B, C, H, W = rgb_spatial.shape
        N = H * W  # Number of tokens (1024 for 32×32)
        
        # Convert spatial to sequence: [B, C, H, W] -> [B, N, C]
        rgb_seq = rgb_spatial.flatten(2).transpose(1, 2)  # [B, N, C]
        texture_seq = texture_spatial.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Apply TGVA attention
        fused_seq, attn_weights = self.tgva_attention(rgb_seq, texture_seq)  # [B, N, 2C]
        
        # Convert sequence back to spatial: [B, N, 2C] -> [B, 2C, H, W]
        fused_spatial = fused_seq.transpose(1, 2).reshape(B, C * 2, H, W)
        
        # Apply 1×1 convolution for refinement
        fused_spatial = self.spatial_projection(fused_spatial)
        
        return fused_spatial, attn_weights
