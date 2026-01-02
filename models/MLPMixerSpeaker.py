#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
MLP-Mixer Architecture for Speaker Verification with Knowledge Distillation

Adapted from: "A Speaker Verification System Based on a Modified MLP-Mixer 
Student Model for Transformer Compression"

Key Innovations:
1. ID Convolution: 1D temporal conv before token-mixing (captures local dependencies)
2. Max-Feature-Map (MFM): Gating mechanism before channel-mixing
3. Grouped projections: Reduces parameters in mixing MLPs
4. Knowledge distillation: Student learns from LSTM+Autoencoder teacher

Architecture Flow:
    Mel-spectrogram (80 × T) 
        → CNN Front-end (80 → hidden_dim)
        → MLP-Mixer Blocks (N layers)
            - ID Conv (temporal context)
            - Token Mixing MLP (across time)
            - MFM activation
            - Channel Mixing MLP (across features)
        → ASP Pooling
        → Embedding (512-dim)

Performance Target:
- Parameters: ~1.5M (60% reduction vs LSTM+AE's 3.87M)
- Speed: 2-3× faster inference (no sequential LSTM)
- EER: 10-11% (distillation typically 5-10% worse than teacher's 9.68%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MaxFeatureMap(nn.Module):
    """
    Max-Feature-Map (MFM) activation function
    
    Paper motivation: "MFM acts as a feature selector that emphasizes 
    speaker-discriminative frequency bands while suppressing redundant information"
    
    Operation: Split channels into two groups, take element-wise maximum
    Input: [B, 2C, T] → Output: [B, C, T]
    """
    def __init__(self):
        super(MaxFeatureMap, self).__init__()
    
    def forward(self, x):
        # Split into two groups and take max
        # x: [batch, channels, time]
        out = torch.chunk(x, 2, dim=1)  # Split into 2 groups
        return torch.max(out[0], out[1])  # Element-wise max


class IDConv1d(nn.Module):
    """
    Identity-enhanced 1D Convolution
    
    Paper motivation: "ID Conv helps capture local temporal dependencies 
    (correlations between adjacent frames) which standard MLPs might miss"
    
    Combines identity connection with 1D convolution for local context
    """
    def __init__(self, channels, kernel_size=3):
        super(IDConv1d, self).__init__()
        self.conv = nn.Conv1d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2,
            groups=channels,  # Depthwise conv for efficiency
            bias=False
        )
        self.bn = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        # x: [batch, channels, time]
        return x + self.bn(self.conv(x))  # Residual connection


class TokenMixingMLP(nn.Module):
    """
    Token-Mixing MLP (mixes information across time dimension)
    
    Paper: "Blends temporal information across different time frames"
    
    Uses grouped linear projections for parameter efficiency
    """
    def __init__(self, num_tokens, hidden_dim, expansion_factor=4, groups=4):
        super(TokenMixingMLP, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.groups = groups
        
        # Two-layer MLP with expansion
        expanded_dim = hidden_dim * expansion_factor
        
        # Grouped projection for efficiency (paper's modification)
        self.fc1 = nn.Conv1d(hidden_dim, expanded_dim, kernel_size=1, groups=groups)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv1d(expanded_dim, hidden_dim, kernel_size=1, groups=groups)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: [batch, channels, time]
        # Transpose to mix time dimension
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = x.transpose(1, 2)  # Back to [batch, channels, time] for conv1d
        
        # MLP operations
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        return out


class ChannelMixingMLP(nn.Module):
    """
    Channel-Mixing MLP (mixes information across feature dimension)
    
    Paper: "Blends feature information across different frequency bands"
    
    Preceded by MFM activation for speaker-discriminative feature selection
    """
    def __init__(self, channels, expansion_factor=4, groups=4):
        super(ChannelMixingMLP, self).__init__()
        
        expanded_dim = channels * expansion_factor * 2  # *2 for MFM
        
        # MFM doubles channels, then reduces via max operation
        self.fc1 = nn.Linear(channels, expanded_dim)
        self.mfm = MaxFeatureMap()  # Reduces back to channels * expansion_factor
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(channels * expansion_factor, channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: [batch, channels, time]
        x = x.transpose(1, 2)  # [batch, time, channels]
        
        # MLP with MFM activation
        out = self.fc1(x)
        out = out.transpose(1, 2)  # [batch, channels, time] for MFM
        out = self.mfm(out)  # Speaker-discriminative feature selection
        out = out.transpose(1, 2)  # Back to [batch, time, channels]
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        return out.transpose(1, 2)  # [batch, channels, time]


class MLPMixerBlock(nn.Module):
    """
    Modified MLP-Mixer Block for Speaker Verification
    
    Paper's modifications:
    1. ID Conv before token mixing (local temporal context)
    2. MFM in channel mixing (speaker-discriminative selection)
    3. Grouped projections (parameter efficiency)
    
    Architecture:
        Input → LayerNorm → ID Conv → Token Mixing → Residual
              → LayerNorm → Channel Mixing (with MFM) → Residual
    """
    def __init__(self, num_tokens, channels, expansion_factor=4, groups=4):
        super(MLPMixerBlock, self).__init__()
        
        # Pre-norm architecture (better training stability)
        self.norm1 = nn.LayerNorm(channels)
        self.id_conv = IDConv1d(channels, kernel_size=3)
        self.token_mixing = TokenMixingMLP(num_tokens, channels, expansion_factor, groups)
        
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mixing = ChannelMixingMLP(channels, expansion_factor, groups)
    
    def forward(self, x):
        # x: [batch, channels, time]
        
        # Token mixing path with ID Conv
        residual = x
        x_norm = self.norm1(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm on channel dim
        x_norm = self.id_conv(x_norm)  # Local temporal context
        x = residual + self.token_mixing(x_norm)
        
        # Channel mixing path with MFM
        residual = x
        x_norm = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = residual + self.channel_mixing(x_norm)
        
        return x


class AttentiveStatsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP)
    
    Aggregates frame-level features to utterance-level embedding
    Outputs: mean + std (2× channels)
    """
    def __init__(self, channels):
        super(AttentiveStatsPooling, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x):
        # x: [batch, channels, time]
        w = self.attention(x)
        
        # Weighted statistics
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        
        # Concatenate mean and std
        return torch.cat([mu, sg], dim=1)


class MLPMixerSpeakerNet(nn.Module):
    """
    MLP-Mixer Student Model for Speaker Verification
    
    Student learns from LSTM+Autoencoder teacher via knowledge distillation
    
    Args:
        nOut: Embedding dimension (default: 512)
        n_mels: Number of mel-filterbanks (default: 80)
        hidden_dim: Hidden dimension of MLP-Mixer (default: 256)
        num_blocks: Number of MLP-Mixer blocks (default: 8)
        expansion_factor: MLP expansion ratio (default: 4)
        groups: Grouped projection groups (default: 4)
        log_input: Apply log to mel-spectrogram (default: True)
    """
    def __init__(self, nOut=512, n_mels=80, hidden_dim=256, num_blocks=8,
                 expansion_factor=4, groups=4, log_input=True, **kwargs):
        super(MLPMixerSpeakerNet, self).__init__()
        
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.log_input = log_input
        
        # Mel-spectrogram extraction (consistent with other models)
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn=torch.hamming_window,
            n_mels=n_mels
        )
        
        # CNN Front-end: Projects mel-spectrogram to hidden dimension
        # Paper: "CNN front-end to extract features from waveform"
        # Adaptation: We use it to project mel features to MLP-Mixer space
        self.frontend = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # MLP-Mixer Encoder: Stack of mixing blocks
        self.mixer_blocks = nn.ModuleList([
            MLPMixerBlock(
                num_tokens=None,  # Dynamic based on input length
                channels=hidden_dim,
                expansion_factor=expansion_factor,
                groups=groups
            )
            for _ in range(num_blocks)
        ])
        
        # Layer normalization after encoder
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        
        # Attentive Statistics Pooling
        self.pooling = AttentiveStatsPooling(hidden_dim)
        
        # Final embedding layer
        # Input: hidden_dim * 2 (mean + std from ASP)
        self.fc = nn.Linear(hidden_dim * 2, nOut)
        self.bn = nn.BatchNorm1d(nOut)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Raw audio waveform [batch, samples]
        
        Returns:
            embeddings: Speaker embeddings [batch, nOut]
        """
        # Extract mel-spectrogram
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
                if self.log_input:
                    x = x.log()
                x = self.instancenorm(x)
        
        # x: [batch, n_mels, time]
        
        # CNN front-end projection
        x = self.frontend(x)
        # x: [batch, hidden_dim, time]
        
        # MLP-Mixer encoder blocks
        for block in self.mixer_blocks:
            x = block(x)
        
        # Post-encoder normalization
        x = self.encoder_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # Attentive statistics pooling
        x = self.pooling(x)
        # x: [batch, hidden_dim * 2]
        
        # Final embedding
        x = self.fc(x)
        x = self.bn(x)
        
        return x


def MainModel(nOut=512, **kwargs):
    """
    Main entry point for MLP-Mixer speaker network
    
    Args:
        nOut: Embedding dimension (default: 512)
        n_mels: Number of mel-filterbanks (default: 80)
        hidden_dim: MLP-Mixer hidden dimension (default: 256)
        num_blocks: Number of MLP-Mixer blocks (default: 8)
        expansion_factor: MLP expansion ratio (default: 4)
        groups: Grouped projection groups (default: 4)
    
    Returns:
        model: MLPMixerSpeakerNet instance
    """
    print('Creating MLP-Mixer Speaker Network (Student Model)')
    
    model = MLPMixerSpeakerNet(nOut=nOut, **kwargs)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)')
    
    # Architecture summary
    print(f'\nArchitecture Configuration:')
    print(f'  - Mel filterbanks: {kwargs.get("n_mels", 80)}')
    print(f'  - Hidden dimension: {kwargs.get("hidden_dim", 256)}')
    print(f'  - MLP-Mixer blocks: {kwargs.get("num_blocks", 8)}')
    print(f'  - Expansion factor: {kwargs.get("expansion_factor", 4)}')
    print(f'  - Grouped projections: {kwargs.get("groups", 4)} groups')
    print(f'  - Embedding dimension: {nOut}')
    print(f'  - Pooling: ASP (mean + std)')
    
    return model
