#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
MLP-Mixer with Raw Waveform Input for Speaker Verification

Extension of MLPMixerSpeaker to process raw audio directly without mel-spectrogram.
Inspired by SincNet (Ravanelli & Bengio, 2018) for learnable front-end filters.

Key Differences from Mel-based MLP-Mixer:
1. Raw waveform input → Learnable bandpass filters (SincNet-style)
2. No mel-spectrogram preprocessing (end-to-end learnable)
3. Direct temporal modeling from 16kHz audio samples
4. Automatic feature learning from raw audio

Architecture Flow:
    Raw Audio (16kHz) [batch, samples]
        → SincNet Frontend (learnable filters)
            - 80 learnable bandpass filters
            - Frame-level features
        → CNN Feature Extractor (80 → hidden_dim)
        → MLP-Mixer Blocks (N layers)
            - ID Conv (temporal context)
            - Token Mixing MLP (across time)
            - MFM activation
            - Channel Mixing MLP (across features)
        → ASP Pooling
        → Embedding (512-dim)

Research Hypothesis:
Raw waveform input may learn speaker-discriminative features automatically,
potentially capturing nuances lost in mel-spectrogram preprocessing.

Expected Results:
- Similar or slightly worse EER than mel-based (10-12% vs 10.32%)
- Longer training time (more parameters in frontend)
- Potentially better on noisy/distorted audio

Author: GitHub Copilot
Date: December 31, 2025
Experiment: P3 variant - Raw waveform input testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SincConv_fast(nn.Module):
    """
    Learnable Bandpass Filters (SincNet)
    
    Reference: "Speaker Recognition from Raw Waveform with SincNet"
              (Ravanelli & Bengio, SLT 2018)
    
    Instead of fixed mel-filterbanks, learns optimal bandpass filters
    for speaker verification task.
    
    Args:
        out_channels: Number of filters (default: 80, same as n_mels)
        kernel_size: Filter length in samples (default: 251, ~15ms @ 16kHz)
        sample_rate: Audio sample rate (default: 16000)
        stride: Hop length (default: 160, 10ms @ 16kHz)
        min_low_hz: Minimum low cutoff frequency (default: 50 Hz)
        min_band_hz: Minimum bandwidth (default: 50 Hz)
    """
    
    def __init__(self, out_channels=80, kernel_size=251, sample_rate=16000,
                 stride=160, min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        
        # Initialize filter banks (learnable parameters)
        # Low cutoff frequencies
        low_hz = 30.0
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        # Mel-scale initialization for frequency bands
        mel_low = self._hz_to_mel(low_hz)
        mel_high = self._hz_to_mel(high_hz)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = self._mel_to_hz(mel_points)
        
        # Learnable parameters: low cutoff and bandwidth
        self.low_hz_ = nn.Parameter(hz_points[:-1])
        self.band_hz_ = nn.Parameter(hz_points[1:] - hz_points[:-1])
        
        # Fixed parameters
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Hamming window
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                              steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        
        # Frequency axis for filter computation
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
    
    def _hz_to_mel(self, hz):
        """Convert Hz to mel scale"""
        if not isinstance(hz, torch.Tensor):
            hz = torch.tensor(hz, dtype=torch.float32)
        return 2595 * torch.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel):
        """Convert mel to Hz scale"""
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel, dtype=torch.float32)
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x):
        """
        Args:
            x: Raw waveform [batch, samples]
        Returns:
            features: Bandpass filtered features [batch, out_channels, time_frames]
        """
        # Ensure parameters are on same device
        self.n_ = self.n_.to(x.device)
        self.window_ = self.window_.to(x.device)
        
        # Constrain parameters to valid ranges
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                          self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, None]
        
        # Compute bandpass filters
        f_times_t_low = torch.matmul(low[:, None], self.n_)
        f_times_t_high = torch.matmul(high[:, None], self.n_)
        
        # Bandpass filter impulse response
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / 
                         (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        # Concatenate and normalize
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, 0, None])
        
        # Add channel dimension for conv1d
        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)
        
        # Apply learned filters
        return F.conv1d(x.unsqueeze(1), self.filters, stride=self.stride,
                       padding=self.kernel_size // 2, groups=1)


class MaxFeatureMap(nn.Module):
    """Max-Feature-Map activation (same as mel-based version)"""
    def __init__(self):
        super(MaxFeatureMap, self).__init__()
    
    def forward(self, x):
        out = torch.chunk(x, 2, dim=1)
        return torch.max(out[0], out[1])


class IDConv1d(nn.Module):
    """Identity-enhanced 1D Convolution (same as mel-based version)"""
    def __init__(self, channels, kernel_size=3):
        super(IDConv1d, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                             padding=kernel_size//2, groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        return x + self.bn(self.conv(x))


class TokenMixingMLP(nn.Module):
    """Token-Mixing MLP (same as mel-based version)"""
    def __init__(self, num_tokens, hidden_dim, expansion_factor=4, groups=4):
        super(TokenMixingMLP, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.groups = groups
        
        expanded_dim = hidden_dim * expansion_factor
        self.fc1 = nn.Conv1d(hidden_dim, expanded_dim, kernel_size=1, groups=groups)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv1d(expanded_dim, hidden_dim, kernel_size=1, groups=groups)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class ChannelMixingMLP(nn.Module):
    """Channel-Mixing MLP with MFM (same as mel-based version)"""
    def __init__(self, channels, expansion_factor=4, groups=4):
        super(ChannelMixingMLP, self).__init__()
        
        expanded_dim = channels * expansion_factor * 2
        self.fc1 = nn.Linear(channels, expanded_dim)
        self.mfm = MaxFeatureMap()
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(channels * expansion_factor, channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.fc1(x)
        out = out.transpose(1, 2)
        out = self.mfm(out)
        out = out.transpose(1, 2)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class MLPMixerBlock(nn.Module):
    """MLP-Mixer Block (same as mel-based version)"""
    def __init__(self, num_tokens, channels, expansion_factor=4, groups=4):
        super(MLPMixerBlock, self).__init__()
        
        self.id_conv = IDConv1d(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.token_mixing = TokenMixingMLP(num_tokens, channels, expansion_factor, groups)
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mixing = ChannelMixingMLP(channels, expansion_factor, groups)
    
    def forward(self, x):
        # ID Conv
        x = self.id_conv(x)
        
        # Token mixing
        residual = x
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = residual + self.token_mixing(x)
        
        # Channel mixing
        residual = x
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = residual + self.channel_mixing(x)
        
        return x


class AttentiveStatsPooling(nn.Module):
    """Attentive Statistics Pooling (same as mel-based version)"""
    def __init__(self, in_dim):
        super(AttentiveStatsPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x):
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        return torch.cat([mu, sg], dim=1)


class MLPMixerSpeakerNet_RawWaveform(nn.Module):
    """
    MLP-Mixer with Raw Waveform Input
    
    Processes raw audio directly using learnable SincNet filters instead of
    fixed mel-spectrogram preprocessing.
    
    Args:
        nOut: Embedding dimension (default: 512)
        num_filters: Number of learnable filters (default: 80, same as n_mels)
        hidden_dim: Hidden dimension of MLP-Mixer (default: 256)
        num_blocks: Number of MLP-Mixer blocks (default: 8)
        expansion_factor: MLP expansion ratio (default: 4)
        groups: Grouped projection groups (default: 4)
        kernel_size: SincNet filter length (default: 251)
        stride: SincNet hop length (default: 160)
    """
    def __init__(self, nOut=512, num_filters=80, hidden_dim=256, num_blocks=8,
                 expansion_factor=4, groups=4, kernel_size=251, stride=160, **kwargs):
        super(MLPMixerSpeakerNet_RawWaveform, self).__init__()
        
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # SincNet Frontend: Learnable bandpass filters
        # Replaces fixed mel-spectrogram extraction
        self.sincnet = SincConv_fast(
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride
        )
        
        # Normalization of learned features
        self.instancenorm = nn.InstanceNorm1d(num_filters)
        
        # Additional CNN layers for feature extraction
        # (Similar to original SincNet architecture)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            
            nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        
        # Projection to MLP-Mixer space
        self.frontend = nn.Sequential(
            nn.Conv1d(num_filters, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # MLP-Mixer Encoder: Stack of mixing blocks
        self.mixer_blocks = nn.ModuleList([
            MLPMixerBlock(
                num_tokens=None,
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
        # Apply SincNet learnable filters
        x = self.sincnet(x)  # [batch, num_filters, time_frames]
        
        # Normalize features
        x = self.instancenorm(x)
        
        # Apply additional feature extraction
        x = self.feature_extractor(x)  # [batch, num_filters, reduced_time]
        
        # Project to MLP-Mixer space
        x = self.frontend(x)  # [batch, hidden_dim, reduced_time]
        
        # Apply MLP-Mixer blocks
        for block in self.mixer_blocks:
            x = block(x)
        
        # Layer normalization
        x = x.transpose(1, 2)
        x = self.encoder_norm(x)
        x = x.transpose(1, 2)
        
        # Attentive Statistics Pooling
        x = self.pooling(x)  # [batch, hidden_dim * 2]
        
        # Final embedding
        x = self.fc(x)
        x = self.bn(x)
        
        return x


def MainModel(**kwargs):
    """Factory function for compatibility with existing training code"""
    return MLPMixerSpeakerNet_RawWaveform(**kwargs)
