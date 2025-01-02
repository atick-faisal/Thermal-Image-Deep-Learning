# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import torch
import torch.nn as nn
from fastonn import SelfONN2d
from torch import Tensor


class SingleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, instance_norm: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels) if instance_norm \
            else nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, instance_norm: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels) if instance_norm \
            else nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class SelfDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, instance_norm: bool = True) -> None:
        super().__init__()
        self.conv1 = SelfONN2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, q=3)
        self.conv2 = SelfONN2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, q=3)
        self.norm = nn.InstanceNorm2d(out_channels) if instance_norm \
            else nn.BatchNorm2d(out_channels)
        self.activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Reduce channel dimensions for efficiency
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Project and reshape
        query = self.query(x).view(batch_size, -1, height * width)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        # Compute attention
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return self.gamma * out + x


class SpatialAttention(nn.Module):
    def __init__(self, gating_channels, feature_channels, inter_channels=None):
        super().__init__()
        """
        Parameters:
            gating_channels (int): Number of channels in the gating signal
            feature_channels (int): Number of channels in the feature map
            inter_channels (int): Number of intermediate channels for dimension reduction
                                If None, will be set to feature_channels // 2
        """
        self.inter_channels = inter_channels or feature_channels // 2

        # Query transform for gating signal
        self.query = nn.Sequential(
            nn.Conv2d(gating_channels, self.inter_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.inter_channels)
        )

        # Key transform for feature map
        self.key = nn.Sequential(
            nn.Conv2d(feature_channels, self.inter_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.inter_channels)
        )

        # Final attention map generation
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, 1,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        """
        Parameters:
            g (torch.Tensor): Gating signal from coarser scale [B, gating_channels, H, W]
            x (torch.Tensor): Feature map from same scale [B, feature_channels, H, W]

        Returns:
            torch.Tensor: Attended feature map [B, feature_channels, H, W]
        """
        batch_size = x.size(0)

        # Project inputs to intermediate space
        query = self.query(g)  # [B, inter_channels, H, W]
        key = self.key(x)  # [B, inter_channels, H, W]

        # Element-wise addition of query and key
        energy = query + key  # [B, inter_channels, H, W]

        # Generate attention map
        attention_map = self.attention(energy)  # [B, 1, H, W]

        # Apply attention
        out = x * attention_map

        return out
