import torch
import torch.nn as nn
from fastonn import SelfONN2d
from torch import Tensor


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

