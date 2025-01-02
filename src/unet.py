# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import torch
import torch.nn as nn
from fastonn import SelfONN2d
from torch import Tensor

from networks import DoubleConv, SelfDoubleConv, SelfAttention


class UNetRegressor(nn.Module):
    def __init__(self, init_features: int = 64) -> None:
        super().__init__()

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(3, init_features)
        self.enc2 = DoubleConv(init_features, init_features * 2)
        self.enc3 = DoubleConv(init_features * 2, init_features * 4)
        self.enc4 = DoubleConv(init_features * 4, init_features * 8)
        self.enc5 = DoubleConv(init_features * 8, init_features * 16)

        # Decoder (Expanding Path)
        self.up4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2,
                                      stride=2)
        self.dec4 = DoubleConv(init_features * 16, init_features * 8)

        self.up3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(init_features * 8, init_features * 4)

        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(init_features * 4, init_features * 2)

        self.up1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(init_features * 2, init_features)

        self.final_conv = nn.Conv2d(init_features, 3, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool(enc4)

        # Bridge
        enc5 = self.enc5(pool4)

        # Decoder
        up4 = self.up4(enc5)
        concat4 = torch.cat([enc4, up4], dim=1)
        dec4 = self.dec4(concat4)

        up3 = self.up3(dec4)
        concat3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.dec3(concat3)

        up2 = self.up2(dec3)
        concat2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.dec2(concat2)

        up1 = self.up1(dec2)
        concat1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.dec1(concat1)

        return nn.functional.tanh(self.final_conv(dec1))


class SelfUNetRegressor(nn.Module):
    def __init__(self, init_features: int = 16) -> None:
        super().__init__()

        # Encoder (Contracting Path)
        self.enc1 = SelfDoubleConv(3, init_features)
        self.enc2 = SelfDoubleConv(init_features, init_features * 2)
        self.enc3 = SelfDoubleConv(init_features * 2, init_features * 4)
        self.enc4 = SelfDoubleConv(init_features * 4, init_features * 8)
        self.enc5 = SelfDoubleConv(init_features * 8, init_features * 16)

        # Decoder (Expanding Path)
        self.up4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2,
                                      stride=2)
        self.dec4 = SelfDoubleConv(init_features * 16, init_features * 8)

        self.up3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.dec3 = SelfDoubleConv(init_features * 8, init_features * 4)

        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec2 = SelfDoubleConv(init_features * 4, init_features * 2)

        self.up1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec1 = SelfDoubleConv(init_features * 2, init_features)

        self.final_conv = SelfONN2d(init_features, 3, kernel_size=1, q=3)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool(enc4)

        # Bridge
        enc5 = self.enc5(pool4)

        # Decoder
        up4 = self.up4(enc5)
        concat4 = torch.cat([enc4, up4], dim=1)
        dec4 = self.dec4(concat4)

        up3 = self.up3(dec4)
        concat3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.dec3(concat3)

        up2 = self.up2(dec3)
        concat2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.dec2(concat2)

        up1 = self.up1(dec2)
        concat1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.dec1(concat1)

        final = self.final_conv(dec1)

        return nn.functional.tanh(final)


class AttentionUNetRegressor(nn.Module):
    def __init__(self, init_features: int = 64) -> None:
        super().__init__()

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(3, init_features)
        self.enc2 = DoubleConv(init_features, init_features * 2)
        self.enc3 = DoubleConv(init_features * 2, init_features * 4)
        self.enc4 = DoubleConv(init_features * 4, init_features * 8)
        self.enc5 = DoubleConv(init_features * 8, init_features * 16)

        # Add self-attention at bottleneck (16x16 feature maps)
        self.bottleneck_attention = SelfAttention(init_features * 16)

        # Add self-attention in decoder path (32x32 feature maps)
        self.decoder_attention = SelfAttention(init_features * 8)

        # Decoder (Expanding Path)
        self.up4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2,
                                      stride=2)
        self.dec4 = DoubleConv(init_features * 16, init_features * 8)

        self.up3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(init_features * 8, init_features * 4)

        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(init_features * 4, init_features * 2)

        self.up1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(init_features * 2, init_features)

        self.final_conv = nn.Conv2d(init_features, 3, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        enc1 = self.enc1(x)  # 256x256
        pool1 = self.pool(enc1)

        enc2 = self.enc2(pool1)  # 128x128
        pool2 = self.pool(enc2)

        enc3 = self.enc3(pool2)  # 64x64
        pool3 = self.pool(enc3)

        enc4 = self.enc4(pool3)  # 32x32
        pool4 = self.pool(enc4)

        # Bridge with attention
        enc5 = self.enc5(pool4)  # 16x16
        enc5 = self.bottleneck_attention(enc5)  # Apply attention at bottleneck

        # Decoder
        up4 = self.up4(enc5)
        concat4 = torch.cat([enc4, up4], dim=1)
        dec4 = self.dec4(concat4)
        dec4 = self.decoder_attention(dec4)  # Apply attention in decoder path

        up3 = self.up3(dec4)
        concat3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.dec3(concat3)

        up2 = self.up2(dec3)
        concat2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.dec2(concat2)

        up1 = self.up1(dec2)
        concat1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.dec1(concat1)

        return nn.functional.tanh(self.final_conv(dec1))


def export_to_onnx(model, save_path='unet.onnx', input_shape=(1, 3, 256, 256)):
    """
    Export PyTorch model to ONNX format

    Args:
        model: PyTorch model
        save_path: Path to save the ONNX file
        input_shape: Input tensor shape (batch_size, channels, height, width)
    """
    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        save_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},  # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {save_path}")


if __name__ == "__main__":
    # Initialize model
    model = UNetRegressor()

    # Set model to evaluation mode
    model.eval()

    # Export model
    export_to_onnx(model)

    print("\nTo visualize the model:")
    print("1. Install Netron: https://netron.app/")
    print("2. Open Netron and load 'attunet.onnx'")
    print("   - Or simply drag and drop 'attunet.onnx' into Netron")
