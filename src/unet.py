# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import torch
import torch.nn as nn
import torchvision.models as models
from fastonn import SelfONN2d
from torch import Tensor

from networks import DoubleConv, DoubleConvolution, UpSample, DownSample, Concat, SelfDoubleConv, SelfAttention


# class UNetRegressor(nn.Module):
#     def __init__(self, init_features: int = 64) -> None:
#         super().__init__()
#
#         # Encoder (Contracting Path)
#         self.enc1 = DoubleConv(3, init_features)
#         self.enc2 = DoubleConv(init_features, init_features * 2)
#         self.enc3 = DoubleConv(init_features * 2, init_features * 4)
#         self.enc4 = DoubleConv(init_features * 4, init_features * 8)
#         self.enc5 = DoubleConv(init_features * 8, init_features * 16)
#
#         # Decoder (Expanding Path)
#         self.up4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2,
#                                       stride=2)
#         self.dec4 = DoubleConv(init_features * 16, init_features * 8)
#
#         self.up3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
#         self.dec3 = DoubleConv(init_features * 8, init_features * 4)
#
#         self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
#         self.dec2 = DoubleConv(init_features * 4, init_features * 2)
#
#         self.up1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
#         self.dec1 = DoubleConv(init_features * 2, init_features)
#
#         self.final_conv = nn.Conv2d(init_features, 3, kernel_size=1)
#
#         self.pool = nn.MaxPool2d(2)
#
#     def forward(self, x: Tensor) -> Tensor:
#         # Encoder
#         enc1 = self.enc1(x)
#         pool1 = self.pool(enc1)
#
#         enc2 = self.enc2(pool1)
#         pool2 = self.pool(enc2)
#
#         enc3 = self.enc3(pool2)
#         pool3 = self.pool(enc3)
#
#         enc4 = self.enc4(pool3)
#         pool4 = self.pool(enc4)
#
#         # Bridge
#         enc5 = self.enc5(pool4)
#
#         # Decoder
#         up4 = self.up4(enc5)
#         concat4 = torch.cat([enc4, up4], dim=1)
#         dec4 = self.dec4(concat4)
#
#         up3 = self.up3(dec4)
#         concat3 = torch.cat([enc3, up3], dim=1)
#         dec3 = self.dec3(concat3)
#
#         up2 = self.up2(dec3)
#         concat2 = torch.cat([enc2, up2], dim=1)
#         dec2 = self.dec2(concat2)
#
#         up1 = self.up1(dec2)
#         concat1 = torch.cat([enc1, up1], dim=1)
#         dec1 = self.dec1(concat1)
#
#         return nn.functional.tanh(self.final_conv(dec1))


class UNetRegressor(nn.Module):
    """
    ## U-Net Regressor
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()

        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o, "instance" if i == in_channels else "none") for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(512, 1024)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([Concat() for _ in range(4)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            x = self.up_conv[i](x)

        # Final $1 \times 1$ convolution layer
        x = self.final_conv(x)

        # [-1, 1] range
        return torch.tanh(x)


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


# class ResNetUNet(nn.Module):
#     def __init__(self, n_classes: int = 3, pretrained: bool = True) -> None:
#         super().__init__()
#
#         # Load pretrained ResNet34 as encoder
#         resnet = models.resnet34(pretrained=pretrained)
#
#         # Encoder layers
#         self.encoder1 = nn.Sequential(
#             resnet.conv1,
#             resnet.bn1,
#             resnet.relu,
#             resnet.maxpool
#         )  # 64 channels
#         self.encoder2 = resnet.layer1  # 64 channels
#         self.encoder3 = resnet.layer2  # 128 channels
#         self.encoder4 = resnet.layer3  # 256 channels
#         self.encoder5 = resnet.layer4  # 512 channels
#
#         # Decoder layers
#         self.decoder4 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
#             DoubleConv(512, 256)  # 512 due to skip connection
#         )
#
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             DoubleConv(256, 128)  # 256 due to skip connection
#         )
#
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#             DoubleConv(128, 64)  # 128 due to skip connection
#         )
#
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#             DoubleConv(96, 32)  # 96 due to skip connection with first encoder
#         )
#
#         self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
#
#     def forward(self, x: Tensor) -> Tensor:
#         # Encoder path with skip connections
#         enc1 = self.encoder1(x)  # skip connection 1
#         enc2 = self.encoder2(enc1)  # skip connection 2
#         enc3 = self.encoder3(enc2)  # skip connection 3
#         enc4 = self.encoder4(enc3)  # skip connection 4
#         enc5 = self.encoder5(enc4)
#
#         # Decoder path using skip connections
#         dec4 = self.decoder4[0](enc5)
#         dec4 = self.decoder4[1](torch.cat([dec4, enc4], dim=1))
#
#         dec3 = self.decoder3[0](dec4)
#         dec3 = self.decoder3[1](torch.cat([dec3, enc3], dim=1))
#
#         dec2 = self.decoder2[0](dec3)
#         dec2 = self.decoder2[1](torch.cat([dec2, enc2], dim=1))
#
#         dec1 = self.decoder1[0](dec2)
#         dec1 = self.decoder1[1](torch.cat([dec1, enc1], dim=1))
#
#         return torch.tanh(self.final_conv(dec1))


class ResNetUNet(nn.Module):
    def __init__(self, n_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()

        resnet = models.resnet34(pretrained=pretrained)

        # Modified first convolution
        self.firstconv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

        # Encoder layers
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder layers
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            DoubleConv(96, 32)
        )

        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        # Initial conv
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0 = x

        # Encoder
        x = self.pool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4[0](e4)
        d4 = self.decoder4[1](torch.cat([d4, e3], dim=1))

        d3 = self.decoder3[0](d4)
        d3 = self.decoder3[1](torch.cat([d3, e2], dim=1))

        d2 = self.decoder2[0](d3)
        d2 = self.decoder2[1](torch.cat([d2, e1], dim=1))

        d1 = self.decoder1[0](d2)
        d1 = self.decoder1[1](torch.cat([d1, e0], dim=1))

        return torch.tanh(self.final_conv(d1))


class DenseNetUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()

        densenet = models.densenet121(weights="DenseNet121_Weights.DEFAULT")

        # Initial conv
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.first_bn = densenet.features.norm0
        self.first_relu = densenet.features.relu0
        self.first_pool = densenet.features.pool0

        # Encoder blocks
        # self.encoder1 = nn.Sequential(*list(densenet.features.denseblock1.children()))

        self.encoder1 = densenet.features.denseblock1
        self.trans1 = densenet.features.transition1

        # self.encoder2 = nn.Sequential(*list(densenet.features.denseblock2.children()))

        self.encoder2 = densenet.features.denseblock2
        self.trans2 = densenet.features.transition2

        # self.encoder3 = nn.Sequential(*list(densenet.features.denseblock3.children()))

        self.encoder3 = densenet.features.denseblock3
        self.trans3 = densenet.features.transition3

        self.encoder4 = densenet.features.denseblock4
        # self.encoder4 = nn.Sequential(*list(densenet.features.denseblock4.children()))

        # Get feature dimensions
        self.feat_dims = {
            'enc1': 256,  # DenseBlock1 output
            'enc2': 512,  # DenseBlock2 output
            'enc3': 1024,  # DenseBlock3 output
            'enc4': 1024  # DenseBlock4 output
        }

        # Decoder layers
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(self.feat_dims['enc4'], self.feat_dims['enc3'],
                               kernel_size=2, stride=2),
            DoubleConv(self.feat_dims['enc4'], self.feat_dims['enc3'])
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(self.feat_dims['enc3'], self.feat_dims['enc2'],
                               kernel_size=2, stride=2),
            DoubleConv(self.feat_dims['enc3'], self.feat_dims['enc2'])
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(self.feat_dims['enc2'], self.feat_dims['enc1'],
                               kernel_size=2, stride=2),
            DoubleConv(self.feat_dims['enc2'], self.feat_dims['enc1'])
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.feat_dims['enc1'], 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)  # 128 from concat with firstconv output
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        # Initial conv
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        e0 = x  # 64

        # Encoder
        x = self.first_pool(x)

        e1 = self.encoder1(x)  # 256
        x = self.trans1(e1)

        e2 = self.encoder2(x)
        x = self.trans2(e2)

        e3 = self.encoder3(x)
        x = self.trans3(e3)

        e4 = self.encoder4(x)

        # Print all encoder output shapes
        print(f"e0: {e0.shape}")
        print(f"e1: {e1.shape}")
        print(f"e2: {e2.shape}")
        print(f"e3: {e3.shape}")
        print(f"e4: {e4.shape}")

        # Decoder
        d4 = self.decoder4[0](e4)
        d4 = self.decoder4[1](d4 + e3)

        d3 = self.decoder3[0](d4)
        d3 = self.decoder3[1](torch.cat([d3, e2], dim=1))

        d2 = self.decoder2[0](d3)
        d2 = self.decoder2[1](torch.cat([d2, e1], dim=1))

        d1 = self.decoder1[0](d2)
        d1 = self.decoder1[1](torch.cat([d1, e0], dim=1))

        return torch.tanh(self.final_conv(d1))


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


class SelfAttentionUNetRegressor(nn.Module):
    def __init__(self, init_features: int = 64) -> None:
        super().__init__()

        # Encoder (Contracting Path)
        self.enc1 = SelfDoubleConv(3, init_features)
        self.enc2 = SelfDoubleConv(init_features, init_features * 2)
        self.enc3 = SelfDoubleConv(init_features * 2, init_features * 4)
        self.enc4 = SelfDoubleConv(init_features * 4, init_features * 8)
        self.enc5 = SelfDoubleConv(init_features * 8, init_features * 16)

        # Add self-attention at bottleneck (16x16 feature maps)
        self.bottleneck_attention = SelfAttention(init_features * 16)

        # Add self-attention in decoder path (32x32 feature maps)
        self.decoder_attention = SelfAttention(init_features * 8)

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


if __name__ == "__main__":
    model = DenseNetUNet(3, 3)
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
