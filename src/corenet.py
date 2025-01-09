# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from networks import SingleConv


class CoreNetDiscriminator(nn.Module):
    def __init__(self, init_features: int = 16, corenet_out_channels: int = 1):
        super(CoreNetDiscriminator, self).__init__()

        self.enc1 = SingleConv(3, init_features)
        self.enc2 = SingleConv(init_features, init_features * 2)
        self.enc3 = SingleConv(init_features * 2, init_features * 4)
        self.enc4 = SingleConv(init_features * 4, init_features * 8)

        self.pool = nn.MaxPool2d(2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(init_features * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, corenet_out_channels),
            nn.ReLU()  # Final ReLU for positive PSNR values
        )

    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.unsqueeze(-1)


class DenseNetNetDiscriminator(nn.Module):
    def __init__(self, corenet_out_channels: int = 3):
        super(DenseNetNetDiscriminator, self).__init__()

        # Load pretrained DenseNet
        self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Get the number of features from the last layer
        num_features = self.densenet.classifier.in_features

        # Replace the classifier
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, corenet_out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.densenet(x).unsqueeze(-1)


class ResNetNetDiscriminator(nn.Module):
    def __init__(self, corenet_out_channels: int = 3):
        super(ResNetNetDiscriminator, self).__init__()

        # Load pretrained DenseNet
        self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Get the number of features from the last layer
        num_features = self.densenet.classifier.in_features

        # Replace the classifier
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, corenet_out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.densenet(x).unsqueeze(-1)
