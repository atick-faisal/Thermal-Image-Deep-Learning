import torch.nn as nn

from src.networks import SingleConv


class CoreNetDiscriminator(nn.Module):
    def __init__(self, init_features=16):
        super(CoreNetDiscriminator, self).__init__()

        # Initial feature extraction
        # self.features = nn.Sequential(
        #     # First conv block
        #     nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 128x128
        #
        #     # Second conv block
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 64x64
        #
        #     # Third conv block
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 32x32
        #
        #     # Fourth conv block
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 16x16
        # )

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
            nn.Linear(64, 1),
            nn.ReLU()  # Final ReLU for positive PSNR values
        )

    def forward(self, x):
        # Input shape: (batch_size, channels, 256, 256)
        # x = self.features(x)  # Shape: (batch_size, 512, 16, 16)
        # x = self.global_pool(x)  # Shape: (batch_size, 512, 1, 1)
        # x = x.view(x.size(0), -1)  # Shape: (batch_size, 512)
        # x = self.fc(x)  # Shape: (batch_size, 1)
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, 1, 1)
