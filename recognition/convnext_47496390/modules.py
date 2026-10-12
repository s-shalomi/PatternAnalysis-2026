import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/Bengal1/CNN-Architecture-Guide/blob/main/SimpleCNN.py
# https://share.gemini.google/ylSwoghQk0f9
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()

        # convolutional layers
        # block 1: (B, 1, H, W) -> (B, 32, H/2, W/2) <- 1 channel for gray scale image to 32 feature maps
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # num features is the out_channels of conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 2: (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 3: (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # reduces feature maps:  (B, 128, 8, 8)
            # gives classifier a fixed size input
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), # # (B, 128, 8, 8) -> (B, 8192)
            nn.Linear(128 * 8 * 8, 128), # 128 feature maps from conv, each map contributes 8 values -- 128 outputs
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes) # feature map outputs to class scores: (B, 2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
