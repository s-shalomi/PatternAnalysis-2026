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


# https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
# https://share.gemini.google/n9s7qtpk7eQD


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerNorm(nn.Module):
    """Channels-first LayerNorm supporting (N, C, H, W) data format."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, H, W) -> permute to (N, H, W, C) for standard PyTorch LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    Structure:
        7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv (expand 4x) -> GELU -> 1x1 Conv (project back)
        Residual connection with LayerScale and DropPath.
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        # 1. 7x7 Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # 2. Normalization
        self.norm = LayerNorm(dim, eps=1e-6)

        # 3. Pointwise inverted bottleneck (dim -> 4*dim -> dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # 4. LayerScale
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        # 5. Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)

        # Linear layers run faster over channels-last format
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem: 4x4 non-overlapping patchify (stride 4) + LayerNorm
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # 3 Downsampling layers between stages (stride 2 conv + LayerNorm)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 Feature stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Classification Head: Global Avg Pool -> LayerNorm -> Linear
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # Global average pool (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)

def convnext_base(num_classes: int = 1000, in_chans: int = 3, **kwargs):
    return ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=kwargs.get("drop_path_rate", 0.5),
    )

def convnext_tiny(num_classes: int = 1000, in_chans: int = 3, **kwargs):
    return ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=kwargs.get("drop_path_rate", 0.2),
    )
