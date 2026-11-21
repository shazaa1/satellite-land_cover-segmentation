import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights, resnet50




class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load ImageNet pretrained weights if requested
        weights = ResNet50_Weights.DEFAULT if pretrained else None

        # Use dilation in the last ResNet stage to get output stride 16
        resnet = resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, False, True]
        )

        # Keep the layers needed for feature extraction
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):

        # Extract early spatial features (low-level features)
        x = self.stem(x)
        low_level_features = self.layer1(x)

        # Extract deeper semantic features
        x = self.layer2(low_level_features)
        x = self.layer3(x)
        high_level_features = self.layer4(x)

        return low_level_features, high_level_features

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolution branches with different dilation rates
        self.branch2 = self._make_atrous_branch(in_channels, out_channels, 6)
        self.branch3 = self._make_atrous_branch(in_channels, out_channels, 12)
        self.branch4 = self._make_atrous_branch(in_channels, out_channels, 18)

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Combine the five branches
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def _make_atrous_branch(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # Get outputs from all five ASPP branches
        size = x.shape[2:]

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Upsample global context to match feature-map size
        global_features = self.global_pool(x)
        global_features = F.interpolate(
            global_features,
            size=size,
            mode="bilinear",
            align_corners=False
        )

        # Concatenate all branches
        features = torch.cat(
            [branch1, branch2, branch3, branch4, global_features],
            dim=1
        )

        return self.project(features)

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        # ResNet50 feature extractor
        self.backbone = ResNet50Backbone(pretrained=pretrained)

        # ResNet50 layer4 produces 2048 channels
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Reduce low-level features from 256 to 48 channels
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder combines ASPP and low-level features
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Predict one score for each of the five classes
        self.classifier = nn.Conv2d(
            256,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        input_size = x.shape[2:]

        # Extract low-level and high-level features
        low_features, high_features = self.backbone(x)

        # Apply ASPP to high-level features
        high_features = self.aspp(high_features)

        # Resize ASPP features to match low-level feature size
        high_features = F.interpolate(
            high_features,
            size=low_features.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # Project low-level features
        low_features = self.low_level_projection(low_features)

        # Combine both feature types
        features = torch.cat(
            [high_features, low_features],
            dim=1
        )

        # Decode combined features
        features = self.decoder(features)

        # Generate class predictions
        output = self.classifier(features)

        # Resize predictions to original image resolution
        output = F.interpolate(
            output,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        return output