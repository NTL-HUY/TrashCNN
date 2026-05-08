# modified_resnet50.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block used in ResNet50 architecture
    """

    def __init__(self, in_channels, filters, reduce=False, stride=2):
        """
        Arguments:
        in_channels -- number of input channels
        filters     -- list of filters [F1, F2, F3]
        reduce      -- whether to reduce spatial size
        stride      -- stride for downsampling
        """

        super(ResidualBlock, self).__init__()

        F1, F2, F3 = filters

        self.reduce = reduce

        # Shortcut branch
        self.shortcut = nn.Sequential()

        if reduce:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    F3,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(F3)
            )

            # Main branch first conv
            self.conv1 = nn.Conv2d(
                in_channels,
                F1,
                kernel_size=1,
                stride=stride,
                bias=False
            )

        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                F1,
                kernel_size=1,
                stride=1,
                bias=False
            )

        self.bn1 = nn.BatchNorm2d(F1)

        # Second conv
        self.conv2 = nn.Conv2d(
            F1,
            F2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)

        # Third conv
        self.conv3 = nn.Conv2d(
            F2,
            F3,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        shortcut = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut path
        if self.reduce:
            shortcut = self.shortcut(shortcut)

        # Add
        out += shortcut
        out = self.relu(out)

        return out


class ResNet50(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Stage 2
        self.layer2 = nn.Sequential(
            ResidualBlock(64, [64, 64, 256], reduce=True, stride=1),
            ResidualBlock(256, [64, 64, 256]),
            ResidualBlock(256, [64, 64, 256]),
        )

        # Stage 3
        self.layer3 = nn.Sequential(
            ResidualBlock(256, [128, 128, 512], reduce=True, stride=2),
            ResidualBlock(512, [128, 128, 512]),
            ResidualBlock(512, [128, 128, 512]),
            ResidualBlock(512, [128, 128, 512]),
        )

        # Stage 4
        self.layer4 = nn.Sequential(
            ResidualBlock(512, [256, 256, 1024], reduce=True, stride=2),
            ResidualBlock(1024, [256, 256, 1024]),
            ResidualBlock(1024, [256, 256, 1024]),
            ResidualBlock(1024, [256, 256, 1024]),
            ResidualBlock(1024, [256, 256, 1024]),
            ResidualBlock(1024, [256, 256, 1024]),
        )

        # Stage 5
        self.layer5 = nn.Sequential(
            ResidualBlock(1024, [512, 512, 2048], reduce=True, stride=2),
            ResidualBlock(2048, [512, 512, 2048]),
            ResidualBlock(2048, [512, 512, 2048]),
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Pooling
        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        return x


class ModifiedResNet50(nn.Module):

    def __init__(self, num_classes=1000, dropout_rate=0.5):
        super(ModifiedResNet50, self).__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Stage 2
        self.layer2 = nn.Sequential(
            ResidualBlock(64, [64, 64, 256], reduce=True, stride=1),
            ResidualBlock(256, [64, 64, 256]),
            ResidualBlock(256, [64, 64, 256]),
        )

        # Stage 3
        self.layer3 = nn.Sequential(
            ResidualBlock(256, [128, 128, 512], reduce=True, stride=2),
            ResidualBlock(512, [128, 128, 512]),
            ResidualBlock(512, [128, 128, 512]),
            ResidualBlock(512, [128, 128, 512]),
        )

        # Stage 4
        self.layer4 = nn.Sequential(
            ResidualBlock(512, [256, 256, 1024], reduce=True, stride=2),
            ResidualBlock(1024, [256, 256, 1024]),
            ResidualBlock(1024, [256, 256, 1024]),
            ResidualBlock(1024, [256, 256, 1024]),
            ResidualBlock(1024, [256, 256, 1024]),
            ResidualBlock(1024, [256, 256, 1024]),
        )

        # Stage 5
        self.layer5 = nn.Sequential(
            ResidualBlock(1024, [512, 512, 2048], reduce=True, stride=2),
            ResidualBlock(2048, [512, 512, 2048]),
            ResidualBlock(2048, [512, 512, 2048]),
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully Connected
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Pooling
        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Dropout
        x = self.dropout(x)

        # Classification
        x = self.fc(x)

        return x


# Testing
if __name__ == "__main__":

    model = ModifiedResNet50(num_classes=10)

    x = torch.randn(4, 3, 224, 224)

    y = model(x)

    print(model)
    print("Output shape:", y.shape)