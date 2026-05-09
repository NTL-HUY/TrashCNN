import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4  # output channels = planes * 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # conv1: 1x1, giảm channel
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # conv2: 3x3, xử lý spatial
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # conv3: 1x1, tăng channel lên planes*4
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()

        self.in_channels = 64

        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes=64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None

        # Cần downsample khi stride>1 hoặc channel thay đổi
        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5):
        super(ModifiedResNet50, self).__init__()

        self.in_channels = 64

        # Stem
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(planes=64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ModifiedResNet50(num_classes=7)
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(model)
    print("Output shape:", y.shape)

    # Kiểm tra key có khớp torchvision không
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    classify_keys = set(model.state_dict().keys())
    backbone_keys = set(backbone.body.state_dict().keys())
    shared = classify_keys & backbone_keys
    print(f"\nKeys khớp với FPN backbone: {len(shared)} / {len(backbone_keys)}")