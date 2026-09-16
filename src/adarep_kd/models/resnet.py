import torch
import torch.nn as nn


class CIFARBasicBlock(nn.Module):
    """
    Basic residual block for CIFAR-style ResNet.

    Each block contains:
        Conv 3x3
        BatchNorm
        ReLU
        Conv 3x3
        BatchNorm

    followed by a residual connection and ReLU.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class CIFARResNet(nn.Module):
    """
    CIFAR-style ResNet.

    Depth:
        depth = 6 * num_blocks + 2

    For this project:
        ResNet-32 -> num_blocks = 5
        ResNet-8  -> num_blocks = 1

    The model exposes outputs from three residual stages.
    """

    def __init__(
        self,
        block,
        num_blocks,
        num_classes=100,
    ):
        super().__init__()

        if len(num_blocks) != 3:
            raise ValueError("num_blocks must contain exactly three values.")

        self.in_channels = 16

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(
            block,
            16,
            num_blocks[0],
            stride=1,
        )

        self.stage2 = self._make_stage(
            block,
            32,
            num_blocks[1],
            stride=2,
        )

        self.stage3 = self._make_stage(
            block,
            64,
            num_blocks[2],
            stride=2,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            64 * block.expansion,
            num_classes,
        )

    def _make_stage(
        self,
        block,
        out_channels,
        num_blocks,
        stride,
    ):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for current_stride in strides:
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    current_stride,
                )
            )

            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)

        pooled = self.avg_pool(stage3)
        pooled = torch.flatten(pooled, 1)

        logits = self.classifier(pooled)

        features = {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }

        return logits, features


def resnet32(num_classes=100):
    """
    Construct a CIFAR ResNet-32.

    5 residual blocks per stage:
        6 * 5 + 2 = 32
    """

    return CIFARResNet(
        block=CIFARBasicBlock,
        num_blocks=[5, 5, 5],
        num_classes=num_classes,
    )


def resnet8(num_classes=100):
    """
    Construct a CIFAR ResNet-8.

    1 residual block per stage:
        6 * 1 + 2 = 8
    """

    return CIFARResNet(
        block=CIFARBasicBlock,
        num_blocks=[1, 1, 1],
        num_classes=num_classes,
    )