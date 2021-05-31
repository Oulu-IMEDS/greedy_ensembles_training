"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math

import torch.nn as nn

__all__ = ["VGG16Drop", "VGG16BNDrop", "VGG16", "VGG16BN"]


def make_layers(cfg, batch_norm=False, dropout_rate=0.0):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [
                    nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
                    conv2d,
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
            else:
                layers += [
                    nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
                    conv2d,
                    nn.ReLU(inplace=True),
                ]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGGDrop(nn.Module):
    def __init__(
            self,
            num_classes=10,
            depth=16,
            batch_norm=False,
            dropout_rate=0.0,
            dropout_rate_fc=0.5,
    ):
        super(VGGDrop, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm, dropout_rate)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate_fc) if dropout_rate_fc else nn.Identity(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate_fc) if dropout_rate_fc else nn.Identity(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(VGGDrop):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__(
            num_classes=num_classes,
            depth=16,
            batch_norm=False,
            dropout_rate=0,
            dropout_rate_fc=0.5,
        )


class VGG16BN(VGGDrop):
    def __init__(self, num_classes=10):
        super(VGG16BN, self).__init__(
            num_classes=num_classes,
            depth=16,
            batch_norm=True,
            dropout_rate=0,
            dropout_rate_fc=0.5,
        )


class VGG16Drop(VGGDrop):
    def __init__(self, num_classes=10):
        super(VGG16Drop, self).__init__(
            num_classes=num_classes,
            depth=16,
            batch_norm=False,
            dropout_rate=0.05,
            dropout_rate_fc=0.05,
        )


class VGG16BNDrop(VGGDrop):
    def __init__(self, num_classes=10):
        super(VGG16BNDrop, self).__init__(
            num_classes=num_classes,
            depth=16,
            batch_norm=True,
            dropout_rate=0.05,
            dropout_rate_fc=0.05,
        )
