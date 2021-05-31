from torch import nn

from gde.models.cifar import PreResNet as PreResNetMeta


class PreResNet8(PreResNetMeta):
    def __init__(self, num_classes=10):
        super(PreResNet8, self).__init__(
            num_classes=num_classes, depth=8, dropout_rate=0
        )
        self.avgpool = nn.AvgPool2d(7)