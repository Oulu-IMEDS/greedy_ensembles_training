import torch
from torch import nn


class ToyModel(torch.nn.Module):
    def __init__(self, n_inp, n_hidden, num_classes):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(n_inp, n_hidden),
                                 nn.ReLU(True))

        self.fc2 = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                 nn.ReLU(True))

        self.out = nn.Linear(n_hidden, num_classes)

    def forward(self, x):
        o = self.fc1(x)
        o = self.fc2(o)
        return self.out(o)
