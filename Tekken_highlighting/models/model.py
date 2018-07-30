import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
import os

import numpy as np

"""
model : c3d - gru x. 실패..
        cnn(c2d) - gru
        accuracy : 71%
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, 4, 2, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(256, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Conv2d(256, 1, 1, 1, 0)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return out


"""
Neural Networks model : GRU
"""



class GRU(nn.Module):
    def __init__(self, cnn):
        super(GRU, self).__init__()

        self.c2d = cnn
        self.gru1 = nn.LSTM(256, 16, batch_first=True)
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Sequential(nn.Linear(16, 1),
                                nn.Sigmoid())

    def forward(self, input):
        input = input.view(-1, 3, 299, 299)
        h = self.c2d(input)
        h = h.view(1, -1, 256)
        h, _ = self.gru1(h)
        h = nn.ReLU()(h)
        h = h.view(-1, 16)
        h = self.dropout(h)  # add dropout
        h = self.fc(h)

        return h
