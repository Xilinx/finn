# imports
import numpy as np
import torch
from kmeans_pytorch import kmeans
from torch.nn import Module, Sequential, BatchNorm2d, MSELoss
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantMaxPool2d, QuantSigmoid
from brevitas.inject.defaults import *
from brevitas.core.restrict_val import RestrictValueType
# qYOLO imports
from qYOLO.cfg import *


class QTinyYOLOv2(Module):

    def __init__(self, n_anchors, weight_bit_width=8, act_bit_width=8, quant_tensor=True):
        super(QTinyYOLOv2, self).__init__()
        self.weight_bit_width = int(np.clip(weight_bit_width, 1, 8))
        self.act_bit_width = int(np.clip(act_bit_width, 1, 8))
        self.n_anchors = n_anchors

        self.input = QuantIdentity(
            act_quant=Int8ActPerTensorFloatMinMaxInit,
            min_val=-1.0,
            max_val=1.0 - 2.0 ** (-7),
            signed=True,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            return_quant_tensor=quant_tensor
        )
        self.conv1 = Sequential(
            QuantConv2d(3, 16, 3, 1, (2, 2), bias=False,
                        weight_bit_width=8, return_quant_tensor=quant_tensor),
            BatchNorm2d(16),
            QuantReLU(bit_width=8, return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (1, 1), return_quant_tensor=quant_tensor)
        )
        self.conv2 = Sequential(
            QuantConv2d(16, 32, 3, 1, (2, 1), bias=False,
                        weight_bit_width=self.weight_bit_width, return_quant_tensor=quant_tensor),
            BatchNorm2d(32),
            QuantReLU(bit_width=self.act_bit_width,
                      return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 1), return_quant_tensor=quant_tensor)
        )
        self.conv3 = Sequential(
            QuantConv2d(32, 64, 3, 1, (1, 1), bias=False,
                        weight_bit_width=self.weight_bit_width, return_quant_tensor=quant_tensor),
            BatchNorm2d(64),
            QuantReLU(bit_width=self.act_bit_width,
                      return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 1), return_quant_tensor=quant_tensor)
        )
        self.conv4 = Sequential(
            QuantConv2d(64, 128, 3, 1, (2, 2), bias=False,
                        weight_bit_width=self.weight_bit_width, return_quant_tensor=quant_tensor),
            BatchNorm2d(128),
            QuantReLU(bit_width=self.act_bit_width,
                      return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 0), return_quant_tensor=quant_tensor)
        )
        self.conv5 = Sequential(
            QuantConv2d(128, 256, 3, 1, (1, 2), bias=False,
                        weight_bit_width=self.weight_bit_width, return_quant_tensor=quant_tensor),
            BatchNorm2d(256),
            QuantReLU(bit_width=self.act_bit_width,
                      return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 0), return_quant_tensor=quant_tensor)
        )
        self.conv6 = Sequential(
            QuantConv2d(256, 512, 3, 1, (2, 2), bias=False,
                        weight_bit_width=self.weight_bit_width, return_quant_tensor=quant_tensor),
            BatchNorm2d(512),
            QuantReLU(bit_width=self.act_bit_width,
                      return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 0), return_quant_tensor=quant_tensor)
        )
        self.conv7 = Sequential(
            QuantConv2d(512, 512, 3, 1, (2, 2), bias=False,
                        weight_bit_width=self.weight_bit_width, return_quant_tensor=quant_tensor),
            BatchNorm2d(512),
            QuantReLU(bit_width=self.act_bit_width,
                      return_quant_tensor=quant_tensor)
        )
        self.conv8 = Sequential(
            QuantConv2d(512, 512, 3, 1, (1, 2), bias=False,
                        weight_bit_width=self.weight_bit_width, return_quant_tensor=quant_tensor),
            BatchNorm2d(512),
            QuantReLU(bit_width=self.act_bit_width,
                      return_quant_tensor=quant_tensor)
        )
        self.conv9 = QuantConv2d(512, self.n_anchors*O_SIZE, 1, 1, 0, bias=False, weight_bit_width=8, return_quant_tensor=quant_tensor
                                 )
        self.sig = QuantSigmoid(bit_width=8, return_quant_tensor=quant_tensor
                                )

    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(-1, np.prod(GRID_SIZE), self.n_anchors, O_SIZE)

        return x


def getAnchors(dataset, n_anchors, device):
    datapoints = False
    anchors = np.empty((n_anchors, 2))
    # collect labels data
    for i in range(len(dataset)):
        data = (dataset[i][1][2:]).unsqueeze(0)
        if torch.is_tensor(datapoints):
            datapoints = torch.vstack([datapoints, data])
        else:
            datapoints = data
    # k-means clustering
    kmean_idx, anchors = kmeans(
        X=datapoints, num_clusters=n_anchors, distance='euclidean', device=device)

    return anchors


def YOLOout(output, anchors):
    gx = ((torch.arange(GRID_SIZE[1]).repeat_interleave(
        GRID_SIZE[0]*anchors.size(0))) / GRID_SIZE[1]).view(np.prod(GRID_SIZE), anchors.size(0))
    gy = ((torch.arange(GRID_SIZE[0]).repeat(GRID_SIZE[1])).repeat_interleave(
        anchors.size(0)) / GRID_SIZE[0]).view(np.prod(GRID_SIZE), anchors.size(0))
    output[..., 0] = (torch.sigmoid(output[..., 0]) / GRID_SIZE[0]) + gx
    output[..., 1] = (torch.sigmoid(output[..., 1]) / GRID_SIZE[1]) + gy
    output[..., 2] = torch.exp(output[..., 2]) * anchors[:, 0]
    output[..., 3] = torch.exp(output[..., 3]) * anchors[:, 1]
    output[..., 4] = torch.sigmoid(output[..., 4])
    return output
