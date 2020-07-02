import torch

from torch.nn import Module, Sequential


class Normalize(Module):
    def __init__(self, mean, std, channels):
        super(Normalize, self).__init__()

        self.mean = mean
        self.std = std
        self.channels = channels

    def forward(self, x):
        x = x - torch.tensor(self.mean, device=x.device).reshape(1, self.channels, 1, 1)
        x = x / self.std
        return x


class ToTensor(Module):
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, x):
        x = x / 255
        return x


class NormalizePreProc(Module):
    def __init__(self, mean, std, channels):
        super(NormalizePreProc, self).__init__()
        self.features = Sequential()
        scaling = ToTensor()
        self.features.add_module("scaling", scaling)
        normalize = Normalize(mean, std, channels)
        self.features.add_module("normalize", normalize)

    def forward(self, x):
        return self.features(x)
