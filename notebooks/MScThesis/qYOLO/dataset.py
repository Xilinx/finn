# imports
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
# qYOLO imports
from qYOLO.cfg import *


class YOLO_dataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None, grid_size=GRID_SIZE):
        self.img_dir = img_dir
        self.imgs = sorted(os.listdir(self.img_dir))
        self.lbl_dir = lbl_dir
        self.lbls = sorted(os.listdir(self.lbl_dir))
        self.transform = transform
        self.grid_size = grid_size

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(os.path.join(self.img_dir, self.imgs[idx]))

        with open(os.path.join(self.lbl_dir, self.lbls[idx])) as f:
            dataline = f.readlines()[1]
            lbl_data = [data.strip() for data in dataline.split('\t')]
            lbl = np.array(lbl_data).astype(float)
            f.close()

        sample = [img, lbl]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        img, lbl = sample

        img = img.transpose((2, 0, 1))
        return [torch.from_numpy(img), torch.from_numpy(lbl)]


class Normalize(object):
    def __call__(self, sample, mean=0.5, std=0.5):
        img, lbl = sample

        img = ((img / 255) - mean) / std

        return [img, lbl]
