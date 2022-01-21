# general use libraries
import os
import sys
import numpy as np
from skimage import io, transform
# from sklearn.model_selection import KFold

# Brevitas ad PyTorch libraries
import torch
import torch.utils.tensorboard
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Module, Sequential, BatchNorm2d
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantMaxPool2d
from brevitas.inject.defaults import *
from brevitas.core.restrict_val import RestrictValueType

# ------------------------------------------------------------------------------------------------------------------------------------------------ #


class QTinyYOLOv2(Module):

    def __init__(self, weight_bit_width=8, act_bit_width=8, quant_tensor=True):
        super(QTinyYOLOv2, self).__init__()
        self.weight_bit_width = int(np.clip(weight_bit_width, 1, 8))
        self.act_bit_width = int(np.clip(act_bit_width, 1, 8))

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
        self.conv9 = QuantConv2d(512, 6, 1, 1, 0, bias=False, weight_bit_width=8, return_quant_tensor=quant_tensor
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
        return x


class YOLO_dataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None, grid_size=(9, 16)):
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
            b_x = float(lbl_data[1])
            b_y = float(lbl_data[2])
            b_w = float(lbl_data[3])
            b_h = float(lbl_data[4])
            lbl_idx = (int(np.floor(self.grid_size[0] * b_y)),
                       int(np.floor(self.grid_size[1] * b_x)))
            lbl = np.zeros((6, self.grid_size[0], self.grid_size[1]))
            lbl[:, lbl_idx[0], lbl_idx[1]] = [1.0, b_x, b_y, b_w, b_h, 1.0]
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


class YOLOLoss(torch.nn.modules.loss._Loss):

    def __init__(self, l_coord=5.0, l_noobj=0.5, l_obj=5.0, l_cls=1.0):
        super().__init__()

        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_obj = l_obj
        self.l_cls = l_cls

    def forward(self, y_pred, y_true):
        print(y_pred.shape, y_true.shape)
        bb_loc = y_true[:, 0, :, :] == 1.0
        bb_nloc = y_true[:, 0, :, :] == 0.0
        print(bb_loc, bb_loc.shape)
        y_pred_bb = y_pred.value[bb_loc]
        y_true_bb = y_true.value[bb_loc]
        y_pred_nbb = y_pred.value[bb_nloc]
        y_true_nbb = y_true.value[bb_nloc]

        classification_loss = self.l_cls * ((y_true_bb[:, 5].squeeze() -
                                             y_pred_bb[:, 5].squeeze()) ** 2.0).sum()

        localization_loss = self.l_coord * ((((y_true_bb[:, 1].squeeze() - y_pred_bb[:, 1].squeeze()) ** 2.0) + ((y_true_bb[:, 2].squeeze() - y_pred_bb[:, 2].squeeze()) ** 2.0)).sum(
        ) + (((y_true_bb[:, 3].sqrt().squeeze() - y_pred_bb[:, 3].sqrt().squeeze()) ** 2.0) + ((y_true_bb[:, 4].sqrt().squeeze() - y_pred_bb[:, 4].sqrt().squeeze()) ** 2.0)).sum())

        confidence_loss = self.l_obj * (((y_true_bb[:, 0].squeeze() - y_pred_bb[:, 0].squeeze()) ** 2.0).sum()) + self.l_noobj * (
            ((y_true_nbb[:, 0].squeeze() - y_pred_nbb[:, 0].squeeze()) ** 2.0).sum())

        return (classification_loss + localization_loss + confidence_loss) / y_pred.size(0)


def IoU_calc(pred, label):
    # localizing most probable bounding box
    label_idx = label[:, :, :, 0].view(
        label[:, :, :, 0].size(0), -1).max(dim=-1).indices
    pred_idx = pred[:, :, :, 0].view(
        pred[:, :, :, 0].size(0), -1).max(dim=-1).indices
    label_y, label_x = torch.tensor(np.unravel_index(
        label_idx, label[:, :, :, 0].shape))[1:3]
    pred_y, pred_x = torch.tensor(np.unravel_index(
        pred_idx, pred[:, :, :, 0].shape))[1:3]
    # x, y, w, h
    label_bb = torch.stack([bb[label_y[i], label_x[i], 1:5]
                            for i, bb in enumerate(label)], 1)
    pred_bb = torch.stack([bb[pred_y[i], pred_x[i], 1:5]
                           for i, bb in enumerate(pred)], 1)
    # xmin, ymin, xmax, ymax
    true_bbm = torch.stack([torch.max(label_bb[0]-(label_bb[2]/2), torch.tensor(0.0)),
                           torch.max(
                               label_bb[1]-(label_bb[3]/2), torch.tensor(0.0)),
                           torch.min(
                               label_bb[0]+(label_bb[2]/2), torch.tensor(1.0)),
                           torch.min(label_bb[1]+(label_bb[3]/2), torch.tensor(1.0))])
    pred_bbm = torch.stack([torch.max(pred_bb[0]-(pred_bb[2]/2), torch.tensor(0.0)),
                           torch.max(pred_bb[1]-(pred_bb[3]/2),
                                     torch.tensor(0.0)),
                           torch.min(pred_bb[0]+(pred_bb[2]/2),
                                     torch.tensor(1.0)),
                           torch.min(pred_bb[1]+(pred_bb[3]/2), torch.tensor(1.0))])
    inter_bbm = torch.stack([torch.max(true_bbm[0], pred_bbm[0]),
                            torch.max(true_bbm[1], pred_bbm[1]),
                            torch.min(true_bbm[2], pred_bbm[2]),
                            torch.min(true_bbm[3], pred_bbm[3])])
    # calculate IoU
    true_area = true_bbm[2]*true_bbm[3]
    pred_area = pred_bbm[2]*pred_bbm[3]
    inter_area = torch.max(inter_bbm[2]-inter_bbm[0], torch.tensor(
        0.0)) * torch.max(inter_bbm[3]-inter_bbm[1], torch.tensor(0.0))
    # return IoU
    return inter_area / (true_area + pred_area - inter_area)

# ------------------------------------------------------------------------------------------------------------------------------------------------ #


if __name__ == "__main__":
    # asses input args
    img_dir = sys.argv[1]
    lbl_dir = sys.argv[2]
    weight_bit_width = int(sys.argv[3])
    act_bit_width = int(sys.argv[4])
    n_epochs = int(sys.argv[5])

    # logger
    logger = torch.utils.tensorboard.SummaryWriter()

    # dataset
    transformers = transforms.Compose(
        [ToTensor(), Normalize()])
    dataset = YOLO_dataset(img_dir, lbl_dir, transformers)
    data_len = len(dataset)
    train_len = int(data_len*0.8)
    test_len = data_len-train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=1,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=True, num_workers=2)

    # network setup
    net = QTinyYOLOv2(weight_bit_width, act_bit_width)
    loss_func = YOLOLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # train network
    for epoch in range(n_epochs):
        # train + train loss
        train_loss = 0.0
        test_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # test loss
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                test_images, test_labels = data
                test_outputs = net(test_images)
                t_loss = loss_func(test_outputs, test_labels)
                test_loss += t_loss.item()
        # log loss statistics
        logger.add_scalar('Loss/train', train_loss/train_len, epoch)
        logger.add_scalar('Loss/test', test_loss/test_len, epoch)

        # train accuracy
        with torch.no_grad():
            train_miou = 0.0
            train_AP50 = 0.0
            train_AP75 = 0.0
            train_total = 0
            for data in train_loader:
                images, labels = data
                outputs = net(images)
                iou = IoU_calc(outputs, labels)
                train_total += labels.size(0)
                train_miou += iou.sum()
                train_AP50 += (iou >= .5).sum()
                train_AP75 += (iou >= .75).sum()
            # log accuracy statistics
            logger.add_scalar('meanIoU/train', train_miou/train_total, epoch)
            logger.add_scalar('meanAP50/train', train_AP50/train_total, epoch)
            logger.add_scalar('meanAP75/train', train_AP75/train_total, epoch)
        # test accuracy
        with torch.no_grad():
            test_miou = 0.0
            test_AP50 = 0.0
            test_AP75 = 0.0
            test_total = 0
            for data in test_loader:
                images, labels = data
                outputs = net(images)
                iou = IoU_calc(outputs, labels)
                test_total += labels.size(0)
                test_miou += iou.sum()
                test_AP50 += (iou >= .5).sum()
                test_AP75 += (iou >= .75).sum()
            # log accuracy statistics
            logger.add_scalar('meanIoU/train', test_miou/test_total, epoch)
            logger.add_scalar('meanAP50/train', test_AP50/test_total, epoch)
            logger.add_scalar('meanAP75/train', test_AP75/test_total, epoch)

    # save network
    path = f"./trained_net_W{weight_bit_width}A{act_bit_width}_e{n_epochs}.pth"
    torch.save(net.state_dict(), path)
