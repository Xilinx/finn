# general use libraries
import os
import sys
from tqdm import tqdm, trange
import numpy as np
from skimage import io, transform
# from sklearn.model_selection import KFold

# Brevitas ad PyTorch libraries
import torch
import torch.utils.tensorboard
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Module, Sequential, BatchNorm2d, MSELoss
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantMaxPool2d, QuantSigmoid
from brevitas.inject.defaults import *
from brevitas.core.restrict_val import RestrictValueType

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
O_SIZE = 5
GRID_SIZE = (9, 16)


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
        self.conv9 = QuantConv2d(512, O_SIZE, 1, 1, 0, bias=False, weight_bit_width=8, return_quant_tensor=quant_tensor
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
        x = self.sig(x.view(-1, np.prod(x.shape[-2:]), O_SIZE))

        return x


class YOLO_dataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None, grid_size=GRID_SIZE):
        self.img_dir = img_dir
        self.imgs = sorted(os.listdir(self.img_dir))
        self.lbl_dir = lbl_dir
        self.lbls = sorted(os.listdir(self.lbl_dir))
        self.transform = transform
        self.grid_size = grid_size

    def __len__(self):
        # return len(os.listdir(self.img_dir))
        return 10

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


class YOLOLoss(Module):
    def __init__(self, device, l_coor_obj=1.0, l_coor_noobj=1.0, l_conf_obj=1.0, l_conf_noobj=1.0):
        super().__init__()
        self.l_coor_obj = l_coor_obj
        self.l_coor_noobj = l_coor_noobj
        self.l_conf_obj = l_conf_obj
        self.l_conf_noobj = l_conf_noobj
        self.mse = MSELoss()

    def forward(self, pred, label):
        # mask of obj and noobj
        idx = ((label[:, 0] * GRID_SIZE[0]).floor() * GRID_SIZE[0] +
               (label[:, 1] * GRID_SIZE[1]).floor()).type(torch.int64)
        obj_mask = (torch.zeros_like(pred)).type(torch.bool)
        obj_mask[np.arange(label.shape[0]), idx, :] = True
        noobj_mask = ~obj_mask
        pred_obj = pred[obj_mask].view(pred.shape[0], 5)
        pred_noobj = pred[noobj_mask].view(pred.shape[0], -1, 5)
        pred_obj.to(device)
        pred_noobj.to(device)

        # coordination loss
        # TODO calculate real average
        average_vec = torch.tensor([0.5, 0.5, 0.01, 0.01])
        coor_l_obj = self.mse(pred_obj[:, :4], label)
        coor_l_noobj = self.mse(
            pred_noobj[:, :, :4], average_vec*torch.ones_like(pred_noobj[:, :, :4], device=device))

        # confidence loss
        conf_l_obj = self.mse(pred_obj[:, 4], IoU_calc(
            pred_obj.unsqueeze(1), label))
        conf_l_noobj = self.mse(pred_noobj[:, :, 4], torch.zeros_like(
            pred_noobj[:, :, 4], device=device))

        return coor_l_obj*self.l_coor_obj + coor_l_noobj*self.l_coor_noobj + conf_l_obj*self.l_conf_obj + conf_l_noobj*self.l_conf_noobj


def IoU_calc(pred_, label):
    # localizing most probable bounding box
    _, bb_idx = pred_[:, :, 4].max(1)
    pred = pred_[np.arange(pred_.shape[0]), bb_idx, :4]
    # xmin, ymin, xmax, ymax
    label_bb = torch.stack([torch.max(label[:, 0]-(label[:, 2]/2), torch.tensor(0.0)),
                            torch.max(label[:, 1]-(label[:, 3]/2),
                                      torch.tensor(0.0)),
                            torch.min(label[:, 0]+(label[:, 2]/2),
                                      torch.tensor(1.0)),
                            torch.min(label[:, 1]+(label[:, 3]/2), torch.tensor(1.0))], 1)
    pred_bb = torch.stack([torch.max(pred[:, 0]-(pred[:, 2]/2), torch.tensor(0.0)),
                           torch.max(pred[:, 1]-(pred[:, 3]/2),
                                     torch.tensor(0.0)),
                           torch.min(pred[:, 0]+(pred[:, 2]/2),
                                     torch.tensor(1.0)),
                           torch.min(pred[:, 1]+(pred[:, 3]/2), torch.tensor(1.0))], 1)
    inter_bb = torch.stack([torch.max(label_bb[:, 0], pred_bb[:, 0]),
                            torch.max(label_bb[:, 1], pred_bb[:, 1]),
                            torch.min(label_bb[:, 2], pred_bb[:, 2]),
                            torch.min(label_bb[:, 3], pred_bb[:, 3])], 1)
    # calculate IoU
    label_area = label_bb[:, 2]*label_bb[:, 3]
    pred_area = pred_bb[:, 2]*pred_bb[:, 3]
    inter_area = torch.max(inter_bb[:, 2]-inter_bb[:, 0], torch.tensor(0.0)) * \
        torch.max(inter_bb[:, 3]-inter_bb[:, 1], torch.tensor(0.0))
    # return IoU
    return inter_area / (label_area + pred_area - inter_area)

# ------------------------------------------------------------------------------------------------------------------------------------------------ #


if __name__ == "__main__":
    # asses input args
    img_dir = sys.argv[1]
    lbl_dir = sys.argv[2]
    weight_bit_width = int(sys.argv[3])
    act_bit_width = int(sys.argv[4])
    n_epochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Trainig on: {device}")

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
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=True, num_workers=4)

    # network setup
    net = QTinyYOLOv2(weight_bit_width, act_bit_width)
    net.to(device)
    loss_func = YOLOLoss(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # train network
    for epoch in trange(n_epochs, desc="epoch", unit="epoch"):
        # train + train loss
        train_loss = 0.0
        test_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0), total=int(np.ceil(train_len/batch_size)), desc="train loss", unit="batch"):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_func(outputs.value.float(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # test loss
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader, 0), total=int(np.ceil(test_len/batch_size)), desc="test loss", unit="batch"):
                test_images, test_labels = data[0].to(
                    device), data[1].to(device)
                test_outputs = net(test_images)
                t_loss = loss_func(
                    test_outputs.value.float(), test_labels.float())
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
            for data in tqdm(train_loader, total=int(np.ceil(train_len/batch_size)), desc="train accuracy", unit="batch"):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                iou = IoU_calc(outputs.value.cpu(), labels.cpu())
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
            for data in tqdm(test_loader, total=int(np.ceil(test_len/batch_size)), desc="test accuracy", unit="batch"):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                iou = IoU_calc(outputs.value.cpu(), labels.cpu())
                test_total += labels.size(0)
                test_miou += iou.sum()
                test_AP50 += (iou >= .5).sum()
                test_AP75 += (iou >= .75).sum()
            # log accuracy statistics
            logger.add_scalar('meanIoU/test', test_miou/test_total, epoch)
            logger.add_scalar('meanAP50/test', test_AP50/test_total, epoch)
            logger.add_scalar('meanAP75/test', test_AP75/test_total, epoch)

    # save network
    path = f"./trained_net_W{weight_bit_width}A{act_bit_width}_e{n_epochs}.pth"
    torch.save(net.state_dict(), path)
