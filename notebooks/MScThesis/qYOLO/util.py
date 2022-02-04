# imports
import numpy as np
import torch
from torch.nn import Module, MSELoss
# qYOLO imports
from qYOLO.module import *


class YOLOLoss(Module):
    def __init__(self, anchors, device, l_coor_obj=1.0, l_coor_noobj=1.0, l_conf_obj=1.0, l_conf_noobj=1.0):
        super().__init__()
        self.anchors = anchors
        self.device = device
        self.l_coor_obj = l_coor_obj
        self.l_coor_noobj = l_coor_noobj
        self.l_conf_obj = l_conf_obj
        self.l_conf_noobj = l_conf_noobj
        self.mse = MSELoss()

    def forward(self, pred, label):
        # locate bounding box location and
        idx_x = (label[:, 0] * GRID_SIZE[1]).floor()
        idx_y = (label[:, 1] * GRID_SIZE[0]).floor()
        idx = (idx_x * GRID_SIZE[1] + idx_y).type(torch.int64)
        # convert predictions to label style
        pred = YOLOout(pred, self.anchors)
        # find closest anchor
        anchor_mask = (((label[:, 2:4].unsqueeze(1).repeat(
            (1, self.anchors.size(0), 1))-self.anchors.unsqueeze(0).repeat((label.size(0), 1, 1))) ** 2.0).sum(2)).argmax(1)
        # create anchors grid
        anchors_grid = torch.ones_like(pred[..., :4])
        anchors_grid[..., 0] = ((torch.arange(GRID_SIZE[1]).repeat_interleave(
            GRID_SIZE[0]*self.anchors.size(0)) + 0.5) / GRID_SIZE[1]).view(np.prod(GRID_SIZE), self.anchors.size(0))
        anchors_grid[..., 1] = (((torch.arange(GRID_SIZE[0]).repeat(GRID_SIZE[1])).repeat_interleave(
            self.anchors.size(0)) + 0.5) / GRID_SIZE[0]).view(np.prod(GRID_SIZE), self.anchors.size(0))
        anchors_grid[..., 2:4] = self.anchors
        # mask of obj and noobj
        obj_mask = (torch.zeros_like(pred)).type(torch.bool)
        obj_mask[np.arange(pred.shape[0]), idx, anchor_mask, :] = True
        noobj_mask = ~obj_mask
        # prepare loss calculation parts
        pred_obj = pred[obj_mask].view(pred.shape[0], 5)
        pred_noobj = pred[noobj_mask].view(pred.shape[0], -1, 5)
        anchors_grid = anchors_grid[noobj_mask[..., :4]].view(
            pred.shape[0], -1, 4)
        pred_obj.to(self.device)
        pred_noobj.to(self.device)
        anchors_grid.to(self.device)

        # coordination loss
        coor_l_obj = self.mse(pred_obj[:, :4], label)
        coor_l_noobj = self.mse(pred_noobj[..., :4], anchors_grid)

        # confidence loss
        conf_l_obj = self.mse(pred_obj[:, 4], IoU_calc(
            pred_obj.unsqueeze(1), label))
        conf_l_noobj = self.mse(pred_noobj[..., 4], torch.zeros_like(
            pred_noobj[..., 4], device=self.device))
        return coor_l_obj*self.l_coor_obj + coor_l_noobj*self.l_coor_noobj + conf_l_obj*self.l_conf_obj + conf_l_noobj*self.l_conf_noobj


def IoU_calc(pred_, label):
    # localizing most probable bounding box
    bb_idx = torch.argmax(pred_[..., 4].view(pred_.size(0), -1), 1)
    pred = (pred_.view(pred_.size(0), -1, 5)
            )[torch.arange(pred_.size(0)), bb_idx]
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
