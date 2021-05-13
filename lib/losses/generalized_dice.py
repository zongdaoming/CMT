import torch
import torch.nn as nn
import sys, os
sys.path.append('/home/zongdaoming/cv/multi-organ/multi-organ-ijcai')
from lib.losses.basic import *
from lib.losses.BaseClass import _AbstractDiceLoss



class GeneralizedDiceLoss(_AbstractDiceLoss):

    """ Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    Codes are adapted and modified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
    """

    def __init__(self, classes=4, sigmoid_normalization=True, skip_index_after=None, epsilon=1e-6,):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        assert input.size() == target.size()
        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


# def dice_loss(score, target):
#     target = target.float()
#     smooth = 1e-5
#     intersect = torch.sum(score * target)
#     y_sum = torch.sum(target * target)
#     z_sum = torch.sum(score * score)
#     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#     loss = 1 - loss
#     return loss

# def dice_loss1(score, target):
#     target = target.float()
#     smooth = 1e-5
#     intersect = torch.sum(score * target)
#     y_sum = torch.sum(target)
#     z_sum = torch.sum(score)
#     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#     loss = 1 - loss
#     return loss
