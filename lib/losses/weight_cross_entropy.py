import torch
import torch.nn as nn
import sys, os
sys.path.append('/home/zongdaoming/cv/multi-organ/multi-organ-ijcai')
import torch.nn.functional as F
# from lib.losses.basic import *
from lib.losses.basic import flatten


class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return torch.nn.functional.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = torch.nn.functional.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = torch.autograd.Variable(nominator / denominator, requires_grad=False)
        return class_weights
 
# def entropy_loss(p,C=2):
#     ## p N*C*W*H*D
#     y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
#     ent = torch.mean(y1)
