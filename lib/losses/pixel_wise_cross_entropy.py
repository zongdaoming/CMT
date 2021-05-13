# /*
#  * @Author: anonymous 
#  * @Date: 2021-01-14 11:46:11 
#  * @Last Modified by:   anonymous 
#  * @Last Modified time: 2021-01-14 11:46:11 
#  */
import torch.nn as nn
import torch
import torch.nn as nn
import sys,os
sys.path.append('/home/zongdaoming/cv/multi-organ/multi-organ-ijcai')
from lib.losses.basic import *


class PixelWiseCrossEntropyLoss(nn.Module):
    """
    Standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
    """
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses3D
        result = -weights * target * log_probabilities
        # average the losses3D
        return result.mean()


# def entropy_loss(p,C=2):
#     ## p N*C*W*H*D
#     y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
#     ent = torch.mean(y1)

#     return ent

        





