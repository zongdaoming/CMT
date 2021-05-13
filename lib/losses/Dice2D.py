import torch
import torch.nn as nn
import sys, os
sys.path.append('/home/zongdaoming/cv/multi-organ/multi-organ-ijcai')

# TODO TEST
class DiceLoss2D(nn.Module):
    def __init__(self, classes, epsilon=1e-5, sigmoid_normalization=True):
        super(DiceLoss2D, self).__init__()
        self.epsilon = epsilon
        self.classes = classes

        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)  # TODO test ?

    def flatten(self, tensor):
        return tensor.view(self.classes, -1)

    def expand_as_one_hot(self, target):
        """
        Converts label image to CxHxW, where each label gets converted to
        its corresponding one-hot vector
        :param target is of shape  (1xHxW)
        :return: 3D output tensor (CxHxW) where C is the classes
        """
        shape = target.size()
        shape = list(shape)
        shape.insert(1, self.classes)
        shape = tuple(shape)
        # expand the input tensor to Nx1xHxW
        src = target.unsqueeze(1)
        return torch.zeros(shape).to(target.device).scatter_(1, src, 1).squeeze(0)

    def compute_per_channel_dice(self, input, target):
        epsilon = 1e-5
        target = self.expand_as_one_hot(target.long())
        assert input.size() == target.size(), "input' and 'target' must have the same shape"+ str(input.size()) + " and " + str(target.size())

        input = self.flatten(input)
        target = self.flatten(target).float()

        # Compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        denominator = (input + target).sum(-1)
        return 2. * intersect / denominator.clamp(min=epsilon)

    def forward(self, input, target):
        input = self.normalization(input)
        per_channel_dice = self.compute_per_channel_dice(input, target)
        DSC = per_channel_dice.clone().cpu().detach().numpy()
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice), DSC



def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice

