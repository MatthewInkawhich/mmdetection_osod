# Mink

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def qualityonly_focal_loss(pred, target, beta=2.0):
    r"""This is an adaptation of QFL for only localization quality
    heads only.
    
    Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted quality estimation with shape (N,),
            sigmoid has already been applied [0,1].
        target (torch.Tensor): Target quality label [0,1] with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    pred_sigmoid = pred.sigmoid()
    focal_weight = torch.abs(target - pred_sigmoid) ** beta
    bceloss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    return focal_weight * bceloss



@LOSSES.register_module()
class QualityOnlyFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityOnlyFocalLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        #print("\npred:", pred, pred.min(), pred.max(), pred.shape)
        #print("target:", target, target.min(), target.max(), target.shape)
        #exit()

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * qualityonly_focal_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor)
        #print("loss:", loss)
        #exit()

        return loss

