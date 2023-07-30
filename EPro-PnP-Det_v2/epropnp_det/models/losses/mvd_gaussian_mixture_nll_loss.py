# Modified from https://github.com/tjiiv-cprg/EPro-PnP

"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import math
import torch
import torch.nn as nn

from mmdet.core import reduce_mean
from mmdet.models import LOSSES, weighted_loss

from ...ops import logsumexp_across_rois


@weighted_loss
def mvd_gaussian_mixture_nll_loss(
        pred, target, logstd=None, logmixweight=None, rois=None,
        adaptive_weight=True, momentum=0.1, mean_inv_std=None,
        dim=1, delta=-1, eps=1e-4, training=True):
    """
    Args:
        pred (torch.Tensor): shape (n, num_mix, h, w, *)
        target (torch.Tensor): shape (n, 1, h, w, *)
        logstd (torch.Tensor): shape (n, num_mix, h, w, *)
        logmixweight (torch.Tensor): shape (n, num_mix, h, w)
        rois (torch.Tensor | None): shape (n, 5)
        adaptive_weight (bool)
        momentum (float)
        mean_inv_std (torch.Tensor)
        dim (int): dimension of mixture
        eps (float)
        training (bool)
    """
    if isinstance(target, int):
        if target == 0:
            diff = torch.abs(pred)
        elif target == -1:
            diff = pred
        else:
            raise ValueError
    else:
        diff = torch.abs(pred - target)
    inverse_std = torch.exp(-logstd).clamp(max=1/eps)
    diff_weighted = diff * inverse_std  # (n, num_mix, h, w, *)
    diff_weighted_sq = diff_weighted.square().sum(dim=-1)  # (n, num_mix, h, w)
    if delta < 0:
        loss_comp = -0.5 * diff_weighted_sq + logmixweight - logstd.sum(dim=-1)  # (n, num_mix, h, w)
    else:
        assert delta > eps
        loss_comp = -torch.where(
            diff_weighted_sq <= delta * delta,
            0.5 * diff_weighted_sq,
            delta * diff_weighted.norm(dim=-1) - 0.5 * delta * delta
        ) + logmixweight - logstd.sum(dim=-1)

    if rois is None:
        loss = -torch.logsumexp(loss_comp, dim=dim)  # (n, h, w)
    else:
        loss = -logsumexp_across_rois(
            torch.logsumexp(loss_comp, dim=dim, keepdim=True),
            rois).squeeze(1)  # (n, h, w)

    if adaptive_weight:
        if training:
            inverse_std_ = inverse_std.detach()  # (n, num_mix, h, w, *)
            mixweight = logmixweight.detach().exp().unsqueeze(-1)  # (n, num_mix, h, w, 1)
            batch_mean_inv_std = reduce_mean((inverse_std_ * mixweight).sum()) / reduce_mean(
                mixweight.sum() * inverse_std_.size(-1)).clamp(min=eps)
            mean_inv_std *= 1 - momentum
            mean_inv_std += momentum * batch_mean_inv_std
        loss = loss / mean_inv_std.clamp(min=eps)
    return loss


@LOSSES.register_module()
class MVDGaussianMixtureNLLLoss(nn.Module):

    def __init__(self, dim=1, reduction='mean', loss_weight=1.0,
                 adaptive_weight=True, sigma=None, freeze_sigma=True,
                 momentum=0.1, delta=-1, eps=1e-4):
        super(MVDGaussianMixtureNLLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.adaptive_weight = adaptive_weight
        if sigma is None:
            self.log_sigma = None
        else:
            self.log_sigma = math.log(sigma) if freeze_sigma \
                else nn.Parameter(torch.Tensor([math.log(sigma)]))
        self.momentum = momentum
        if self.adaptive_weight:
            self.register_buffer('mean_inv_std', torch.tensor(1, dtype=torch.float))
        else:
            self.mean_inv_std = None
        self.dim = dim
        self.delta = delta
        self.eps = eps

    def forward(self,
                pred,
                target,
                logstd=None,
                logmixweight=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.log_sigma is not None:
            if isinstance(self.log_sigma, nn.Parameter):
                logstd = self.log_sigma.expand(pred.size(-1))
            else:
                logstd = pred.new_full((pred.size(-1), ), self.log_sigma)
        loss = self.loss_weight * mvd_gaussian_mixture_nll_loss(
            pred,
            target,
            weight,
            logstd=logstd,
            logmixweight=logmixweight,
            adaptive_weight=self.adaptive_weight,
            momentum=self.momentum,
            mean_inv_std=self.mean_inv_std,
            dim=self.dim,
            delta=self.delta,
            eps=self.eps,
            training=self.training,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
