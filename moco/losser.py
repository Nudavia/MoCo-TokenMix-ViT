# -*-coding:utf-8-*-
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch import nn


def contrastive_loss(mix=False, smoothing=None):
    if mix:
        # smoothing is handled with mix label transform
        criterion = SoftTargetCrossEntropy()
    elif smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


class ContrastiveLoss(nn.Module):
    def __init__(self, smoothing=0.1, T=1.0):
        super(ContrastiveLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.T = T

    def forward(self, logits: torch.Tensor, target: torch.Tensor, use_mix=False) -> torch.Tensor:
        N = logits.shape[0]  # batch size per GPU
        num_classes = logits.shape[1]
        # 加上偏移量
        target = (target + N * torch.distributed.get_rank()).cuda()
        if use_mix:
            off_value = self.smoothing / N
            true_num = target.shape[1]
            on_value = (1.0 - self.smoothing) / true_num + off_value
            target = self._one_hot(target, num_classes, on_value, off_value)
            return SoftTargetCrossEntropy()(logits, target) * (2 * self.T)
        elif self.smoothing > 0.0:
            return LabelSmoothingCrossEntropy(self.smoothing)(logits, target) * (2 * self.T)
        else:
            return nn.CrossEntropyLoss()(logits, target) * (2 * self.T)

    def _one_hot(self, x, num_classes, on_value=1., off_value=0., device='cuda'):
        x = x.long()
        return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)
