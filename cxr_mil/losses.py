from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for long-tail multi-label classification."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        pos_prob = probs
        neg_prob = 1.0 - probs

        if self.clip > 0:
            neg_prob = (neg_prob + self.clip).clamp(max=1.0)

        pos_loss = targets * torch.log(pos_prob.clamp(min=self.eps))
        neg_loss = (1.0 - targets) * torch.log(neg_prob.clamp(min=self.eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = targets * pos_prob + (1.0 - targets) * neg_prob
            gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
            focal_weight = torch.pow(1.0 - pt, gamma)
            loss = (pos_loss + neg_loss) * focal_weight
        else:
            loss = pos_loss + neg_loss

        return -loss.mean()
