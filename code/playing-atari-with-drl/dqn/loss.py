"""TD loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_td_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute loss between Q(s, a) and Bellman targets for taken actions.

    Targets should already mask bootstrap at terminal states (reward only when done).
    """
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    if loss_type == "huber":
        return F.smooth_l1_loss(q_sa, targets)
    return F.mse_loss(q_sa, targets)
