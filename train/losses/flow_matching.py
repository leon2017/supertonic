import torch


def flow_matching_loss(
    model_output: torch.Tensor,
    target_velocity: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    loss = torch.abs(model_output - target_velocity)

    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)

    return loss.mean()


def compute_target_velocity(z1: torch.Tensor, z0: torch.Tensor, sigma_min: float = 1e-8) -> torch.Tensor:
    return z1 - (1 - sigma_min) * z0


def sample_zt(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor, sigma_min: float = 1e-8) -> torch.Tensor:
    t = t.view(-1, 1, 1)
    zt = (1 - (1 - sigma_min) * t) * z0 + t * z1
    return zt
