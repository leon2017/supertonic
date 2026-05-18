import torch
import torch.nn as nn
import torch.nn.functional as F


def generator_loss(disc_outputs: list[torch.Tensor]) -> torch.Tensor:
    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for out in disc_outputs:
        loss = loss + torch.mean((out - 1.0) ** 2)
    return loss / len(disc_outputs)


def discriminator_loss(real_outputs: list[torch.Tensor], fake_outputs: list[torch.Tensor]) -> torch.Tensor:
    loss = torch.tensor(0.0, device=real_outputs[0].device)
    for real, fake in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((real - 1.0) ** 2) + torch.mean(fake ** 2)
    return loss / len(real_outputs)


def feature_matching_loss(real_features: list[list[torch.Tensor]], fake_features: list[list[torch.Tensor]]) -> torch.Tensor:
    loss = torch.tensor(0.0, device=real_features[0][0].device)
    count = 0
    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feats, fake_feats):
            loss = loss + F.l1_loss(fake_f, real_f.detach())
            count += 1
    return loss / max(count, 1)
