# Loss Functions Rules (TTS)

## Core Training Losses

### 1. Flow Matching Loss (Primary)

```python
def flow_matching_loss(
    model_output: torch.Tensor,  # predicted velocity
    target_velocity: torch.Tensor,  # ground truth velocity
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.mse_loss(model_output, target_velocity, reduction='none')
    if mask is not None:
        loss = loss * mask
    return loss.sum() / mask.sum() if mask is not None else loss.mean()
```

Flow matching trains the model to predict the velocity field along the ODE path from noise to data.

### 2. Duration Prediction Loss

```python
def duration_loss(
    pred_duration: torch.Tensor,  # (B,)
    target_duration: torch.Tensor,  # (B,)
) -> torch.Tensor:
    return F.l1_loss(pred_duration, target_duration)
```

### 3. Reconstruction Loss (if using vocoder jointly)

```python
def reconstruction_loss(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    loss_type: str = "l1",  # or "l2", "stft"
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(pred_wav, target_wav)
    elif loss_type == "stft":
        return stft_loss(pred_wav, target_wav)
```

## Auxiliary Losses

### 4. Style Consistency Loss

Ensures style vectors remain consistent across different texts from the same speaker:

```python
def style_consistency_loss(
    style_pred: torch.Tensor,  # predicted from audio
    style_target: torch.Tensor,  # reference style
) -> torch.Tensor:
    return F.mse_loss(style_pred, style_target)
```

### 5. Prosody Loss

For natural prosody modeling:

```python
def prosody_loss(
    pred_pitch: torch.Tensor,
    target_pitch: torch.Tensor,
    pred_energy: torch.Tensor,
    target_energy: torch.Tensor,
) -> torch.Tensor:
    pitch_loss = F.mse_loss(pred_pitch, target_pitch)
    energy_loss = F.mse_loss(pred_energy, target_energy)
    return pitch_loss + energy_loss
```

## Loss Weights

Typical configuration:

```yaml
loss_weights:
  flow_matching: 1.0
  duration: 0.1
  reconstruction: 0.5  # if training vocoder jointly
  style_consistency: 0.01
  prosody: 0.05
```

## Implementation Rules

- Each loss in its own file under `losses/`
- All losses return a scalar tensor
- All losses accept a `weight` parameter (default 1.0)
- Loss weights configured in YAML, not hardcoded
- Combined loss logged as total + individual components to WandB

## Gradient Safety

- Clip gradients to max_norm=1.0 by default
- Monitor gradient norms per loss component
- If any loss produces NaN, skip that batch and log warning
- Use gradient accumulation for large batch sizes

## Loss Function Signature

```python
class FlowMatchingLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # scalar
        ...
        return loss * self.weight
```
