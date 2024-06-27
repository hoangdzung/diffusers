
import torch
import torch.nn as nn
import numpy as np

def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()

class FSQ(nn.Module):
    """Quantizer."""

    def __init__(self, levels: int, eps: float = 1e-3):
        super(FSQ, self).__init__()
        self._levels = levels
        self._eps = eps

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        device = z.device
        self._levels_np = np.ones(z.shape[-1]) * self._levels
        half_l = torch.tensor((self._levels_np - 1) * (1 - self._eps) / 2, device=device)
        offset = torch.where(torch.tensor(np.mod(self._levels_np, 2) == 1, device=device), torch.tensor(0.0, device=device), torch.tensor(0.5, device=device))
        shift = torch.tan(offset / half_l).to(device)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = torch.tensor(self._levels_np // 2, device=z.device)
        return quantized / half_width

    def forward(self, x):
        return self.quantize(x).to(dtype=x.dtype)

