"""Walsh-Hadamard Transform (matches firmware wht32.c).

32-point WHT via butterfly decomposition. Both numpy (codec) and
torch (neural encoder/decoder) variants.
"""
import numpy as np


def forward_32(x):
    """32-point Walsh-Hadamard Transform (unnormalized).

    In-place butterfly decomposition: 5 stages x 16 butterflies = 80 add/sub.
    Matches firmware wht32_forward() exactly.

    Args:
        x: numpy array of length 32
    Returns:
        transformed array (same type as input)
    """
    if len(x) != 32:
        raise ValueError(
            f"WHT-32 requires exactly 32 elements, got {len(x)}")
    x = x.copy() if isinstance(x, np.ndarray) else x.clone()
    N = 32
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    return x


def inverse_32(x):
    """32-point inverse Walsh-Hadamard Transform.

    WHT is self-inverse: H * H = N * I, so inverse = forward + divide by N.

    Args:
        x: numpy array of length 32
    Returns:
        inverse-transformed array
    """
    y = forward_32(x)
    return y / 32.0


def forward_32_torch(latent):
    """Apply 32-point WHT to latent tensor [B, 32, T].

    PyTorch-compatible, differentiable (WHT is linear -> autograd works).
    Uses the Hadamard matrix directly for GPU efficiency.

    Args:
        latent: [B, 32, T] torch tensor
    Returns:
        WHT-transformed latent [B, 32, T]
    """
    import torch

    if latent.shape[1] != 32:
        raise ValueError(
            f"WHT-32 requires dim 1 == 32, got shape {latent.shape}")

    # Build 32x32 Hadamard matrix (unnormalized, matches firmware)
    H = torch.tensor([[1.0]], device=latent.device, dtype=latent.dtype)
    for _ in range(5):  # log2(32) = 5
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)

    # Apply: result[b, :, t] = H @ latent[b, :, t]
    return torch.einsum('ij,bjt->bit', H, latent)


def inverse_32_torch(latent):
    """Inverse 32-point WHT: forward WHT + divide by 32."""
    return forward_32_torch(latent) / 32.0


__all__ = ['forward_32', 'inverse_32', 'forward_32_torch', 'inverse_32_torch']
