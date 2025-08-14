import torch
import torch.nn as nn

# Import the core PointWeb segmentation model
from model.pointweb.pointweb_seg import PointWebSeg as _PointWebSeg

class PointWebSegHead(nn.Module):
    """
    A thin wrapper for the PointWebSeg model that handles device placement
    and allows either [B, C, N] or [B, N, C] inputs.

    Args:
        c (int): Number of extra feature channels per point (beyond x,y,z).
        k (int): Number of segmentation classes.
        use_xyz (bool): Whether to include (x,y,z) as part of the learned features.
        device (str or torch.device): Device to run the model on ('cpu' or 'cuda').
    """
    def __init__(self, c: int, k: int, use_xyz: bool = True, device: str = 'cpu'):
        super().__init__()
        self.c = c
        self.k = k
        self.use_xyz = use_xyz
        self.device = torch.device(device)

        # Instantiate the underlying PointWeb segmentation network
        # PointWebSeg expects inputs of shape [B, N, 3 + c]
        self.model = _PointWebSeg(c=self.c, k=self.k, use_xyz=self.use_xyz)
        # Move both wrapper and core model to the target device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PointWeb.

        Accepts either:
        - x of shape [B, C, N] (will be transposed to [B, N, C])
        - x of shape [B, N, C]

        Returns:
            preds (torch.Tensor): Logits of shape [B, k, N]
        """
        if x.dim() != 3:
            raise ValueError(f"PointWebSegHead expects 3D input, got {x.dim()}D tensor")

        B, D, N = x.size()
        expected_C = 3 + self.c  # total features per point

        # If in [B, C, N] format, transpose to [B, N, C]
        if D == expected_C:
            x = x.transpose(1, 2).contiguous()
        elif N == expected_C:
            # already in [B, N, C]
            pass
        else:
            raise ValueError(
                f"Unexpected input shape {x.shape}; expected dims (*, {expected_C}, N) or (B, N, {expected_C})"
            )

        # Core PointWeb forward (returns [B, k, N])
        return self.model(x)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load a checkpoint into the underlying PointWebSeg core.
        """
        return self.model.load_state_dict(state_dict, strict=strict)

    def to(self, *args, **kwargs):
        """
        Override .to() so both wrapper and underlying model move devices.
        """
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True):
        self.model.train(mode)
        return super().train(mode)

    def eval(self):
        self.model.eval()
        return super().eval()