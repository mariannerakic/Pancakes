"""
G-vector modules for the `brunch` project.
"""

__all__ = [
    "GVectorModule",                    
]

import torch
from torch import nn

from brunch.nn.attention import SelfAttention
from brunch.utils import make_noise, gvector_spatial_expansion


class GVectorModule(nn.Module):
    """
    Generate, interact, and reshape g-vector for concatenation with features.

    This module generates random, low resolution noise, computes its average
    across a specific dimension, concatenates the original noise with the
    average, and interacts the result with self-attention.
    """

    def __init__(
        self,
        G: int,
        attention_hidden_dims: int,
        reduction: str = 'mean',
        device: str = 'cpu',
    ) -> None:
        """
        Instantiate `GVectorModule`

        Parameters
        ----------
        G : int
            Number of noise elements per instance ("resolution")
        attention_hidden_dims : int
            The number of hidden dimensions in the `SelfAttention` layer.
        reduction : str
            Aggregation operation for reducing noise across the candidate
            dimension. Can be one of {'mean', 'sum', 'max', 'min', 'prod'}.
            Default is 'mean'
        """

        # Call super constructor
        super().__init__()

        # Assign instance attributes
        self.G = G
        self.reduction = reduction
        self.device = device
        self.noise_interactor = SelfAttention(
            input_dim=2 * G,
            output_dim=G,
            hidden_dim=attention_hidden_dims,
            device=self.device,
        )

    def forward(
        self,
        B: int,
        K: int,
        spatial: list,
    ) -> torch.Tensor:
        """
        Generate, interact, and spatially expand noise.

        Parameters
        ----------
        B : int
            Batch size of the feature tensor.
        K : int
            Number of stochastic candidates.
        spatial : list of int
            List representing the spatial dimensions (e.g., [height, width])
            of the feature tensor (for expanding the noise tensor).

        Returns
        -------
        torch.Tensor
            Interacted noise tensor of shape (B, K, G, *spatial).
        """

        # Get the combined noise!
        combined_noise = make_noise(                       # (B, K, 2G)
            B=B, K=K, G=self.G,
            reduction=self.reduction
        )

        # Interact the features using self-attention
        interacted_noise = self.noise_interactor(           # (B, K, G)
            combined_noise
        )

        # Expand the interacted noise to the appropriate spatial dimensions
        interacted_noise = gvector_spatial_expansion(       # (B, K, G, H, W)
            interacted_noise=interacted_noise,
            spatial=spatial,
        )

        return interacted_noise
