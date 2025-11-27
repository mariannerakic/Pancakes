"""
Position Embeddings for the `brunch` project.
"""

__all__ = [
    "SinPositionEmbedding",             # From buttermilk
]

import torch
from torch import nn
import einops as E

from .utils.utils import make_sin_position_embedding_frequencies, make_timesteps, compute_sin_embeddings, gvector_spatial_expansion


class SinPositionEmbedding(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(
        self,
        B: int,
        K: int,
        spatial: list,
    ) -> torch.Tensor:

        """
        Generate sinusoidal positional embeddings for candidates.

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

        # Compute the frequencies of the sinusoids
        embedding_frequencies = make_sin_position_embedding_frequencies(
            n_embeddings=self.G,
            n_timesteps=K,
        )

        # Compute the normalized timesteps (with the correct shape wrt freqs)
        timesteps = make_timesteps(
            n_embeddings=self.G,
            n_timesteps=K,
        )

        # Compute the embeddings
        embeddings = compute_sin_embeddings(embedding_frequencies, timesteps)

        # Add batch dimension
        embeddings = E.repeat(embeddings, 'K G -> B K G', B=B)

        # Expand the embeddings to match the shape of the input tensor
        expanded = gvector_spatial_expansion(
            interacted_noise=embeddings,
            spatial=spatial,
        )

        return expanded
