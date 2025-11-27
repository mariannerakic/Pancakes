"""
Position Embeddings for the `brunch` project.
"""

__all__ = [
    "SinPositionEmbedding",             # From buttermilk
]

import torch
from torch import nn
import einops as E
import brunch


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
        embedding_frequencies = brunch.utils.make_sin_position_embedding_frequencies(
            n_embeddings=self.G,
            n_timesteps=K,
        )

        # Compute the normalized timesteps (with the correct shape wrt freqs)
        timesteps = brunch.utils.make_timesteps(
            n_embeddings=self.G,
            n_timesteps=K,
        )

        # Compute the embeddings
        embeddings = brunch.utils.compute_sin_embeddings(embedding_frequencies, timesteps)

        # Add batch dimension
        embeddings = E.repeat(embeddings, 'K G -> B K G', B=B)

        # Expand the embeddings to match the shape of the input tensor
        expanded = brunch.utils.gvector_spatial_expansion(
            interacted_noise=embeddings,
            spatial=spatial,
        )

        return expanded
