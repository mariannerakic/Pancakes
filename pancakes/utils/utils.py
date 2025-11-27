"""
Utils for the `brunch` project
"""

__all__ = [
    'get_nonlinearity',
    'reshape_features_to_k',
    'make_noise',
    'gvector_spatial_expansion',
    'make_sin_position_embedding_frequencies',
    'make_timesteps',
    'compute_sin_embeddings',
    'compute_original_sin_embeddings',
    'log_epoch_slices',
    'make_joint_embeddings',
    'make_old_sinusoidal_embedding'
]

# Standard Library imports
from typing import Union, Tuple
import subprocess
import re
import os
import getpass
import math

# Third-party imports
import torch
import wandb
from torch import nn
import einops as E
import matplotlib.pyplot as plt


def get_nonlinearity(
    nonlinearity: Union[str, nn.Module]
) -> nn.Module:
    """
    Get an instance of a nonlinearity.

    Parameters
    ----------
    nonlinearity : str or nn.Module or none
        The nonlinearity to instantiate and return.

    Return
    ------
    nn.Module
        The instantiated nonlinearity.
    """

    # Return the identity if no nonlinearity is desired.
    if nonlinearity is None:
        return nn.Identity()

    # If softmax, returh the softmax over the first (feature) dimension
    if nonlinearity == "Softmax":
        # For Softmax, we need to specify the channel dimension
        return nn.Softmax(dim=1)

    if hasattr(torch.nn, nonlinearity):
        return getattr(torch.nn, nonlinearity)()

    # Handle a nonlineaerit class
    if isinstance(nonlinearity, nn.Module):
        return getattr(torch.nn, nonlinearity)()

    raise ValueError(f"nonlinearity {nonlinearity} not found")


def reshape_features_to_k(features: torch.Tensor, K: int):
    """
    nD function to reshape a tensor of features from (B, C, *spatial) to 
    (B, K, C, *spatial).

    Parameters
    ----------
    features : torch.Tensor
        Output of model/featurizer with shape (B, C, *spatial), where
        `spatial` represents a tuple of the shape of the spatial dimensions.
    K : int
        The number of candidates.

    Returns
    -------
    torch.Tensor
        `features` reshaped to (B, K, C, *spatial)
    """

    # Add a channel between batch and features
    reshaped_features = features.unsqueeze(1)               # (B, 1, C, H, W)

    # Repeat K times along K dimension
    reshaped_features = E.repeat(                           # (B, K, C, H, W)
        reshaped_features,
        "B 1 C H W -> B K C H W",
        K=K
    )

    return reshaped_features


def make_noise(B: int, K: int, G: int, reduction: str = 'mean'):
    """
    Make Gaussian noise with appropriate size for interacting.

    Parameters
    ----------
    B : int
        The batch size of the featurizer's output tensor
    K : int
        The number of candidates.
    G : int
        The number of noise elements for each candidate.
    reduction : str
        Aggregation operation for reducing noise across the candidate
        dimension. Can be one of {'mean', 'sum', 'max', 'min', 'prod'}.
        Default is 'mean'

    Returns
    -------
    torch.Tensor
        The combined (noise, avg_noise) noise of shape (B, K, 2G)
    """

    # Sample the noise
    noise = torch.randn((B, K, G))                          # (B, K, G)

    # Average the noise over the entire K axis
    reduced_noise = E.reduce(                               # (B, 1, G)
        tensor=noise,
        pattern="B K G -> B 1 G",
        reduction=reduction
    )

    # Expand the noise to turn it back to the original shape (of `noise`)
    reduced_noise = E.repeat(                               # (B, K, G)
        tensor=reduced_noise,
        pattern="B 1 G -> B K G",
        K=K
    )

    # Combine the individual noise vectors with the average noise vectors
    combined_noise = torch.cat([noise, reduced_noise], dim=-1)  # (B, K, 2G)

    return combined_noise


def gvector_spatial_expansion(
    interacted_noise: torch.Tensor,
    spatial: list,
) -> torch.Tensor:
    """
    Expand the interacted noise of shape (B, K, G) to the full spatial
    dimensions and shape of the feature tensor for concatenation.

    Parameters
    ----------
    interacted_noise : torch.Tensor
        Interacted noise of shape (B, K, G).
    spatial : list of int
        List representing the spatial dimensions (e.g., [height, width])
        of the feature tensor (for expanding the noise tensor).

    Returns
    -------
    torch.Tensor
        Noise with shape (B, K, G, *spatial), ready for concatenation along
        the G dimension with the feature tensor.
    """

    # Generate unique names for each spatial dimension.
    spatial_dims = [f"s{i}" for i in range(len(spatial))]

    # Construct the pattern string for einops.repeat.
    pattern = "B K G -> B K G " + " ".join(spatial_dims)

    # Use einops.repeat to add new dimensions and repeat along them.
    interacted_noise = E.repeat(                            # (B, K, G, H, W)
        interacted_noise,
        pattern,
        **dict(zip(spatial_dims, spatial))
    )

    return interacted_noise


def make_sin_position_embedding_frequencies(
    n_embeddings: int,
    n_timesteps: int,
) -> torch.Tensor:
    """
    Make angular frequencies for sinusoidal timestep embeddings.

    This function creates a tensor of angular frequencies intended for 
    sinusoidal position embeddings. It calculates an exponential scaling 
    factor based on embedding indices (repeated in pairs) to modulate the
    frequency at each index. The resulting tensor has shape (n_timesteps,
    n_embeddings), where the frequencies are repeated through the rows (second
    dimension). The function is specifically designed for position embeddings
    whose timesteps are normalized to [0, 1].

    Parameters
    ----------
    n_embeddings : int
        The total number of embeddings. Must be a multiple of two.
    n_timesteps : int
        The number of timesteps for which the embedding frequencies are 
        generated.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_timesteps, n_embeddings) with the computed
        frequencies, repeated for each timestep.

    Examples
    --------
    >>> Example 1: Frequencies for a single timestep/position
    >>> # Make the frequencies for a single position embedding described by
    >>> # four sinusoids
    >>> make_sin_position_embedding_frequencies(4, 1)
    tensor([[3.1416, 3.1416, 9.3327, 9.3327]])

    >>> # Example 2: Frequencies for multiple timestep/position
    >>> # Make the frequencies for 3 timesteps described by 6 sinusoids
    >>> make_sin_position_embedding_frequencies(6, 3)
    tensor([
        [ 3.1416,  3.1416,  6.4921,  6.4921, 13.4161, 13.4161],
        [ 3.1416,  3.1416,  6.4921,  6.4921, 13.4161, 13.4161],
        [ 3.1416,  3.1416,  6.4921,  6.4921, 13.4161, 13.4161]
    ])
    """

    assert n_embeddings % 2 == 0, (
        f"`n_embeddings` must be a multiple of 2. Got {n_embeddings}"
    )

    # Make the indices to control the frequencies
    embedding_idx = torch.arange(0, n_embeddings // 2).repeat_interleave(2)

    # Compute the factor for exponentiating the frequencies
    exponential_factor = (2 * torch.pi * embedding_idx) / n_embeddings

    # Compute the embedding frequencies
    embedding_frequencies = torch.pi * (2 ** exponential_factor)

    # Unsqueeze to add `n_timesteps` dimension
    embedding_frequencies = embedding_frequencies.unsqueeze(0)

    # Repeat over `n_timesteps` dimension
    embedding_frequencies = embedding_frequencies.repeat(n_timesteps, 1)

    return embedding_frequencies


def make_timesteps(
    n_embeddings: int,
    n_timesteps: int,
) -> torch.Tensor:
    """
    Make timesteps with the correct shape for position embeddings.

    Parameters
    ----------
    n_embeddings : int
        The total number of embeddings. Must be a multiple of two.
    n_timesteps : int
        The number of timesteps for which the embedding frequencies are
        generated.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_timesteps, n_embeddings) containing normalized
        timesteps.
    """

    # Make discrete timesteps on the range [0, n_timesteps-1]
    timesteps = torch.arange(0, n_timesteps).unsqueeze(1).float()

    # Normalize! This is a requirement for these timesteps!
    timesteps /= timesteps.max()

    # Expand along `n_embeddings` dimension to match the shape of the frequencies
    timesteps = timesteps.repeat(1, n_embeddings)

    return timesteps


def compute_sin_embeddings(
    embedding_frequencies: torch.Tensor,
    timesteps: torch.Tensor,
    phase_shift: float = torch.pi/2
) -> torch.Tensor:
    """
    Compute sinusoidal position embeddings.

    Generate sinusoidal embeddings by multiplying the angular frequencies
    with the timesteps and applying sine functions to alternating columns.
    A positive phase shift is added to even-indexed columns and a negative
    shift to odd-indexed columns to create sine pairs with different phases,
    starting at either -1 or 1 (when the timestep == 0)

    Parameters
    ----------
    embedding_frequencies : torch.Tensor
        A tensor of angular frequencies with shape (n_timesteps, n_embeddings).
        Generated by `make_sin_position_embedding_frequencies()`.
    timesteps : torch.Tensor
        A tensor of *normalized* timesteps with shape (n_timesteps,
        n_embeddings), generated by `make_timesteps()`.
    phase_shift : float, optional
        The phase shift to apply to the sine functions. Default is pi/2.

    Returns
    -------
    torch.Tensor
        A tensor of sinusoidal position embeddings with the same shape as
        the input tensors (n_timesteps, n_embeddings)
    """

    # Compute the sin arguments by multiplying frequencies and timesteps
    embeddings = embedding_frequencies * timesteps

    # Apply sins with alternating phase shifts to alternating columns
    embeddings[:, ::2] = torch.sin(embeddings[:, ::2] + phase_shift)
    embeddings[:, 1::2] = torch.sin(embeddings[:, 1::2] - phase_shift)

    return embeddings


def compute_original_sin_embeddings(
    timesteps: torch.Tensor,
    denominator_scale: float = 10_000,
    d_model: int = 1
) -> torch.Tensor:
    """
    Compute sin/cos embeddings as proposed in Attention Is All You Need. [1]

    Parameters
    ----------
    timesteps : torch.Tensor
        Tensor of timesteps with shape (n_timesteps, n_embeddings). Likely
        derived from `brunch.utils.make_timesteps()`
    denominator_scale : float
        The scale of the denominator when computing the frequency for the
        sinusiods. Default from [1] is 10,000.
    d_model : int
        The number of dimensions for the vector embedding in [1]. They used a
        default value of 512. Not sure how to set this for our UNet, but might
        be good to have as an option. Default is 1.

    Returns
    -------
    torch.Tensor
        A tensor of sinusoidal position embeddings with the same shape as
        the input tensor: (n_timesteps, n_embeddings). Each row represents a
        `n_embeddings`-dimensional vector describing a single timestep.

    Examples
    --------
    >>> # Example 1: Making 2-dimensional embeddings for 4 timesteps
    >>> timesteps = make_timesteps(n_embeddings=6, n_timesteps=4)
    >>> embeddings = compute_original_sin_embeddings(timesteps=timesteps)
    >>> print(embeddings)
    tensor(
        [
            [0.0000, 1.0000],  # <- Embeddings describing 1st timestep
            [0.3272, 0.9450],  # <- Embeddings describing 2nd timestep
            [0.6184, 0.7859],  # <- Embeddings describing 3rd timestep
            [0.8415, 0.5403],  # <- Embeddings describing 4th timestep
        ]
    )

    References
    ----------
    1. Attention is all you need: https://arxiv.org/abs/1706.03762
    """

    # Make the indices for the embedding dimension: [0, 0, 1, 1, ...]
    embedding_idx = torch.arange(
        0, timesteps.shape[1] // 2).repeat_interleave(2)

    # Compute the factor by which we will exponentiate the denominator
    denominator_exponentiation_factor = 2 * embedding_idx / d_model

    # Compute the denominator. AKA the angular frequency!!
    denominator = denominator_scale ** denominator_exponentiation_factor
    denominator = denominator.unsqueeze(0)

    # Repeat denominator for all timesteps
    denominator = denominator.repeat(timesteps.size(0), 1)

    # Compute sin and cosin for each embedding idx
    timesteps = timesteps / denominator
    timesteps[:, 0::2] = timesteps[:, 0::2].sin()
    timesteps[:, 1::2] = timesteps[:, 1::2].cos()

    return timesteps

def make_old_sinusoidal_embedding(time, G):
    half_dim = G // 2
    time_space = torch.linspace(0, time, time, device='cuda')
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device='cuda') * -embeddings)
    embeddings = time_space[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings


def log_epoch_slices(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    epoch: int,
    n_candidates: int,
) -> None:
    """
    Run inference on the first sample, plot each candidate, and log as a single
    multi-panel image to W&B.
    """

    model.eval()
    # Prep input
    x, _ = dataset[0]
    # h, w = crop_size
    x = x.to(device)[None]

    # Forward
    # K = ne.samplers.RandInt(3, 9)()
    K = n_candidates
    with torch.no_grad():
        pred = model(x, K=K).sigmoid()

    # Threshold
    pred = (pred >= 0.5).float().squeeze(0).cpu().numpy()

    # Get num candidates
    n_slices = pred.shape[0]

    # Plot subplots
    fig, axes = plt.subplots(1, n_slices, figsize=(n_slices*3, 3))

    for i, ax in enumerate(axes):
        ax.imshow(pred[i][0], cmap="gray", interpolation="none")
        ax.set_title(f"Slice {i}")
        ax.axis("off")

    # Log and cleanup
    wandb.log({"pred_slices": wandb.Image(fig)})#, step=epoch)
    plt.close(fig)
    model.train()


def make_joint_embeddings(
    protocol_embedding_module,
    candidate_embedding_module,
    B: int,
    S: int,
    M: int,
    K: int,
    spatial: Tuple[int, int]
):
    """
    Returns
    -------
    torch.Tensor
        Tensor of shape B S M K (Gk+Gm) H W
    """

    protocol_embedding = protocol_embedding_module(         # B M Gm H W
        B=B,
        K=M,
        spatial=spatial
    )

    candidate_embedding = candidate_embedding_module(       # B K Gk H W
        B=B,
        K=K,
        spatial=spatial
    )

    # print(protocol_embedding.shape)
    # Add singleton dimension to make room for candidates
    protocol_embedding = E.rearrange(                       # B M 1 Gm H W
        tensor=protocol_embedding,
        pattern="B M G H W -> B M 1 G H W"
    )

    # Repeat over candidate dimension for correct size
    protocol_embedding = E.repeat(                          # B M K Gm H W
        tensor=protocol_embedding,
        pattern="B M 1 G H W -> B M K G H W",
        K=K
    )

    # Add singleton dimension to make room for protocols
    candidate_embedding = E.rearrange(                      # B 1 K Gk H W
        tensor=candidate_embedding,
        pattern="B K G H W -> B 1 K G H W"
    )

    # Repeat over protocol dimension for correct size
    candidate_embedding = E.repeat(                         # B M K Gk H W
        tensor=candidate_embedding,
        pattern="B 1 K G H W -> B M K G H W",
        M=M
    )

    # Concat along the embedding dimension
    combined_embedding = torch.cat(                         # B M K Gk+Gm H W
        [
            protocol_embedding,
            candidate_embedding,
        ],
        dim=3
    )

    combined_embedding = E.rearrange(                       # B 1 M K Gk+Gm H W
        tensor=combined_embedding,
        pattern='B M K G H W -> B 1 M K G H W'
    )

    combined_embedding = E.repeat(                          # B S M K Gk+Gm H W
        tensor=combined_embedding,
        pattern='B 1 M K G H W -> B S M K G H W',
        S=S
    )     

    return combined_embedding



