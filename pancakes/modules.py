"""
Neural network building blocks.
"""

__all__ = [
    "ConvOp",
    "GVectorModule",
    "MultiConv2d",
    "MultiConvOp",
    "MultiMaxPool2d",
    "MultiUpsample2d",
    "MultiBatchNorm2d",
    "SinPositionEmbedding",
]

# Standard library imports
from typing import Optional, Union, Tuple

# Third-party imports
import torch
from torch import nn
import einops as E
import torch.nn.functional as F

# Local imports
from .utils.init import reset_conv2d_parameters
from .utils.utils import get_nonlinearity, make_noise, gvector_spatial_expansion


class ConvOp(nn.Module):
    """
    Convolution operation with optional nonlinearity and norm.

    Notes
    -----
    The order of operations is as follows:
        1. Convolution
        2. Normalization (optional)
        3. Activation (optional)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        padding: Union[int, tuple] = 1,
        normalization: Optional[bool] = False,
        nonlinearity: Optional[str] = "LeakyReLU",
        init_distribution: Optional[str] = "kaiming_normal",
        init_bias: Union[None, float, int] = 0.0,
    ) -> None:

        """
        Initialize the `ConvOp` module.

        Parameters
        ----------
        in_channels : int
            Number of channels/features in the input tensor.
        out_channels : int
            Number of output channels/features.
        kernel_size : int or tuple of int, optional
            Size of the convolving kernel. Default is 3.
        padding : int or tuple of int, optional
            Zero-padding added to both sides of the input. Default is 1.
        normalization : bool
            Optionally apply (instance) normalization after convolution.
        nonlinearity : str or None, optional
            Nonlinear activation function to apply after the convolution.
            If None, no activation is applied. Default is "LeakyReLU".
        init_distribution : str, optional
            Initialization distribution to use for the convolution weights.
            Default is "kaiming_normal".
        init_bias : float, int, or None, optional
            Initial bias value for the convolutional layer. Default is 0.0.
        """

        # Initialize the parent class
        super().__init__()

        # Set instance attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.normalization = normalization
        self.nonlinearity = nonlinearity
        self.init_distribution = init_distribution
        self.init_bias = init_bias

        # Define Conv2d layer
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            padding_mode="zeros",
            bias=True,
        )

        # Optionally initiallize a normalization
        if self.normalization:
            self.normalization = nn.InstanceNorm2d(self.out_channels)

        # Optionally initiallize a nonlinearity
        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)
        else:
            self.nonlin = None

        # # Initialize parameters
        # reset_conv2d_parameters(
        #     self, self.init_distribution, self.init_bias, self.nonlinearity
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `ConvOp`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after applying convolution and nonlinearity.
        """

        # Extract the features with the covolution
        x = self.conv(x)

        # Optionally normalize
        if self.normalization:
            x = self.normalization(x)

        # Optionally apply nonlinearity
        if self.nonlin is not None:
            x = self.nonlin(x)
        
        return x


class MultiConv2d(nn.Conv2d):
    """
    Convolve all elements of `x` with shape (B, K, C, H, W) where K are
    multiple labels.

    MultiConv2d is a convolutional layer that performs pairwise convolutions
    between additional axis of an input tensor.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int]],
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: str = None,
        dtype: str = None,
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int or tuple of ints
            Number of channels in the input tensor(s). If the tensors have
            different number of channels, `in_channels`, must be a tuple.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple of ints, optional
            Size of the convolutional kernel. Default is 3.
        stride : int or tuple of ints, optional
            Stride of the convolution. Default is 1.
        padding : int or tuple of ints, optional
            Zero-padding added to both sides of the input. Default is 0.
        dilation : int or tuple of ints, optional
            Spacing between kernel elements. Default is 1.
        groups : int, optional
            Number of blocked connections from input channels to output
            channels. Default is 1.
        bias : bool, optional
            If True, adds a learnable bias to the output. Default is True.
        padding_mode : str, optional
            Padding mode. Default is "zeros".
        device : str, optional
            Device on which to allocate the tensor. Default is None.
        dtype : torch.dtype, optional
            Data type assigned to the tensor. Default is None.

        Returns
        -------
        torch.Tensor
            Tensor resulting from the pairwise convolution between the
            elements of `x` and `y`.

        Notes
        -----
        x is a tensor of size (B, K, C, H, W).
        """

        # Compute the correct `concat_channels` from the list (if list)
        if isinstance(in_channels, (list, tuple)):
            concat_channels = sum(in_channels)
        else:
            concat_channels = 2 * in_channels

        # Initialize parent
        super().__init__(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute convolution between all elements of `x` of shape
        (B, K, Cin, H, W) across dimension K.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, K, Cin, H, W).

        Returns
        -------
        torch.Tensor
            Tensor resulting from the cross-convolution between the dimensions
            K of `x`. Has size (B, K, Cout, H, W), where Co is the number of
            output channels.
        """

        # Unpack shape of input for einops stuff and general reshaping
        B, K, Cin, H, W = x.shape

        # Reduce/average across the candidate dimension
        xk = E.reduce(                                  # (B, 1, Cin, H, W)
            tensor=x,
            pattern="B K Cin H W -> B 1 Cin H W",
            reduction="mean"
        )

        # Repeat average over the candidate dimension
        xk = E.repeat(                                  # (B, K, Cin, H, W)
            tensor=xk,
            pattern="B 1 Cin H W -> B K Cin H W",
            K=K
        )

        # Concatenate along channel dimension 
        xsmk = torch.cat([x, xk], dim=2)                # (B, K, 2*Cin, H, W)

        # Gather B and K dims into the batch dim
        batched_xsmk = E.rearrange(                     # (B*K, 2*Cin, H, W)
            tensor=xsmk,
            pattern="B K C2 H W -> (B K) C2 H W"
        )

        # Apply the convolution to the batched/grouped data
        batched_output = super().forward(batched_xsmk)

        # Reconstitute the batch and candidate dimensions
        output = E.rearrange(                           # (B, K, Co, H, W)
            tensor=batched_output,
            pattern="(B K) Co H W -> B K Co H W",
            B=B,
            K=K
        )

        return output


class MultiConvOp(nn.Module):
    """
    Convolution operation for multiple segmentations with optional nonlinearity
    and norm.

    Notes
    -----
    The order of operations is as follows:
        1. Convolution (multiple labels)
        2. Normalization (optional)
        3. Activation (optional)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        padding: Union[int, tuple] = 1,
        normalization: Optional[bool] = False,
        nonlinearity: Optional[str] = "LeakyReLU",
        init_distribution: Optional[str] = "kaiming_normal",
        init_bias: Union[None, float, int] = 0.0,
    ) -> None:

        """
        Initialize the `MultiConvOp` module.

        Parameters
        ----------
        in_channels : int
            Number of channels/features in the input tensor.
        out_channels : int
            Number of output channels/features.
        kernel_size : int or tuple of int, optional
            Size of the convolving kernel. Default is 3.
        padding : int or tuple of int, optional
            Zero-padding added to both sides of the input. Default is 1.
        normalization : bool
            Optionally apply (instance) normalization after convolution.
        nonlinearity : str or None, optional
            Nonlinear activation function to apply after the convolution.
            If None, no activation is applied. Default is "LeakyReLU".
        init_distribution : str, optional
            Initialization distribution to use for the convolution weights.
            Default is "kaiming_normal".
        init_bias : float, int, or None, optional
            Initial bias value for the convolutional layer. Default is 0.0.
        """

        # Initialize the parent class
        super().__init__()

        # Set instance attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.normalization = normalization
        self.nonlinearity = nonlinearity
        self.init_distribution = init_distribution
        self.init_bias = init_bias

        # Define multiple convolution layer
        self.multi_conv = MultiConv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

        # Optionally initiallize a normalization
        if self.normalization:
            self.normalization = nn.InstanceNorm2d(self.out_channels)

        # Optionally initiallize a nonlinearity
        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)
        else:
            self.nonlin = None

        # Initialize parameters
        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `MultiConvOp`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after applying convolution and nonlinearity.
        """

        # Extract the features with the covolution
        x = self.multi_conv(x)

        # Optionally normalize
        if self.normalization:
            x = self.normalization(x)

        # Optionally apply nonlinearity
        if self.nonlin is not None:
            x = self.nonlin(x)
        
        return x


class MultiMaxPool2d(nn.Module):
    """
    Spatial (max) pooling layer for multiple candidates.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]] = 2,
        stride: Union[int, Tuple[int]] = None,
    ):
        """
        Initialize `MultiMaxPool2d`

        Parameters
        ----------
        kernel_size : int or tuple of int, optional
            Size of the convolving kernel. Default is 2.
        stride : int or tuple of ints, optional
            Stride of the convolution. Default is None.
        """

        # Initialize parent module
        super().__init__()

        # Assign instance attribute
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `MultiMaxPool2d`

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to pool. Expected shape (B, K, C, H, W) where K is
            the candidate dimension.

        Returns
        -------
        torch.Tensor
            Pooled tensor with shape (B, K, C, H, W).
        """

        # Get the number of candidates (for reshaping after pool)
        K = x.shape[1]

        # Gather candidate dim into the batch dim
        x = E.rearrange(
            tensor=x,
            pattern='B K C H W -> (B K) C H W'
        )

        # Apply the pooling operation
        x = self.pool(x)

        # Reconstitute the candidate dimension
        x = E.rearrange(x, '(B K) C H W -> B K C H W', K=K)

        return x


class MultiUpsample2d(nn.Module):
    """
    Spatial upsampling layer for multiple candidates.
    """

    def __init__(
        self,
        scale_factor=2,
        mode='bilinear',
        align_corners=True,
    ):
        """
        Initialize `MultiUpsample2d`

        Parameters
        ----------
        scale_factor : int or tuple of int, optional
            Factor by which to upsample the input tensor. Default is 2.
        mode : str, optional.
            The upsampling algorithm. By default, 'bilinear'.
        align_corners : bool, optional
            Align the corner elements of the inpt of the input
        """

        # Initialie parent
        super().__init__()

        # Assign instance attribute
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `MultiUpsample2d`

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to upsample. Expected shape (B, K, C, H, W) where K is
            the candidate dimension.

        Returns
        -------
        torch.Tensor
            Upsampled tensor with shape (B, K, C, H, W).
        """

        # Get the number of candidates (for reshaping after pool)
        K = x.shape[1]

        # Gather candidate dim into the batch dim
        x = E.rearrange(
            tensor=x,
            pattern='B K C H W -> (B K) C H W'
        )

        # Apply the upsampling
        x = self.upsample(x)

        # Reconstitute the candidate dimension
        x = E.rearrange(x, '(B K) C H W -> B K C H W', K=K)

        return x

class MultiBatchNorm2d(nn.Module):
    """
    Batch normalization layer for multiple candidates.
    """
    def __init__(
        self,
        out_channels: int,
    ):
        """
        Initialize `BatchNorm2d`

        Parameters
        ----------
        out_channels : int
            Number of channels/features for the batch normalization layer.
        """

        # Initialize the parent module
        super().__init__()

        # Assign instance attribute
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `BatchNorm2d`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be normalized. Expected shape (B, K, C, H, W)
            where K is the candidate dimension.

        Returns
        -------
        torch.Tensor
            Normalized tensor witdh shape (B, K, C, H, W).
        """

        # Get the number of candidates (for reshaping after pool)
        K = x.shape[1]

        # Gather candidate dim into the batch dim
        x = E.rearrange(
            tensor=x,
            pattern='B K C H W -> (B K) C H W'
        )

        # Apply the normalization layer
        x = self.batchnorm(x)

        # Reconstitute the candidate dimension
        x = E.rearrange(x, '(B K) C H W -> B K C H W', K=K)

        return x