# Standard library imports
from typing import Optional

# Third-party imports
import torch
import einops as E
from torch import nn
from typing import Optional, Union, Literal


# Local imports
# from .multi_conv import MaxPool2d, Upsample
import neurite as ne
from .modules import ConvOp, MultiConvOp
from .utils.utils import reshape_features_to_k, make_joint_embeddings
from .position_embeddings import SinPositionEmbedding
from .utils.utils import get_nonlinearity
from .activations import LogSoftmaxClass, SoftmaxClass


def get_finalnonlinearity(nonlinearity: Optional[str], dim: Optional[int]=1) -> torch.nn.Module:
    if nonlinearity is None:
        return torch.nn.Identity()
    
    elif nonlinearity == 'logSoftmax':
        print('LogSoftmax activation selected for final layer')
        return LogSoftmaxClass(dim=dim)
    
    elif nonlinearity == "Softmax" or nonlinearity == "softmax":
        print('Softmax activation selected for final layer')
        # For Softmax, we need to specify the channel dimension
        return SoftmaxClass(dim=dim)
    
    elif nonlinearity == "Sigmoid":
        print('Sigmoid activation selected for final layer')
        return torch.sigmoid

    elif nonlinearity == "Identity":
        print('Identity activation selected for final layer')
        return torch.nn.Identity()
        
    if hasattr(torch.nn, nonlinearity):
        return getattr(torch.nn, nonlinearity)()
    raise ValueError(f"nonlinearity {nonlinearity} not found")


class PancakeStochasticFusion(nn.Module):

    def __init__(
        self,
        featurizer: nn.Module,
        protocol_sampler: nn.Module,
        candidate_sampler: nn.Module,
        stochastic_head: nn.Module = None,
        final_activation: Optional[str] = "logSoftmax",
        padding_mode: str = 'zeros',
        final_act_dim: Optional[int] = 1,
    ) -> None:
        """
        Initialize `PancakeStochasticFusion`

        Parameters
        ----------
        featurizer : torch.nn.Module
            Deterministic model/feature extractor.

        protocol_sampler : torch.nn.Module
            A module whose forward pass takes the following parameters:
                - `K`: The numver of stochastic predictions.
                - `B`: The expected number of batches in the feature tensor
                    after model prediction/feature extraction.
                - `spatial`: A list of the spatial dimensions expected in the
                    tensor of features after model prediction/feature
                    extraction.

        candidate_sampler : torch.nn.Module
            A module whose forward pass takes the following parameters:
                - `K`: The number of stochastic predictions.
                - `B`: The expected number of batches in the feature tensor
                    after model prediction/feature extraction.
                - `spatial`: A list of the spatial dimensions expected in the
                    tensor of features after model prediction/feature
                    extraction.

            And whose forward pass returns:
                - A tensor of shape (B, K, G, *spatial).

        stochastic_head : torch.nn.Module
            The final layer(s) responsible for producing stochastic predictions
            by processing the fused noise and features. Expects as input a
            tensor of shape (B * K, G + Co, *spatial), where `Co` represents
            the number of output channels from `featurizer`.
        padding_mode : str
            Padding mode to use in the final convolutional kernel. Default is
            'zeros'.
        """

        super().__init__()

        # Setting instance attributes
        self.featurizer = featurizer
        self.protocol_sampler = protocol_sampler
        self.candidate_sampler = candidate_sampler
        self.stochastic_head = stochastic_head
        self.final_conv = ne.pytorch.modules.ConvBlock(
            ndim=featurizer.ndim,
            in_channels=stochastic_head.out_channels,
            out_channels=1,
            kernel_size=1,
            padding=0,
            #padding_mode=padding_mode,
        )
        
        # used to be LogSoftmaxClass(1)
        self.final_activation = get_finalnonlinearity(
            nonlinearity=final_activation,
            dim=final_act_dim
        )

    def forward(
        self,
        tensor: torch.Tensor,
        M: int,
        K: int,
    ):
        """
        Forward pass of `PancakeStochasticFusion`
        
        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape (B, S, C, H, W).
        S : int
            Set dimension
        M : int
            Protocol dimenion
        K : int
            Candidate dimension

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, S, M, K, C, H, W)
        """

        # Unpack shape
        B, S, C, *spatial = tensor.shape
        Cout = self.featurizer.out_channels

        # Group batch and sample dimensions so we can do convs
        tensor = E.rearrange(                               # (B*S) C1 H W
            tensor=tensor,
            pattern='B S C H W -> (B S) C H W',
            B=B, S=S, C=C,
        )

        # Pass through featurizer
        tensor = self.featurizer(tensor)                    # (B*S) C2 H W

        # Unpack batch and sample dimensions
        tensor = E.rearrange(                               # B S C2 H W
            tensor=tensor,
            pattern='(B S) C H W -> B S C H W',
            B=B, S=S, C=Cout,
        )

        # print(f'Unet output shape: {tensor.shape}')
        # ne.plot.slices([*tensor[0, 0, :6, ...].cpu()], titles = ['b=0, s=0, c varies'])
        # ne.plot.slices([*tensor[0, :, 0, ...].cpu()], titles = ['b=0, c=0, s varies'])

        # print('tensor.shape: ', tensor.shape)

        # Add singletons to make room for protocol and candidate dims
        tensor = E.rearrange(                               # B S 1 1 C2 H W
            tensor=tensor,
            pattern='B S C H W -> B S 1 1 C H W',
            B=B, S=S, C=Cout,
        )
        # print('tensor.shape: ', tensor.shape)

        # Repeat to match embedding dim
        tensor = E.repeat(                                  # B S M K C2 H W
            tensor=tensor,
            pattern='B S 1 1 C H W -> B S M K C H W',
            B=B, S=S, M=M, K=K, C=Cout,
        )


        joint_embeddings = make_joint_embeddings(           
            self.protocol_sampler,
            self.candidate_sampler,
            B=B, S=S, M=M, K=K, spatial=spatial,
        )


        fused_features = torch.cat(                         # B S M K (C2+G) H W
            [
                tensor,
                joint_embeddings.to(tensor.device),
            ],
            dim=4
        )

        fused_features = E.rearrange(                       # (B S M K) (C2+G) H W
            tensor=fused_features,
            pattern='B S M K C H W -> (B S M K) C H W'
        )

        stochastic_predictions = self.stochastic_head(      # (B S M K) C3 H W
            fused_features
        )

        stochastic_predictions = self.final_conv(           # (B S M K) C4 H W
            stochastic_predictions
        )

        stochastic_predictions = E.rearrange(               # B S M K C4 H W
            tensor=stochastic_predictions,
            pattern="(B S M K) C H W -> B S M K C H W",
            B=B, S=S, M=M, K=K, C=1,
        )

        stochastic_predictions = self.final_activation(stochastic_predictions)

        return stochastic_predictions







def pancakesmodel(version: Literal["v1"] = "v1", pretrained: bool = False) -> nn.Module:
    weights = {
            "v1": "https://github.com/mariannerakic/Pancakes/releases/download/weights/pancakes_v1_model_weights_Neurips.pt"
            }
    if version == "v1":
        # Build featurizer, noise_module, and stochastic_head from config
        featurizer = ne.models.BasicUNet(ndim=2,
                                        in_channels=1,
                                        nb_features=[32, 32, 32, 32],
                                        out_channels=32,
                                        activations=torch.nn.PReLU,
                                        final_activation=torch.nn.PReLU)

        candidate_sampler = SinPositionEmbedding(G=6)
        protocol_sampler = SinPositionEmbedding(G=6)
        featurizer_out_channels = 32
        G = 12
        inchannels = featurizer_out_channels + G
        stochastic_head = ne.modules.ConvBlock(in_channels=inchannels,
                                                            ndim=2,
                                                            out_channels=32,
                                                            order="cacaca",
                                                            activation=[torch.nn.PReLU, torch.nn.PReLU, torch.nn.PReLU])

        model = PancakeStochasticFusion(
            featurizer=featurizer,
            candidate_sampler=candidate_sampler,
            protocol_sampler=protocol_sampler,
            stochastic_head=stochastic_head,
            final_activation='Softmax',
        )

    if pretrained:
       state_dict = torch.hub.load_state_dict_from_url(weights[version])
       model.load_state_dict(state_dict['model'])

    return model
