"""
Attention modules for the `brunch` project
"""

__all__ =[
    'SelfAttention'                     # From buttermilk
    'SelfAttentionPancake'              # From pancake TODO: Check that it is the same as the buttermilk one!!
]

import torch
import torch.nn.functional as F

class SelfAttention(torch.nn.Module):
    """
    Parametric self attention mechanism.

    This implementation uses learnable linear projections to transform the
    input into query, key, and value representations, computes scaled dot
    product attention scores, normalizes these scores to obtain attention
    weights, and then aggregates the values accordingly. An additional linear
    transformation is applied to the weighted sum to produce the final output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        device: str = 'cpu',
    ):
        """
        Initialize `SelfAttention`.

        Parameters
        ----------
        input_dim : int
            Size of the (single) input dimension.
        output_dim : int
            Size of the (single) output tensor.
        hidden_dim : int
            Size of the dimension used for the intermediate representations of
            the queries, keys, and values.

        Attributes
        ----------
        query : nn.Linear
            Linear layer that projects the input into the query space
        key : nn.Linear
            Layer that projects the keys into the key space (of shape
            `hidden_dim`)
        value : nn.Linear
            Layer that projects the values into the common intermediate space
            of shape `hidden_dim`
        output : nn.Linear
            Layer that maps the resultant of the attention mechanism to the
            output space of shape `output_dim`
        """

        # Initialize parent module
        super(SelfAttention, self).__init__()

        # Set instance attributes
        self.device = device
        self.query = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.key = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.value = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.output = torch.nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of `SelfAttention`

        Parameters
        ----------
        x : torch.Tensor
            The query tensor of shape (B, C, input_dim)
        """
        x = x.to(self.device)

        # Map the query, key, and value, into their respective spaces
        Q = self.query(x)                               # (B, A, hidden_dim)
        K = self.key(x)                                 # (B, A, hidden_dim)
        V = self.value(x)                               # (B, A, hidden_dim)

        # Compute attention scores
        attention_scores = torch.matmul(                # (B, A, A)
            Q,
            K.transpose(-2, -1)
        ) / (K.size(-1) ** 0.5)

        # TODO: Should this really be log softmax?
        # Apply softmax to get attention weights
        attention_weights = F.log_softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32
        )

        # Convert attention weights from the log domain
        attention_weights = torch.exp(attention_weights)

        # Compute the weighted sum of values
        attention_output = torch.matmul(                # (B, A, hidden_dim)
            attention_weights,
            V
        )

        # Another linear layer
        output = self.output(attention_output)          # (B, A, G)

        return output
