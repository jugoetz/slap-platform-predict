from typing import Callable, List, Optional, Union

import torch.nn as nn

from src.layer.util import get_activation


class FFN(nn.Module):
    """
    A simple feed-forward network (FFN) containing dropout layers.

    Identical to chemprop's FFN implementation if batch_norm is False. In addition, we support multiple different
    hidden sizes. To emulate chemprop behavior, keep all hidden sizes the same.

    For in_size, the MPNN output size has to be used, which can be (mpnn_hidden_size * number of molecules)
    or (mpnn_hidden_size * number of molecules + features size) if additional features are used,
    or atom_descriptors_size if descriptors (e.g. ECFP) are used,
    or features_size if only features are used.

    Args:
        in_size: input feature size
        hidden_sizes: sizes for hidden layers in FFN
        dropout_ratio: Probability of dropout
        activation: activation function for hidden layers
        hidden_bias: Whether to add bias to hidden layers
        out_size: size of output layer
        out_bias: bias for output layer, this use set to False internally if
            out_batch_norm is used.
        out_sigmoid: Whether to apply sigmoid to output
    """

    def __init__(
            self,
            in_size: int,
            hidden_sizes: List[int],
            *,
            dropout_ratio: float = 0.0,
            activation: Union[Callable, str] = "ReLU",
            hidden_bias: bool = True,
            out_size: Optional[int] = None,
            out_bias: bool = True,
            out_sigmoid: bool = False,
    ):
        super().__init__()
        self.num_hidden_layers = len(hidden_sizes)

        self.has_out_layer = out_size is not None

        dropout = nn.Dropout(dropout_ratio)

        layers = []

        # hidden layers
        for i, size in enumerate(hidden_sizes):
            layers.append(dropout)
            layers.append(nn.Linear(in_size, size, bias=hidden_bias))

            if activation is not None:
                layers.append(get_activation(activation))

            in_size = size

        # output layer
        if out_size is not None:
            layers.append(dropout)
            layers.append(nn.Linear(in_size, out_size, bias=out_bias))

        if out_sigmoid:
            layers.append(nn.Sigmoid())

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)

    def __repr__(self):
        s = f"FFN, num hidden layers: {self.num_hidden_layers}"
        if self.has_out_layer:
            s += "; with output layer"
        return s
