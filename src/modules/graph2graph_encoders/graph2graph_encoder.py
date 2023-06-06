from typing import Callable

import torch
import torch_geometric ## nn __init__ has from .conv import * can use directly
from overrides import overrides

from allennlp.common import Registrable

from .RGCNConv import RGCNConv
from .HGTConv import HGTConv
from .GATConv import GATConv
from .CGConv import CGConv

"""
An `Graph2GraphEncoder` is known as `GraphConvolutionLayer`
that takes some graph input and return its updated version.
For the most part we just use
[PyTorch Geometric GraphConvs](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html).
Here we provide a thin wrapper to allow registering them and instantiating them `from_params`.
The available options are
* ["gat"](link_to_gat)
* ["rgcn"](link_to_rgcn)
"""

"""
https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2seq_encoders/pytorch_seq2seq_wrapper.py
"""

class Graph2GraphEncoder(torch.nn.Module, Registrable):
    """
    A `Graph2GraphEncoder` is a `Module` that
    takes a graph and returs its updated version
    
    the API to outside world are `self.in_channels`, `self.out_channels`
    
    Input shape :
        `(SparseBatch, SparseGraphBatch)`;
    output shape:
        `(SparseBatch, SparseGraphBatch)`.
    """
    
    def get_input_dim(self) -> int: # not used
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2VecEncoder`.
        """
        raise NotImplementedError
        
    def get_output_dim(self) -> int: # not used
        """
        Returns the dimension of the final vector output by this `Seq2VecEncoder`.  This is `not`
        the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError


# There are no classes to decorate, so we hack these into Registrable._registry.
# If you want to instantiate it, you can do like this:
# <T:Registrable>.by_name('relu')()
Registrable._registry[Graph2GraphEncoder] = {
    "gat": (GATConv, None),
    #"rgcn": (torch_geometric.nn.conv.RGCNConv, None), 
    "rgcn": (RGCNConv, None),
    "hgt": (HGTConv, None),
    "cg": (CGConv, None),
    # "linear": (lambda: _Graph2GraphEncoderLambda(lambda x: x, "Linear"), None),  # type: ignore
}
