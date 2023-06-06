from typing import Callable

import torch
import torch_geometric ## nn __init__ has from .conv import * can use directly
from overrides import overrides

from allennlp.common import Registrable

#from .attention import GlobalAttention recursive import


class Graph2VecEncoder(torch.nn.Module, Registrable):
    """
    A `Graph2VecEncoder` is a `Module` that
    takes a graph and returs its pooled representation
    
    Input shape :
        `(SparseBatch of Seq)`;
    output shape:
        `(Batch)`.
    """
    pass

# There are no classes to decorate, so we hack these into Registrable._registry.
# If you want to instantiate it, you can do like this:
# <T:Registrable>.by_name('relu')()
#Registrable._registry[Graph2VecEncoder] = {
#    "global_attention": (GlobalAttention, None), # init takes gate_nn and nn
#    #"linear": (lambda: _Graph2GraphEncoderLambda(lambda x: x, "Linear"), None),  # type: ignore
#}
