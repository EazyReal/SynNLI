from overrides import overrides
from typing import Optional, Dict, Iterable, List, Union

#import torch_geometric
import torch

from allennlp.common import Registrable

class GraphPair2VecEncoder(torch.nn.Module, Registrable):
    """
    A `GraphPair2VecEncoder` is a `Module` that takes
    two sequence of vectors and two graphs
    and returns
    a single vector.
    
    Input shape : `(SparseBatch, SparseGraphBatch)`;
    output shape: `(batch_size, output_dimension)`.
    """
    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2VecEncoder`.
        """
        raise NotImplementedError
        
    def get_output_dim(self) -> int:
        """
        Returns the dimension of the final vector output by this `Seq2VecEncoder`.  This is `not`
        the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

