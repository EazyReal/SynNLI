from allennlp.common import Registrable
from overrides import overrides
#from torch_geometric.nn import GATConv, RGCNConv, CGConv

class Graph2GraphEncoder(Registrable):
    """
    A `Graph2GraphEncoder` is a `Module` that
    takes a graph and returs its updated version
    
    Input shape :
        `(SparseBatch, SparseGraphBatch)`;
    output shape:
        `(SparseBatch, SparseGraphBatch)`.
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
    pass