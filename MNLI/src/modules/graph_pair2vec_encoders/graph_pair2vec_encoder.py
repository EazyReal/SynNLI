from allennlp.common import Registrable
from overrides import overrides
from torch_geometric.nn import GATConv, RGCNConv, CGConv

from src.modules.graph2graph_encoders import Graph2GraphEncoder
from src.modules.attention import GraphPairAttention


class GraphPair2VecEncoder(Registrable):
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
        
    
# add layer param same or not? (fiexed point view or difference stage view)
@GraphPair2VecEncoder.register("graph_embedding_net")
class GraphEmbeddingNet(GraphPair2VecEncoder):
    __slots__ = (
        "input_dim",
        "output_dim",
        "conv",
        "num_conv_layers"
    )
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 conv: Graph2GraphEncoder,
                 att: GraphPairAttention,
                 num_conv_layers: int,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = conv
        self.num_conv_layers = num_conv_layers
        return 
    
    def forward(g1, g2, n1, n2, **kwargs):
        """
        input:
            g1, g2 : sparse_adjacency batch
            n1, n2 : sparse_node_embedding batch
            e1, e2 : sparse_edge_embedding batch, Opt
        """
        g1, n1 = conv(g1, n1)
        
        
        
        return 
    
    def get_input_dim(self) -> int:
        return self.input_dim
    
    def get_output_dim(self) -> int:
        return self.output_dim
    
