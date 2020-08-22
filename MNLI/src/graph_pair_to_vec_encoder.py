from allennlp.common import Registrable
from overrides import overrides
from torch_geometric.nn import GATConv, RGCNConv, CGConv

convs = {
    "gat" : GATConv(in_channels=, out_channels=, heads=1),
    "gcn" : ,
    "ggcn" : ,
    "cgc" : ,
}

class GraphPair2VecEncoder(Registrable):
    """
    A `GraphPair2VecEncoder` is a `Module` that takes
    two sequence of vectors and two graphs
    and returns
    a single vector.
    
    Input shape : `(SparseBatch, SparseGraphBatch)`;
    output shape: `(batch_size, output_dimension)`.
    """
    
    @overrides
    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2VecEncoder`.
        """
        raise NotImplementedError
        
    @overrides
    def get_output_dim(self) -> int:
        """
        Returns the dimension of the final vector output by this `Seq2VecEncoder`.  This is `not`
        the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError
        
class GraphConvolutionLayer(Registrable):
    """
    A `GraphConvolutionLayer` is a `Module` that
    takes a graph and returs its updated version
    Input shape : `(SparseBatch, SparseGraphBatch)`;
    output shape: `(SparseBatch, SparseGraphBatch)`.
    """
    pass
        
    
    
@GraphPair2VecEncoder.register("graph_embedding_net")
class GraphEmbeddingNet(GraphPair2VecEncoder):
    __slots__ = (
        "input_dim",
        "output_dim",
    )
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 conv_type : str,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if conv_type not in convs.keys():
            raise ConvNameNotSupportedError
        self.conv = convs[conv_type]
        return 
    
    def forward():
        return 
    
    def get_input_dim(self) -> int:
        return self.input_dim
    
    def get_output_dim(self) -> int:
        return self.output_dim
    
