from allennlp.common import Registrable


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
    def __init__(self, input_dim: int, output_dim: int, gconv : GraphConvolutionLayer, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        return 
    def forward:
        pass
    
    def get_input_dim(self) -> int:
        return self.input_dim
    
    def get_output_dim(self) -> int:
        return self.output_dim
    
