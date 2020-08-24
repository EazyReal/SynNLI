from overrides import overrides
from typing import Optional, Dict, Iterable, List, Union

import torch_geometric
import torch

from allennlp.common import Registrable

from src.modules.graph2graph_encoders import (
    Graph2GraphEncoder,
)
from src.modules.graph2vec_encoder import (
    Graph2VecEncoder,
)

from src.modules.attention import GraphPairAttention


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

# graph matching net can use List style to generate layer
# num_layers = 9
# ex: layers = [0, 0, 1, 2, 2, 0, 1, 0, 1] where 0 denotes convolution, 1 denotes graph matching, 2 denotes all
    
# add layer param same or not? (fiexed point view or difference stage view)
@GraphPair2VecEncoder.register("graph_embedding_net")
class GraphEmbeddingNet(GraphPair2VecEncoder):
    """
    `GraphEmbeddingNet` encodes 2 graphs with `Graph2GraphEncoder` seperately,
    then use 'Graph2VecEncoder' to project 2 graphs into the same representation space,
    then return a vector $[g1;g2;g1-g2;g1 \odot g2]$ for further classification
    """
    
    """ todo: will uncomment after test
    __slots__ = (
        "_input_dim",
        "_output_dim",
        "_convs",
        "num_layers"
    )
    """
    
    def __init__(
        self,
        num_layers: int,
        convs: Union[Graph2GraphEncoder, List[Graph2GraphEncoder]], # this should be the name of the convs
        pooler: Graph2VecEncoder, # graph pooler for mapping graph to embedding space
        **kwargs,
        #dropout: Union[float, List[float]] = 0.0,
    ) -> None:
        """
        Old Style(I think this is not ok since if convname is not the same, they may need different params)
        `GraphEmbeddingNet` constructor
        note that convs is str or List[str] (in Graph2GraphEncoder.list_availabels() )
        so that this constructor calls Graph2GraphEncoder.from_name(convs[i]) for constructing convolutions
        """
        super().__init__()
        # if given is not List[item], create List[item]
        # this method implicitly share params? 
        if not isinstance(convs, list):
            convs = [convs] * num_layers
            
        if len(convs) != num_layers:
            raise ConfigurationError(
                "len(convs) (%d) != num_layers (%d)" % (len(convs), num_layers)
            )
            
        self._convs = torch.nn.ModuleList(convs)
        self._output_dim = 4*convs[-1].out_channels # for vector pair comparison 
        self._input_dim = convs[0].in_channels
        self._pooler = pooler
        self.num_layers = num_layers
    
    @overrides
    def get_output_dim(self):
        return self._output_dim
    
    @overrides
    def get_input_dim(self):
        return self._input_dim

    def forward(
        self,
        x1: Dict[str, torch.Tensor],
        x2: Dict[str, torch.Tensor],
        g1: Dict[str, torch.Tensor],
        g2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        input:
            g1, g2 : Dict[str, torch.Tensor] sparse_adjacency batch
            n1, n2 : Dict[str, torch.Tensor] sparse_node_embedding batch
            e1, e2 : OptTensor sparse_edge_embedding batch
        """
        # node_tensor, node_batch_indices
        x1, b1 = x1["data"], x1["batch_indices"]
        x2, b2 = x2["data"], x2["batch_indices"]
        # edge_index, edge_type, edge_batch
        e1, t1, eb1 = g1['edge_index'], g1['edge_attr'], g1['batch_id']
        e2, t2, eb2 = g2['edge_index'], g2['edge_attr'], g2['batch_id']
        
        # apply Graph2Graph Encoders by module list
        for conv, in zip(
            self._convs
        ):
            x1 = conv(x=x1, edge_index=e1, edge_type=t1)
            x2 = conv(x=x2, edge_index=e2, edge_type=t2)
        
        # Graph Pooling
        v1 = self._pooler(x1, batch=b1)
        v2 = self._pooler(x2, batch=b2)
        # Shape: (batch_size, _out_put_dim)
        out = torch.cat([v1, v2, v1-v2, v1*v2], dim=1)

        return out
