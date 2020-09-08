from overrides import overrides
from typing import Optional, Dict, Iterable, List, Union

import torch_geometric
import torch

from allennlp.common import Registrable

from src.modules.graph2graph_encoders import  Graph2GraphEncoder
from src.modules.graph2vec_encoders import Graph2VecEncoder
from src.modules.graph_pair2graph_pair_encoders import GraphPair2GraphPairEncoder 

from src.modules.graph_pair2vec_encoders.graph_pair2vec_encoder import GraphPair2VecEncoder 

@GraphPair2VecEncoder.register("graph_embedding_net", exist_ok=True)
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
        convs: Graph2GraphEncoder, # todo: list version
        pooler: Graph2VecEncoder, # graph pooler for mapping graph to embedding space
        #**kwargs, todo: will, 
        #dropout: Union[float, List[float]] = 0.0,
    ) -> None:
        """
        Old Style(I think this is not ok since if convname is not the same, they may need different params)
        `GraphEmbeddingNet` constructor
        note that convs is str or List[str] (in Graph2GraphEncoder.list_availabels() )
        so that this constructor calls Graph2GraphEncoder.from_name(convs[i]) for constructing convolutions
        """
        super(GraphEmbeddingNet, self).__init__()
        # if given is not List[item], create List[item]
        # this method implicitly share params? 
        if not isinstance(convs, list):
            convs = [convs] * num_layers
            
        if len(convs) != num_layers:
            raise ConfigurationError(
                "len(convs) (%d) != num_layers (%d)" % (len(convs), num_layers)
            )
            
        self._convs = torch.nn.ModuleList(convs)
        self._output_dim = 4*convs[-1].get_output_dim() # for vector pair comparison 
        self._input_dim = convs[0].get_input_dim()
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
        return_attention: bool=False,
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
        # return attention
        if return_attention:
            atts = {}
        
        # Graph Pooling => (batch_size, _input_dim)
        if return_attention:
            v1, pooler_att1 = self._pooler(x1, batch=b1, return_attention=return_attention)
            v2, pooler_att2 = self._pooler(x2, batch=b2, return_attention=return_attention)
            atts["pooler1"] = pooler_att1
            atts["pooler2"] = pooler_att2
        else:
            v1 = self._pooler(x1, batch=b1, return_attention=return_attention)
            v2 = self._pooler(x2, batch=b2, return_attention=return_attention)
        # Shape: (batch_size, _out_put_dim)
        out = torch.cat([v1, v2, v1-v2, v1*v2], dim=1)
        
        if return_attention:
            return out, atts
        else:
            return out
