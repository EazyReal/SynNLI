from overrides import overrides
from typing import Optional, Dict, Iterable, List, Union

import torch_geometric
import torch

from allennlp.common import Registrable

from src.modules import (
    Graph2VecEncoder,
    GraphPair2VecEncoder,
    Graph2GraphEncoder,
    NodeUpdater,
    GraphPair2GraphPairEncoder
)
#from src.modules.attention import GraphPairAttention

# todo, figure out why this is bugged
@GraphPair2VecEncoder.register("graph_matching_net", exist_ok=True)
class GraphMatchingNet(GraphPair2VecEncoder):
    """
    `GraphMatchingNet` differs from `GraphEmbeddingNet` with an extra cross graph attention.
    
    In each layer,
    `Graph2GraphEncoder` encodes each graph seperately,
    `GraphMatchingNet` matches two graph by a similarity function and produces the soft-aligned version of the other graph for each graph,
    `NodeUpdater` than is used to update the current node repr by the msgs.
    
    After L layers of encoding and matching,'Graph2VecEncoder' is applied to project 2 graphs into the same representation space,
    then return a vector $[g1;g2;g1-g2;g1 \odot g2]$ for further classification
    """
    
    def __init__(
        self,
        num_layers: int,
        convs: Graph2GraphEncoder, 
        atts: GraphPair2GraphPairEncoder,
        updater: NodeUpdater,  
        pooler: Graph2VecEncoder, 
    ) -> None:
        """
        `GraphMatchingNet` constructor
        """
        super(GraphMatchingNet, self).__init__()
        # if given is not List[item], create List[item]
        if not isinstance(convs, list):
            convs = [convs] * num_layers # share param
        if not isinstance(atts, list):
            atts = [atts] * num_layers
            
        if len(convs) != num_layers:
            raise ConfigurationError(
                "len(convs) (%d) != num_layers (%d)" % (len(convs), num_layers)
            )
        if len(atts) != num_layers:
            raise ConfigurationError(
                "len(atts) (%d) != num_layers (%d)" % (len(atts), num_layers)
            )
        
            
        self._convs = torch.nn.ModuleList(convs)
        self._atts = torch.nn.ModuleList(atts)
        self._updater = updater
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
        for i in range(self.num_layers):
            # calculate message (n_nodes, dim_encoder)
            x1_msg = self._convs[i](x=x1, edge_index=e1, edge_type=t1)
            x2_msg = self._convs[i](x=x2, edge_index=e2, edge_type=t2)
            # calculate matching (n_nodes, dim_matching)
            x1_match, x2_match = self._atts[i](x1, x2, b1, b2)
            # update (n_nodes, dim_encoder)
            x1 = self._updater([x1_msg, x1_match], x1)
            x2 = self._updater([x2_msg, x2_match], x2)
        
        # Graph Pooling => (batch_size, _input_dim)
        v1 = self._pooler(x1, batch=b1)
        v2 = self._pooler(x2, batch=b2)
        # Shape: (batch_size, _out_put_dim)
        out = torch.cat([v1, v2, v1-v2, v1*v2], dim=1)

        return out
