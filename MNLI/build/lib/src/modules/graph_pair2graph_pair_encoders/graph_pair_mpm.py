from overrides import overrides
from typing import Dict, Tuple, List

import torch

from allennlp.modules import Attention
from allennlp.modules.bimpm_matching  import BiMpmMatching

from src.tensor_op import sorted_dynamic_parition, dense2sparse, sparse2dense
from .graph_pair2graph_pair_encoder import GraphPair2GraphPairEncoder


@GraphPair2GraphPairEncoder.register("bimpm", exist_ok=True)
class GraphPairMPM(GraphPair2GraphPairEncoder):
    """
    BiMPM for Sparse Batch By Dense, Sparse Convertion
    """
    def __init__(self,
                 bimpm: BiMpmMatching,
                ):
        super(GraphPairMPM, self).__init__()
        self._bimpm = bimpm
        self._dim_match = bimpm.get_output_dim()
        
    def get_output_dim():
        return self._dim_match
    
    def forward(self,
                x1: torch.Tensor,x2: torch.Tensor,
                b1: torch.Tensor,
                b2: torch.Tensor,
                return_attention: bool = False,
    ) -> Tuple[torch.Tensor]:
        x1 = sparse2dense(x1, b1)
        x2 = sparse2dense(x2, b2)
        x1, m1 = x1["data"], x1["mask"]
        x2, m2 = x2["data"], x2["mask"]
        x1, x2 = self._bimpm(x1, m1, x2, m2) 
        x1 = torch.cat(x1, dim=-1)
        x2 = torch.cat(x2, dim=-1)
        # concat different view (max/mean/attentive/attentivemax)
        # can try use attentive max + attentive only
        x1 = dense2sparse(x1, m1)
        x2 = dense2sparse(x2, m2)
        if return_attention:
            return x1["data"], x2["data"], None
        else:
            return x1["data"], x2["data"]
    
    def __repr__(self):
        return f"({str(self._bimpm)}, dim_matching_output={str(self._dim_match)})"
    
    
"""
BiMPM
    hidden_dim: int = 100,
    num_perspectives: int = 20,
    share_weights_between_directions: bool = True,
    is_forward: bool = None,
    with_full_match: bool = True,
    with_maxpool_match: bool = True,
    with_attentive_match: bool = True,
    with_max_attentive_match: bool = True,
"""

"""
def forward(self,
           x1: torch.Tensor,
           x2: torch.Tensor,
           b1: torch.Tensor,
           b2: torch.Tensor) -> Tuple[torch.Tensor]:
    # N is number of batches
    N: int = b1[-1]+1
    x1: List[torch.Tensor] = sorted_dynamic_parition(x1, b1)
    x2: List[torch.Tensor] = sorted_dynamic_parition(x2, b2)
    # return all nodes
    r1 = []
    r2 = []
    # for each batch
    for i in range(N):
        bx1 = xs1[i] # n*d
        bx2 = xs2[i] # m*d
        S = self._att(bx1, bx2) # n*m
        r1 += [bx1]
        r2 += []

    r1 = torch.cat(r1, dim=0)
    r2 = torch.cat(r2, dim=0)

    return r1, r2
"""


