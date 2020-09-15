from overrides import overrides
from typing import Dict, Tuple, List

import torch
from allennlp.modules import MatrixAttention

from src.tensor_op import sorted_dynamic_parition, dense2sparse, sparse2dense
from .graph_pair2graph_pair_encoder import GraphPair2GraphPairEncoder

# from src.nn.functional import vec_diff


@GraphPair2GraphPairEncoder.register("att-diff")
class AttentiveSumDiff(GraphPair2GraphPairEncoder):
    """
    `AttentiveSumDiff` is a `GraphPair2GraphPairEncoder` that produce compmarison between graph pair by,
    Attentive Pooling with similarity function from another graph (AttentiveSum) + 
    Comparison (Diff),
    A naive implementation is from `https://arxiv.org/pdf/1904.12787.pdf`,
    where consine similarity and substraction is used
    
    dynamic_partition + cross attention + substraction
    dim_match = dim_encode
    # can add a projection before output, but this should be left for the NodeUpdater part
    """
    def __init__(self,
                 att: MatrixAttention,
                ):
        super(AttentiveSumDiff, self).__init__()
        self._att = att
    
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
            r1 += torch.mm(S, bx2) # n*m, m*d => n*d, content is from bx2
            r2 += torch.mm(S.T, bx1) # m*n, n*d => m*d, content is from bx2
            
        r1 = torch.cat(r1, dim=0) # cat for 
        r2 = torch.cat(r2, dim=0) # cat to

        return r1, r2
    
    def __repr__(self):
        return f"({str(self._bimpm)}, dim_matching_output={str(self._dim_match)})"
   