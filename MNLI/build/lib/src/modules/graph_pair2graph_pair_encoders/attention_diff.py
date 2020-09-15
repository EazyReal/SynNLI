from overrides import overrides
from typing import Dict, Tuple, List

import torch
from allennlp.modules import MatrixAttention

from src.tensor_op import sorted_dynamic_parition, dense2sparse, sparse2dense
from .graph_pair2graph_pair_encoder import GraphPair2GraphPairEncoder

# from src.nn.functional import vec_diff


@GraphPair2GraphPairEncoder.register("att_diff", exist_ok=True)
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
                 dim: int,
                ):
        super(AttentiveSumDiff, self).__init__()
        self._att = att
        self._dim_match = dim
    
    def get_output_dim():
        return self._dim_match
    
    def forward(self,
               x1: torch.Tensor,
               x2: torch.Tensor,
               b1: torch.Tensor,
               b2: torch.Tensor,
               return_attention: bool = False) -> Tuple[torch.Tensor]:
        # N is number of batches
        N: int = b1[-1]+1
        xs1: List[torch.Tensor] = sorted_dynamic_parition(x1, b1)
        xs2: List[torch.Tensor] = sorted_dynamic_parition(x2, b2)
        # return all nodes
        r1 = []
        r2 = []
        Ss = []
        # for each instance in batch
        for i in range(N):
            bx1 = xs1[i] # n*d
            bx2 = xs2[i] # m*d
            S = self._att(bx1.unsqueeze(0), bx2.unsqueeze(0)).squeeze(0) # n*m similarity matrix
            Ss += [S.detach().T]
            r1 += [torch.mm(S, bx2)] # n*m, m*d => n*d, content is from bx2
            r2 += [torch.mm(S.T, bx1)] # m*n, n*d => m*d, content is from bx1
        # concat to make sparse batch
        r1 = torch.cat(r1, dim=0) # 
        r2 = torch.cat(r2, dim=0) # 
        # calculate the difference (use simple substraction here)
        d1 = x1-r1
        d2 = x2-r2
        # return attention if required, TODO
        if return_attention is True:
            return d1, d2, Ss
        else:
            return d1, d2
    
    def __repr__(self):
        return f"({str(self._att)}, dim_matching_output={str(self._dim_match)})"
   