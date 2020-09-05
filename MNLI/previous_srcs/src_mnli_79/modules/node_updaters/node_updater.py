from allennlp.common import Registrable
from typing import Union, List

import torch

from overrides import overrides

class NodeUpdater(torch.nn.Module, Registrable):
    """
    cross attention  for sparse batches 
    """
    pass


@NodeUpdater.register("gru", exist_ok=True)
class GRUNodeUpdater(NodeUpdater):
    """
    use previous node state as hidden,
    the next node state is GRU(message, hidden)
    message can be list, can use concat to aggregate
    """
    
    def __init__(self,
                 input_size: int,  # dim_encoder + dim_attention
                 hidden_size: int, # din_encoder
                 bias: bool = True,
                 aggr: str = "concat", # concat or sum
                ):
        super(GRUNodeUpdater, self).__init__()
        self._rnn = torch.nn.GRUCell(input_size, hidden_size, bias)
        
        assert(aggr in ["concat", "sum"])
        if aggr == "concat":
            self._aggr = torch.cat
        elif aggr == "sum":
            self._aggr = torch.sum
            
    def forward(self,
                message: Union[List[torch.Tensor], torch.Tensor],
                hidden: torch.Tensor,
               ):
        msg = self._aggr(message, dim=-1) # concat the representation dim (i.e. -1)
        return self._rnn(msg, hidden)