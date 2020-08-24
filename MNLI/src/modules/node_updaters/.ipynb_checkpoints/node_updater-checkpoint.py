from allennlp.common import Registrable

import torch

from overrides import overrides

class NodeUpdater(torch.nn.Module, Registrable):
    """
    cross attention  for sparse batches 
    """
    pass


@NodeUpdater.register("gru")
class GRUNodeUpdater(NodeUpdater):
    pass