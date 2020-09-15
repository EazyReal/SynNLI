from allennlp.common import Registrable # if add more 

from overrides import overrides
from typing import Dict, Tuple, List
from src.tensor_op import sorted_dynamic_parition

import torch


class GraphPair2GraphPairEncoder(torch.nn.Module, Registrable):
    pass