from typing import Any, Dict
from overrides import overrides

from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import (
    DatasetReader,
    Instance
)

@Predictor.register("parsed_graph_predictor", exist_ok=True)
class GraphNLIPredictor(Predictor):
    """
    allennlp predictor needs you to overide _json_to_instance
    this predictor takes "parsed" data
    
    Work Flow of Model and Predictor:
    read jsonl file
    self._json_tol_instance
    self._model.forward_on_instances
        self._model.make_output_human_readable
    return result
    """
    
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 frozen: bool = True
                ) -> None:
        if frozen:
            model.eval()
        self._model = model
        self._dataset_reader = dataset_reader
        self.cuda_device = next(self._model.named_parameters())[1].get_device()
        
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Converts a JSON object into an [`Instance`]
        and a `JsonDict` of information which the `Predictor` should pass through,
        such as tokenised inputs.
        
        dependenct on nlp
        """
        instance = self._dataset_reader.text_to_instance(
            premise =  json_dict["sentence1"],
            hypothesis = json_dict["sentence2"],
            gold_label = json_dict["gold_label"] if "gold_label" in json_dict.keys() else None,
        ) 
        return instance