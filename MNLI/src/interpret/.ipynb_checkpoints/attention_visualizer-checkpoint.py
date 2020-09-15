import logging
from typing import Any, Dict, Union
from pathlib import Path

from allennlp.data import Batch, Instance, DatasetReader
from allennlp.common import Registrable
from allennlp.models import Model
from src.interpret import (
    show_matrix_attention,
    show_sequence_attention,
)
from src.data_git.utils import (
    text2graph,
    draw
)

def instance2json(instance: Instance):
    text_p = [token.text for token in  instance.fields["tokens_p"].tokens]
    text_h = [token.text for token in  instance.fields["tokens_h"].tokens]
    gold_label = instance.fields["label"].label
    return {
        "sentence1" : text_p,
        "sentence2" : text_h,
        "gold_label" : gold_label,
    }

class AttentionVisualizer(Registrable):
    """
    visualizer functions
    this tool can be used for error analysis
    """
    # init 
    def __init__(self, model: Model, reader: DatasetReader):
        """
        give a model and reader
        reader with nlp is required for probing
        reader with not be used if provided instance
        """
        self._model = model
        self._reader = reader
        return
    
    # visualize a json dictionary
    def visualize_json(self, json_dict: Dict[str, Any], serialization_dir: Union[str, Path]=None):
        """
        require reader with nlp
        """
        instance = self._reader.text_to_instance(
            premise =  json_dict["sentence1"],
            hypothesis = json_dict["sentence2"],
            gold_label = json_dict["gold_label"] if "gold_label" in json_dict.keys() else None,
        )
        self.visualize_instance(instance, serialization_dir)
        return
        
    # visualize an instance
    def visualize_instance(self, instance: Instance, serialization_dir: Union[str, Path]=None):
        """
        main function of this visualizer
        usage: take an instance and visualize it
        _model have to support "return_attention=True" kwarg
        _model have to have the correct vocab in it
        """
        logger = logging.getLogger(__name__)
        # indexing with model
        instance.index_fields(self._model.vocab)
        # get tokens from instance
        json_dict = instance2json(instance)
        tokens_p = json_dict["sentence1" ]
        tokens_h = json_dict["sentence2" ]
        gold_label = json_dict["gold_label"]
        # get predictions and attentions
        batch = Batch([instance])
        batch_tensor = batch.as_tensor_dict()
        #print(batch.as_tensor_dict())
        ret = self._model.forward(**batch_tensor, return_attention=True)
        ret = self._model.make_output_human_readable(ret)
        pooler_p = ret["attentions"]["pooler1"][0]
        pooler_h = ret["attentions"]["pooler2"][0]
        logger.setLevel(logging.DEBUG)
        logger.info(f"tokens_p are {tokens_p}")
        logger.info(f"tokens_h are {tokens_h}")
        logger.info(f"the predicted label is {ret['predicted_label']}")
        logger.info(f"the gold label is {gold_label}")
        # hope to show_sequence_attention(strlist, att, msg=None)
        if serialization_dir is not None:
            show_sequence_attention(tokens_p, pooler_p, serialization_dir + "pooler_p.png")
            show_sequence_attention(tokens_h, pooler_h, serialization_dir + "pooler_h.png")
            num_layers = 3
            for i in range(num_layers):
                logger.info(f"matrix attention at layer {i}")
                show_matrix_attention(tokens_p, tokens_h, ret["attentions"][f"matching{i}"][0], serialization_dir + f"matching{i}.png")
        else:
            show_sequence_attention(tokens_p, pooler_p)
            show_sequence_attention(tokens_h, pooler_h)
            num_layers = 3
            for i in range(num_layers):
                logger.info(f"matrix attention at layer {i}")
                show_matrix_attention(tokens_p, tokens_h, ret["attentions"][f"matching{i}"][0])
            
    
    def visualize_batch_and_save():
        pass