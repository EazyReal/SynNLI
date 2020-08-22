## util
import os
import logging
from argparse import ArgumentParser
from tqdm import tqdm as tqdm
import pickle
import json
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, Union
from pathlib import Path
## Allennlp
from allennlp.predictors import Predictor
## Stanza
import stanza
from stanza.models.common.doc import Document as StanzaDocument
from stanza.pipeline.core import Pipeline as StanzaPipeline


@Predictor.register("stanza_dependency")
class StanzaDependencyPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the [`BasicClassifier`](../models/basic_classifier.md) model.
    Registered as a `Predictor` with name "text_classifier".
    """
    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        sentence = json_dict["sentence"]
        reader_has_tokenizer = (
            getattr(self._dataset_reader, "tokenizer", None) is not None
            or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = SpacyTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
    
    

    
    
    
@Predictor.register("biaffine_dependency_parser", exist_ok=True)
class BiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the [`BiaffineDependencyParser`](../models/biaffine_dependency_parser.md) model.
    """

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        # Parameters
        sentence The sentence to parse.
        # Returns
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        """
        spacy_tokens = self._tokenizer.tokenize(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        if self._dataset_reader.use_language_specific_pos:  # type: ignore
            # fine-grained part of speech
            pos_tags = [token.tag_ for token in spacy_tokens]
        else:
            # coarse-grained part of speech (Universal Depdendencies format)
            pos_tags = [token.pos_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        words = outputs["words"]
        pos = outputs["pos"]
        heads = outputs["predicted_heads"]
        tags = outputs["predicted_dependencies"]
        outputs["hierplane_tree"] = self._build_hierplane_tree(words, heads, tags, pos)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            words = output["words"]
            pos = output["pos"]
            heads = output["predicted_heads"]
            tags = output["predicted_dependencies"]
            output["hierplane_tree"] = self._build_hierplane_tree(words, heads, tags, pos)
        return sanitize(outputs)