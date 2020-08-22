import itertools
from typing import Dict, Optional
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer, PretrainedTransformerMismatchedIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("nli-graph")
class NLI_Graph_Reader(DatasetReader):
    """
    Reads a file from a NLI ataset.
    This data is formatted as jsonl, one json-formatted instance per line.
    The keys in the data are in config.
    along with a metadata field containing the tokenized strings of the premise and hypothesis.
    Registered as a `DatasetReader` with name "nli-graph".
    
    # Parameters
    tokenizer : `Tokenizer`, optional (default=`None`)
        We use this `Tokenizer` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": PretrainedTransformerMismatchedIndexer()}`)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    combine_input_fields : None, if want to use BERT like NLI method, see original reader on github "allennlp-models/esim..."
    
    # Notes
    We do not need to tokenize, input is already tokenized when using Stanza Pipeneline
    However, to get index, we need token_indexer!
    (note that "[ROOT]" will be unkown...)
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)
        self._tokenizer = tokenizer #or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as fo:
            example_iter = (json.loads(line) for line in fo)
            filtered_example_iter = (
                example for example in example_iter if example["gold_label"] != "-"
            )
            for example in filtered_example_iter:
                label = example["gold_label"]
                premise = example["sentence1"]
                hypothesis = example["sentence2"]
                yield self.text_to_instance(premise, hypothesis, label)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        premise = self._tokenizer.tokenize(premise)
        hypothesis = self._tokenizer.tokenize(hypothesis)

        if self._combine_input_fields:
            tokens = self._tokenizer.add_special_tokens(premise, hypothesis)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            premise_tokens = self._tokenizer.add_special_tokens(premise)
            hypothesis_tokens = self._tokenizer.add_special_tokens(hypothesis)
            fields["premise"] = TextField(premise_tokens, self._token_indexers)
            fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)

            metadata = {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            }
            fields["metadata"] = MetadataField(metadata)

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)