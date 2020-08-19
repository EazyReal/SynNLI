import config
import utils

#import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#from src_gmn.sparse_adjacency_field import SparseAdjacencyField
from sparse_adjacency_field import SparseAdjacencyField

import itertools
from typing import Iterable, List, Dict, Tuple, Union
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ArrayField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer
from allennlp.data import Token, Vocabulary

# pytorch geometric
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data.data import Data as PyGeoData
from torch_geometric.data import DataLoader
## Stanza
import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline

logger = logging.getLogger(__name__)


# comment for development step
# @DatasetReader.register("nli-graph")
class NLI_Graph_Reader(DatasetReader):
    """
    Reads a file from a preprocessed NLI ataset.
    This data is formatted as jsonl, one json-formatted instance per line.
    The keys in the data are in config.
    along with a metadata field containing the tokenized strings of the premise and hypothesis.
    Registered as a `DatasetReader` with name "nli-graph".
    
    # Parameters
    parser: 
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": PretrainedTransformerMismatchedIndexer()}`)
    combine_input_fields : encode P and H at the same time with [CLS]P[SEP]H?
    Note: if want to use BERT like NLI method, see original reader on github "allennlp-models/esim..."
    
    # Notes
    We do not need to tokenize, input is already tokenized when using Stanza Pipeneline
    However, to get index, we need token_indexer! (note that "[ROOT]" will be unkown...)
    """

    def __init__(
        self,
        wordpiece_tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields : bool = None,
        **kwargs,
    ) -> None:
        #super().__init__(manual_distributed_sharding=True, **kwargs)
        super().__init__(**kwargs)
        self._wordpiece_tokenizer = wordpiece_tokenizer or PretrainedTransformerTokenizer(config.TRANSFORMER_NAME)
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerMismatchedIndexer(config.TRANSFORMER_NAME)}
        self._combine_input_fields = combine_input_fields or False

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as fo:
            example_iter = (json.loads(line) for line in fo)
            # we need gold label
            filtered_example_iter = (
                example for example in example_iter if example["gold_label"] != "-"
            )
            for example in filtered_example_iter:
                label = example["gold_label"]
                premise : List = example["sentence1"]
                hypothesis : List = example["sentence2"]
                yield self.graph_to_instance(premise, hypothesis, label)

    def graph_to_instance(
        self, 
        premise: List,
        hypothesis: List,
        label: int = None,
    ) -> Instance:
        """
        input: premise/hypothesis as List of Graph Infromation
        output: allennlp Instance
        
        convert List to PytorchGeo Data by utils.doc2graph
        node_attr : word tokens 
        edge_attr : edge labels
        edge_index : sparse edge
        
        ## tokenize intra_word_tokenize
        """

        fields: Dict[str, Field] = {}
            
        g_p: PyGeoData = utils.doc2graph(premise)
        g_h: PyGeoData = utils.doc2graph(hypothesis)
        tokens_p: List[Token] = [Token(w) for w in  g_p.node_attr]
        tokens_h: List[Token] = [Token(w) for w in  g_h.node_attr]  
            
        if self._combine_input_fields:
            # see the doc, the output should be indexed
            tokens = self._wordpiece_tokenizer.add_special_tokens(tokens_p, tokens_h)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            #premise_tokens = self._wordpiece_tokenizer.add_special_tokens(g_p.node_attr)
            #hypothesis_tokens = self._wordpiece_tokenizer.add_special_tokens(g_h.node_attr)
            fields["tokens_p"] = TextField(tokens_p, self._token_indexers)
            fields["tokens_h"] = TextField(tokens_h, self._token_indexers)
            fields["g_p"] = SparseAdjacencyField(graph=g_p,
                                                 sequence_field=fields["tokens_p"],
                                                 label_namespace = "edge_labels"
                                                )
            fields["g_h"] = SparseAdjacencyField(graph=g_h,
                                                 sequence_field=fields["tokens_h"],
                                                 label_namespace = "edge_labels"
                                                )
        
        # care do not use `if label` for label can be 0(in int)
        if label is not None:
            # already turn to index when preprocessing
            fields["label"] = LabelField(label, skip_indexing=True)

        return Instance(fields)
