# to be solved "[ROOT]" is not special for transformer
# can use "[SEP]" ?, since we are not using [SEP], and [SEP] is not meaningful itself without "type_id"?
 
import src.data_git.reader_config as config
import src.data_git.utils as utils
from src.data_git.sparse_adjacency_field import SparseAdjacencyField

#import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#from src_gmn.sparse_adjacency_field import SparseAdjacencyField

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
from stanza.models.common.doc import Document as StanzaDoc
from stanza.pipeline.core import Pipeline as StanzaPipeline

logger = logging.getLogger(__name__)

default_fields = ["sentence1", "sentence2", "gold_label"]

"""
Caveat:
    the file is current dependent on reader_config.py
    "[ROOT]" is not special token should be fixed
    combined_input_field is not implemented yet
"""

# comment if in development step for ipython notebook import
# or use @DatasetReader.register("nli-graph", exist_ok=True)
@DatasetReader.register("nli-graph", exist_ok=True)
class NLIGraphReader(DatasetReader):
    """
    Reads a file from a preprocessed/raw NLI dataset.
    the input type can be determined by __init__ parameter
    This data is formatted as jsonl, one json-formatted instance per line.
    {
        "gold_label": {0: contradiction, 1: neutral, 2: entailment} or RawLabel
        "sentence1": StanzaDoc in List form or RawText
        "sentence2": StanzaDoc in List form or RawText
    }
    along with a metadata field containing the tokenized strings of the premise and hypothesis.
    Registered as a `DatasetReader` with name "nli-graph".
    
    # Parameter:
        wordpiece_tokenizer: Tokenizer, optional (default=`{"tokens": PretrainedTransformerTokenizer(config.TRANSFORMER_NAME)}`)
            tokenize token into smaller pieces
        token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": PretrainedTransformerMismatchedIndexer(config.TRANSFORMER_NAME)}`)
            index token
        combine_input_fields : `bool`, optional(default=False)
            whether to encode P and H at the same time with [CLS]P[SEP]H[SEP]
            Note: if want to use BERT like NLI method, see original reader on github "allennlp-models/esim..."
        input_parsed: `bool`, optional (default=`True`)
            if the input is rawtext or parsed stanza doc
            if not, the parsing part is in text2instance function
            and the cache option for readsetreader should be open(cache_dir)
        parser: `StanzaPipeline`, optional
            if input_parsed is provided False, provide parser
        cache_directory : `str`, optional (default=`None`)
            this is a param for parent class `DatasetReader`
            from  `https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py`
            If given, we will use this directory to store a cache of already-processed `Instances` in
            every file passed to :func:`read`, serialized (by default, though you can override this) as
            one string-formatted `Instance` per line.  If the cache file for a given `file_path` exists,
            we read the `Instances` from the cache instead of re-processing the data (using
            :func:`_instances_from_cache_file`).  If the cache file does _not_ exist, we will _create_
            it on our first pass through the data (using :func:`_instances_to_cache_file`).
            !!! NOTE
                It is the _caller's_ responsibility to make sure that this directory is
                unique for any combination of code and parameters that you use.  That is, if you pass a
                directory here, we will use any existing cache files in that directory _regardless of the
                parameters you set for this DatasetReader!_
    
    # Caveat:
        the file is current dependent on reader_config.py
        "[ROOT]" is not special token should be fixed
        combined_input_field is not implemented yet
    """

    def __init__(
        self,
        wordpiece_tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: bool = None,
        input_parsed: bool = None,
        parser: StanzaPipeline = None,
        input_fields: List = None,
        **kwargs,
    ) -> None:
        #super().__init__(manual_distributed_sharding=True, **kwargs)
        super(NLIGraphReader, self).__init__(**kwargs)
        self._wordpiece_tokenizer = wordpiece_tokenizer # used when want to combine input field if use bert
        self._token_indexers = token_indexers 
        self._combine_input_fields = combine_input_fields or False
        self._input_parsed = input_parsed if (input_parsed is not None) else True 
        self._parser = parser or None
        self.f = input_fields or default_fields #remove dependency to config

    @overrides
    def _read(self, file_path: str):
        """
        Reads a file, yield instances
        can take raw or parsed depends on __init__ param (input_parsed)
        the parsing part is in text2instance function
        besure to open cache in config file to store instance if using raw text to speed up coming experiments 
        """
        file_path = cached_path(file_path)
        with open(file_path, "r") as fo:
            example_iter = (json.loads(line) for line in fo.readlines())
            # we need gold label
            filtered_example_iter = (
                example for example in example_iter if example[self.f[2]] != "-"
            )
            for example in filtered_example_iter:
                # we want label to be in string here, use vocab to label
                label = example[self.f[2]]
                if(isinstance(label, int)):
                    label = config.id_to_label[label]
                premise : Union[StanzaDoc, List, str] = example[self.f[0]]
                hypothesis :  Union[StanzaDoc, List, str] = example[self.f[1]]
                yield self.text_to_instance(premise, hypothesis, label)
                
    
    #@classmethod
    def instance2sent(self, instance):
        """
        return string for instance
        todo: add graph visualization
        """
        return {
            "premise": " ".join([token.text for token in instance.fields["tokens_p"].tokens]),
            "hypothesis": " ".join([token.text for token in instance.fields["tokens_h"].tokens]),
        }
        
    
    @overrides
    def text_to_instance(
        self, 
        premise: Union[StanzaDoc, List, str],
        hypothesis: Union[StanzaDoc, List, str],
        gold_label: Union[int, str]= None,
    ) -> Instance:
        """
        input: premise/hypothesis as List of Graph Infromation
        output: allennlp Instance
        
        convert List to PytorchGeo Data by utils.doc2graph
        node_attr : word tokens 
        edge_attr : edge labels
        edge_index : sparse edge
        """

        fields: Dict[str, Field] = {}
        
        if isinstance(premise, str): # or use ` not self._input_paresed`
            premise : StanzaDoc = self._parser(premise)
            hypothesis : StanzaDoc = self._parser(hypothesis)
        g_p: PyGeoData = utils.doc2graph(premise)
        g_h: PyGeoData = utils.doc2graph(hypothesis)
        def stanza_word2allennlp_token(w):
            t = Token(
                text = w.text,
                lemma_ = w.lemma,
                pos_ = w.pos,
                dep_ =  w.deprel,
            )
            return t 
        tokens_p: List[Token] = [stanza_word2allennlp_token(w) for w in  g_p.node_attr]
        tokens_h: List[Token] = [stanza_word2allennlp_token(w) for w in  g_h.node_attr]
            
        if self._combine_input_fields:
            # see the doc, the output should be indexed
            raise NotImplementedError
            tokens = self._wordpiece_tokenizer.add_special_tokens(tokens_p, tokens_h)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            #premise_tokens = self._wordpiece_tokenizer.add_special_tokens(g_p.node_attr)
            #hypothesis_tokens = self._wordpiece_tokenizer.add_special_tokens(g_h.node_attr)
            fields["tokens_p"] = TextField(tokens_p, self._token_indexers) # defualt = {tokens: tokens}?
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
        if gold_label is not None:
            fields["label"] = LabelField(gold_label, skip_indexing=False, label_namespace="labels")

        return Instance(fields)