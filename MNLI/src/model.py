"""
author = YT Lin
"""
# external
import json
from typing import Dict, Iterable, List

# allennlp
import torch
from allennlp.data import DatasetReader, DataLoader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.modules.token_embedders import TokenEmbedder, PretrainedTransformerMismatchedEmbedder

#self import 
import config
from sparse_adjacency_field import SparseAdjacencyField, SparseAdjacencyFieldTensors
#from gmn import GraphMatchingNetwork
import tensor_op # for batch transform
from graph_pair_to_vec_encoder import GraphPair2VecEncoder

# defualt choice for model embedding, can use config file later

@Model.register("simple_model")
class SynNLIModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TokenEmbedder,
                 graph_pair2vec_encoder: GraphPair2VecEncoder,
                ):
        """
        vocab : for edge_labels mainly
        embedder: text_token_ids => text_embedding_space
        gmn : GraphMatchingNetwork, take tokens, graph_adj pair to produce a single vector for cls
        cls : classifier
        """
        super().__init__(vocab)
        num_labels = vocab.get_vocab_size("labels") #3
        self.embedder = embedder 
        self.graph_pair2vec_encoder = graph_pair2vec_encoder
        self.classifier = torch.nn.Linear(gmn.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        return
        
    def forward(self,
            tokens_p: TextFieldTensors,
            tokens_h: TextFieldTensors,
            g_p: SparseAdjacencyFieldTensors,
            g_h: SparseAdjacencyFieldTensors,
            label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        GMN for NLI
        let B be batch size
        let N be p length
        let M be h length
        
        input :
            tokens_p in shape [*] (B, N)
            g_p["edge_index"]
        ouput : tensor dict
        """
        # Shape: (batch_size, num_tokens, embedding_dim)
        #print(tokens_p["tokens"])
        embedded_p = self.embedder(**tokens_p["tokens"])
        embedded_h = self.embedder(**tokens_h["tokens"])
        # Shape:
        # node_attr : (num_tokens, embedding_dim)
        # batch_id : (num_tokens)
        sparse_p = tensor_op.dense2sparse(embedded_p, tokens_p["tokens"]["mask"])
        sparse_h = tensor_op.dense2sparse(embedded_h, tokens_h["tokens"]["mask"])
        # Shape: (batch_size, classifier_in_dim)
        cls_vector = self.graph_pair2vec_encoder(sparse_p, sparse_h, g_p, g_h)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(cls_vector)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=0)
        # Shape: TensorDict
        output = {'probs': probs}
        if label is not None:
            #print(logits.size(), label.size())
            self.accuracy(logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}