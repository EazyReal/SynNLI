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

# defualt choice for model embedding, can use config file later
transformer_embedder = PretrainedTransformerMismatchedEmbedder(
    model_name=config.TRANSFORMER_NAME,
    max_length=None, # concat if over max len (512 for BERT base)
    train_parameters=True,
    #last_layer_only=True, unsupported? why
    #gradient_checkpointing=None
)


@Model.register("simple_model")
class SynNLIModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TokenEmbedder,
                 pooler: Seq2VecEncoder
                 #gmn: GraphMatchingNetwork,
                ):
        """
        vocab : for edge_labels mainly
        embedder: text_token_ids => text_embedding_space
        gmn : GraphMatchingNetwork, take tokens, graph_adj pair to produce a single vector for cls
        cls : classifier
        """
        super().__init__(vocab)
        num_labels = vocab.get_vocab_size("labels") #3
        self.embedder = embedder or transformer_embedder
        #self.gmn = gmn
        #self.classifier = torch.nn.Linear(gmn.get_output_dim(), num_labels)
        self.pooler = pooler or BagOfEmbeddingsEncoder(768, averaged=True)
        self.classifier = torch.nn.Linear(768, num_labels)
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
        # inside or outside GMN?
        sparse_p = tensor_op.dense2sparse(embedded_p, tokens_p["tokens"]["mask"])
        sparse_h = tensor_op.dense2sparse(embedded_h, tokens_h["tokens"]["mask"])
        dense_p = tensor_op.sparse2dense(**sparse_p)
        dense_h = tensor_op.sparse2dense(**sparse_h)
        # Shape: (batch_size, classifier_in_dim)
        # cls_vector = self.gmn(sparse_p, sparse_h, g_p, g_h)
        pool_p = self.pooler(dense_p["data"], dense_p["mask"])
        pool_h = self.pooler(dense_p["data"], dense_p["mask"])
        #print(pool_p.size())
        cls_vector = (pool_p+pool_h)/2
        #print(cls_vector.size())
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