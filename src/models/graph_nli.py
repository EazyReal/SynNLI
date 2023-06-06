"""
author = YT Lin
graph_nli.py is for the models that requires edge embedding,
can use Passthrough provided in allennlp if not required
"""
# external
import json
from typing import Dict, Iterable, List
from copy import deepcopy

# allennlp
import numpy as np
import torch

from allennlp.data import DatasetReader, DataLoader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
#from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Entropy

from allennlp.modules import TextFieldEmbedder, TokenEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
# BasicTextFieldEmbedder is Dict[str, TokenEmbedder]

from allennlp.nn import (
    InitializerApplicator, 
    RegularizerApplicator,
)

# FeedForward(124, 2, [64, 32], torch.nn.ReLU(), 0.2)

#self import 
# import src.config as config
from src.data_git.sparse_adjacency_field import SparseAdjacencyField, SparseAdjacencyFieldTensors
# for batch transform
import src.tensor_op as tensor_op
from src.modules.graph_pair2vec_encoders import GraphPair2VecEncoder

@Model.register("graph-nli-ee", exist_ok=True)
class GraphNLIModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 edge_embedder: TokenEmbedder,
                 projector: FeedForward,
                 encoder: GraphPair2VecEncoder,
                 classifier: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator()
                ):
        """
        vocab : for edge_labels mainly
        embedder: text_token_ids => text_embedding_space
        gmn : GraphMatchingNetwork, take tokens, graph_adj pair to produce a single vector for cls
        cls : classifier
        """
        super().__init__(vocab)
        num_labels = vocab.get_vocab_size("labels") #3
        num_relations = vocab.get_vocab_size("relations") #20?
        self.embedder = embedder 
        self.edge_embedder = edge_embedder
        self.projector = projector
        self.encoder = encoder
        self.classifier = classifier
        self.accuracy = CategoricalAccuracy()
        self.entropy = Entropy()
        initializer(self) # init parameters, would not modify model slots
        # check dimension match if required
        return
        
    def forward(self,
            tokens_p: TextFieldTensors,
            tokens_h: TextFieldTensors,
            g_p: SparseAdjacencyFieldTensors,
            g_h: SparseAdjacencyFieldTensors,
            label: torch.Tensor = None,
            return_attention: bool = False,
        ) -> Dict[str, torch.Tensor]:
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
        # embedder will take out the desired entry, for ex: ["tokens"]
        embedded_p = self.embedder(tokens_p) 
        embedded_h = self.embedder(tokens_h)
        # Shape: (num_batch_edges, edge_embedding_dim)
        # can use pass through if type is the information required
        # this is general for any kind of edge representation to GP2Vencoder
        # https://docs.allennlp.org/master/api/modules/token_embedders/embedding/
        # inplace will change input!!!
        g_p_embedded = deepcopy(g_p)
        g_h_embedded = deepcopy(g_h)
        g_p_embedded["edge_attr"] = self.edge_embedder(g_p["edge_attr"])
        g_h_embedded["edge_attr"] = self.edge_embedder(g_h["edge_attr"])
        assert(not torch.any(torch.isnan(embedded_p)))
        assert(not torch.any(torch.isnan(embedded_h)))
        # Shape: (batch_size, num_tokens, projected_dim)
        embedded_p = self.projector(embedded_p)
        embedded_h = self.projector(embedded_h)
        assert(not torch.any(torch.isnan(embedded_p)))
        assert(not torch.any(torch.isnan(embedded_h)))
        # Shape:
        # node_attr : (num_tokens, embedding_dim)
        # batch_id : (num_tokens)
        sparse_p = tensor_op.dense2sparse(embedded_p, tokens_p["tokens"]["mask"]) #need to overload indexer for this
        sparse_h = tensor_op.dense2sparse(embedded_h, tokens_h["tokens"]["mask"])
        assert(not torch.any(torch.isnan(sparse_p["data"])))
        assert(not torch.any(torch.isnan(sparse_h["data"])))
        # Shape: (batch_size, classifier_in_dim)
        if return_attention:
            cls_vector, attention_dict = self.encoder(sparse_p,
                                                      sparse_h,
                                                      g_p_embedded,
                                                      g_h_embedded,
                                                      return_attention=return_attention)
        else:
            cls_vector = self.encoder(sparse_p,
                                      sparse_h,
                                      g_p_embedded,
                                      g_h_embedded,
                                      return_attention=return_attention)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(cls_vector)
        assert(not torch.any(torch.isnan(cls_vector)))
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Shape: TensorDict
        output = {'probs': probs}
        if label is not None:
            #print(logits.size(), label.size())
            self.accuracy(logits, label)
            # the two value can be kind of different for numerical isse IMO
            self.entropy(logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        if return_attention is True:
            output['attentions'] = attention_dict
        return output
    
    # add entropy support for visualizing loss
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy.get_metric(reset),
            "entropy": self.entropy.get_metric(reset)["entropy"].item(),
        }
    
    # for interpretation
    def make_output_human_readable(
        self,
        output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        will be used by forward_on_instances() of Model
        which is called by Predictors
        """
        # Take the logits from the forward pass, and compute the label
        # IDs for maximum values
        probs = output_dict['probs'].cpu().data.numpy()
        predicted_id = np.argmax(probs, axis=-1)
        # Convert these IDs back to label strings using vocab
        output_dict['predicted_label'] = [
            self.vocab.get_token_from_index(x, namespace='labels')
            for x in predicted_id
        ]
        # attention is in output_dict already
        # original tokens/glod_label/problemID?
        return output_dict