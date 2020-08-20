## util
import os
import logging
from argparse import ArgumentParser
from tqdm import tqdm_notebook as tqdmnb
from tqdm import tqdm as tqdm
import pickle
import json 
import jsonlines as jsonl
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, Union
from pathlib import Path
## graph
import networkx as nx
import matplotlib.pyplot as plt
## nn
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data.data import Data
from torch_geometric.data import DataLoader
## Stanza
import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
## allennlp model
from allennlp_models.structured_prediction.predictors.srl import SemanticRoleLabelerPredictor
from allennlp_models.structured_prediction.predictors.biaffine_dependency_parser import BiaffineDependencyParserPredictor
from allennlp.predictors.predictor import Predictor #

## self
import config

###########################################################################################

# alias
p = config.pf
h = config.hf
l = config.lf

def g2sent(g : Data):
    return " ".join(g.node_attr)

def draw(data : Data, node_size=1000, font_size=12, save_img_file=None):
    """
    input: (torch_geometric.data.data.Data, path or string)
    effect: show and save graph data, with graphviz layout visualization
    """
    G = to_networkx(data)
    pos = nx.nx_pydot.graphviz_layout(G)
    if(data.edge_attr != None):
        edge_labels = {(u,v):lab for u,v,lab in data.edge_attr}
    if(data.node_attr != None):
        node_labels = dict(zip(G.nodes, data.node_attr))
    nx.draw(G, pos=pos, nodecolor='r', edge_color='b', node_size=node_size, with_labels=False)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=font_size)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=font_size)
    print(G.nodes)
    print(G.edges)
    if save_img_file != None:
        plt.savefig(save_img_file)
    plt.show()
    return

######################################################################################################
# Stanza   #
######################################################################################################

def text2graph(text : str, nlp : Pipeline):
    """
    text2doc by Stanza
    doc2graph by utils.doc2graph 
    """
    return doc2graph(nlp(text))
    
def doc2graph(doc : Union[Document, List]) -> Data:
    """
    input Stanza Document : doc
    output PytorchGeoData : G
    G = {
     x: id tensor
     edge_idx : edges size = (2, l-1)
     edge_attr: (u, v, edge_type in str)
     node_attr: text
    }
    """
    if isinstance(doc, list): #convert to Doc first if is in dict form ([[dict]])
        doc = Document(doc)
    # add root token for each sentences
    n = doc.num_tokens+len(doc.sentences)
    e = [[],[]]
    edge_info = []
    node_info = []
    prev_token_sum = 0
    prev_root_id = 0
    cur_root_id = 0
    # get original dependency
    for idx, sent in enumerate(doc.sentences):
        # node info by index(add root at the beginning of every sentence)
        cur_root_id = len(node_info)
        node_info.append("[ROOT]")
        for token in sent.tokens:
            node_info.append(token.to_dict()[0]['text'])
        # edge info by index of u in edge (u,v)
        for dep in sent.dependencies:
            id1 = prev_token_sum + int(dep[0].to_dict()["id"])
            id2 = prev_token_sum + int(dep[2].to_dict()["id"])
            e[0].append(id1)
            e[1].append(id2)
            edge_info.append(dep[1])
        prev_token_sum += len(sent.tokens)+1
        # add links between sentence roots
        if(cur_root_id != 0):
            id1 = prev_root_id
            id2 = cur_root_id
            e[0].append(id1)
            e[1].append(id2)
            edge_info.append("bridge")
        prev_root_id = cur_root_id
    # id to embeddings
    # x = torch.tensor([ for token in node_attr])
    # done building edges and nodes
    x = torch.tensor(list(range(n)))
    e = torch.tensor(e)
    G = Data(x=x, edge_index=e, edge_attr=edge_info, node_attr=node_info)
    return G



####################################################################################
# if direct use allen nlp dependencies predictor without sentence segmentation     #
####################################################################################
def text2graph_allennlp(text: str, dep_predictor : BiaffineDependencyParserPredictor) -> Data:
    """
    text2doc by nlp
    doc2graph by utils.doc2graph 
    """
    return doc2graph_allennlp(dep_predictor.predict(sentence=text))

# Stanza Parse Seems Stabler than Allennlp's...
def doc2graph_allennlp(doc: Dict) -> Data:
    """
    input: allen dependecies (Dict)
    return G = {
     x: id tensor
     edge_idx : edges size = (2, l-1)
     edge_attr: (u, v, edge_type in str)
     node_attr: text
    }
    """
    # add root token for each sentences
    n = len(doc["words"])
    e = [list(range(1, n+1)),doc["predicted_heads"]]
    edge_attr = list(zip(e[0], e[1], doc["predicted_dependencies"]))
    node_attr = ["[ROOT]"]
    node_attr.extend(doc["words"])
    x = torch.tensor(list(range(n)))
    e = torch.tensor(e)
    G = Data(x=x, edge_index=e, edge_attr=edge_attr, node_attr=node_attr)
    return G