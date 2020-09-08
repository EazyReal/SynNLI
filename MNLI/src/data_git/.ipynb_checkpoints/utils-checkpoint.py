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
from torch_geometric.data.data import Data as PytorchGeoData
from torch_geometric.data import DataLoader as PytorchGeoDataLoader
## Stanza
import stanza
from stanza.models.common.doc import Document as StanzaDocument
from stanza.pipeline.core import Pipeline as StanzaPipeline
## allennlp model
from allennlp_models.structured_prediction.predictors.srl import SemanticRoleLabelerPredictor
from allennlp_models.structured_prediction.predictors.biaffine_dependency_parser import BiaffineDependencyParserPredictor
from allennlp.predictors.predictor import Predictor #

## self

###########################################################################################

# root_token
root_token = "$"

def g2sent(g : PytorchGeoData):
    return " ".join(g.node_attr)



def draw_edges(edge_index: torch.Tensor, tokens: List[str])->None:
    fig = plt.figure() # figsize=None
    ax = fig.add_subplot(111)
    cax = ax.imshow(att, cmap='bone', origin='upper')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticks(np.arange(len(seq)))
    ax.set_xticklabels(seq)
    ax.set_yticklabels([])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    # Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title("Attention")
    fig.tight_layout()
    plt.show()
    return 

def draw(data : PytorchGeoData, node_size=1000, font_size=12, save_img_file=None):
    """
    input: (torch_geometric.data.data.Data, path or string)
    effect: show and save graph data, with graphviz layout visualization
    """
    G = to_networkx(data)
    pos = nx.nx_pydot.graphviz_layout(G)
    if(data.edge_attr != None):
        edge_labels = {(u.item(),v.item()):lab for u,v,lab in zip(data.edge_index[0], data.edge_index[1], data.edge_attr)}
    else:
        edge_labels = None
    if(data.node_attr != None):
        node_labels = dict(zip(G.nodes, data.node_attr))
    else:
        node_labels = None
    nx.draw(G, pos=pos, nodecolor='r', edge_color='b', node_size=node_size, with_labels=False)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=font_size)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=font_size)
    print(G.nodes)
    print(G.edges)
    if save_img_file is not  None:
        plt.savefig(save_img_file)
    plt.show()
    return

######################################################################################################
# Stanza   #
######################################################################################################

def text2graph(text : str, nlp : StanzaPipeline) -> PytorchGeoData:
    """
    text2doc by Stanza
    doc2graph by utils.doc2graph 
    """
    return doc2graph(nlp(text))
    
def doc2graph(doc : Union[StanzaDocument, List]) -> PytorchGeoData:
    """
    input:
        doc : Union[StanzaDocument, List]
        selected_features : List[str] (this is to be added)
    output:
        PytorchGeoData : G
        G = {
         x: id tensor
         edge_idx : edges size = (2, l-1)
         edge_attr: (u, v, edge_type in str)
         node_attr: list of StanzaWords
        }
    Note that in Stanza, token is a list of words
    """
    if isinstance(doc, list): #convert to Doc first if is in dict form ([[dict]])
        doc = StanzaDocument(doc)
    # add root token for each sentences
    n = doc.num_words
    e = [[],[]]
    edge_info = []
    node_info = []
    prev_token_sum = 0
    # add edge function:
    def add_edge(id1, id2, type_, bidirectional=True):
        """
        add edge to list
        bidirectional?
        """
        e[0].append(id1)
        e[1].append(id2)
        edge_info.append(type_)
        if bidirectional:
            e[0].append(id2)
            e[1].append(id1)
            edge_info.append("reverse:"+type_)
    # get original dependency
    for idx, sent in enumerate(doc.sentences):
        # node info by index(add root at the beginning of every sentence)
        node_info.extend(sent.words)
        # edge info by index of u in edge (u,v)
        for dep in sent.dependencies:
            if dep[1] != "root":
                id1 = prev_token_sum + int(dep[0].id)-1
                id2 = prev_token_sum + int(dep[2].id)-1
                add_edge(id1, id2, dep[1])
        prev_token_sum += len(sent.words)
        # sent.print_dependencies()
        # sent.print_words()
    # add constituent edges
    for i in range(n-1):
        add_edge(i, i+1, "const:next", bidirectional=False)
        add_edge(i+1, i, "const:prev", bidirectional=False)
    # done building edges and nodes
    # print(n, e, edge_info, node_info, sep='\n')
    x = torch.tensor(list(range(n)))
    e = torch.LongTensor(e) # use longtensor explicitly so that Data can init without type issue
    G = PytorchGeoData(x=x, edge_index=e, edge_attr=edge_info, node_attr=node_info)
    return G



####################################################################################
# if direct use allen nlp dependencies predictor without sentence segmentation     #
####################################################################################
def text2graph_allennlp(text: str, dep_predictor : BiaffineDependencyParserPredictor) -> PytorchGeoData:
    """
    text2doc by nlp
    doc2graph by utils.doc2graph 
    """
    return doc2graph_allennlp(dep_predictor.predict(sentence=text))

# Stanza Parse Seems Stabler than Allennlp's...
def doc2graph_allennlp(doc: Dict) -> PytorchGeoData:
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
    node_attr = [root_token]
    node_attr.extend(doc["words"])
    x = torch.tensor(list(range(n)))
    e = torch.tensor(e)
    G =PytorchGeoData(x=x, edge_index=e, edge_attr=edge_attr, node_attr=node_attr)
    return G


"""
# root version doc2graph
def doc2graph(doc : Union[StanzaDocument, List]) -> PytorchGeoData:
    if isinstance(doc, list): #convert to Doc first if is in dict form ([[dict]])
        doc = StanzaDocument(doc)
    # add root token for each sentences
    n = doc.num_tokens+len(doc.sentences)
    e = [[],[]]
    edge_info = []
    node_info = []
    prev_token_sum = 0
    prev_root_id = 0
    cur_root_id = 0
    # add edge function:
    def add_edge(id1, id2, type_, bidirectional=True):
        e[0].append(id1)
        e[1].append(id2)
        edge_info.append(type_)
        if bidirectional:
            e[0].append(id2)
            e[1].append(id1)
            edge_info.append("reverse:"+type_)
    # get original dependency
    for idx, sent in enumerate(doc.sentences):
        # node info by index(add root at the beginning of every sentence)
        cur_root_id = len(node_info)
        node_info.append(root_token)
        for token in sent.tokens:
            node_info.append(token.to_dict()[0]['text'])
        # edge info by index of u in edge (u,v)
        for dep in sent.dependencies:
            id1 = prev_token_sum + int(dep[0].to_dict()["id"]) # here no need to -1 because of first root
            id2 = prev_token_sum + int(dep[2].to_dict()["id"])
            add_edge(id1, id2, dep[1])
        prev_token_sum += len(sent.tokens)+1
        # add links between sentence roots
        if(cur_root_id != 0):
            id1 = prev_root_id
            id2 = cur_root_id
            add_edge(id1, id2, "bridge")
        prev_root_id = cur_root_id
    # done building edges and nodes
    x = torch.tensor(list(range(n)))
    e = torch.tensor(e)
    G = PytorchGeoData(x=x, edge_index=e, edge_attr=edge_info, node_attr=node_info)
    return G
"""