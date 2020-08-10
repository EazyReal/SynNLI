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
## graph
import networkx as nx
import matplotlib.pyplot as plt
## nn
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data.data import Data
from torch_geometric.data import DataLoader
## model
import stanza
from stanza.models.common.doc import Document

## self
import config
import utils
from model import *

###########################################################################################



###########################################################################################


# alias
p = config.pf
h = config.hf
l = config.lf


def draw(data, node_size=1000, font_size=12, save_img_file=None):
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

def token2sent(ids, word):
    return ' '.join([ word[idx] for idx in ids])

def text2graph(text, nlp, word2idx=None):
    """
    text2doc by Stanza
    doc2graph by utils.doc2graph 
    """
    return doc2graph(nlp(text), word2idx=word2idx)
    
def doc2graph(doc, word2idx=None):
    """
    2020/8/4 18:30
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
    e = [[],[]]
    edge_info = []
    node_info = []
    prev_token_sum = 0
    prev_root_id = 0
    cur_root_id = 0
    # get original dependency
    for idx, sent in enumerate(doc.sentences):
        sent.print_dependencies
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
            edge_info.append((id1, id2, dep[1]))
        prev_token_sum += len(sent.tokens)+1
        # add links between sentence roots
        if(cur_root_id != 0):
            id1 = prev_root_id
            id2 = cur_root_id
            e[0].append(id1)
            e[1].append(id2)
            edge_info.append((id1, id2, "bridge"))
        prev_root_id = cur_root_id
    # id to embeddings
    # x = torch.tensor([ for token in node_attr])
    # done building edges and nodes
    if word2idx == None:
        # print("x is not id now, node info is in node_attr as text")
        x = torch.tensor(list(range(doc.num_tokens+len(doc.sentences))))
    else:
        x = torch.tensor([ word2idx[token] for token in node_info])
    e = torch.tensor(e)
    G = Data(x=x, edge_index=e, edge_attr=edge_info, node_attr=node_info)
    return G

# load glove vector, care return type
def load_glove_vector(glove_embedding_file = config.GLOVE, dimension=config.GLOVE_DIMENSION, save_vocab = config.GLOVE_VOCAB, save_word2id = config.GLOVE_WORD2ID, save_dict=True):
    words = []
    idx = 0
    word2idx = defaultdict(int) # return 0 for unkown word
    glove = []
    #glove = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(glove_embedding_file, 'r') as fo:
        lines = fo.readlines()
        # add [UNK] handler
        words.append("[UNK]")
        word2idx["[UNK]"] = idx
        glove.append(np.zeros(300)) # 300 is vector dimension
        idx += 1
        # add [ROOT] handler
        words.append("[ROOT]")
        word2idx["[ROOT]"] = idx
        glove.append(np.zeros(300)) # 300 is vector dimension
        idx += 1
        # load vectors
        for line in tqdm(lines):
            line = line.split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            vector = np.asarray(line[1:], "float32")
            glove.append(vector)
            idx += 1
    #glove = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    #glove.flush()
    glove = torch.tensor(glove, dtype=torch.float32) 
    if save_dict == True:
        pickle.dump(words, open(config.GLOVE_ROOT / config.GLOVE_VOCAB, 'wb'))
        pickle.dump(word2idx, open(config.GLOVE_ROOT / config.GLOVE_WORD2ID, 'wb'))
        pickle.dump(glove, open(config.GLOVE_ROOT / config.GLOVE_SAVED_TENSOR, 'wb'))
    return glove, words, word2idx, idx


# parse one example of MNLI style data
def process_one_example(data, nlp):
    ret = {}
    ret[config.idf] = data[config.idf]
    ret[config.pf] = nlp(data[config.pf]).to_dict()
    ret[config.hf] = nlp(data[config.hf]).to_dict()
    ret[config.lf] = config.label_to_id[data[config.lf]]
    return ret

# parse the MNLI style data with Stanza and save the result
def parse_data(data_file=config.DEV_MA_FILE, target=config.PDEV_MA_FILE, function_test=False, force_exe=False):
    """
    input (data = str, embedding = str, target file = str)
    effect preprocess and save data to target
    ouput preprocessed data
    
    parsed data is in jsonl (each line is a json)
    {
        config.idf : id(in string)
        config.hf : Stanza Doc,
        config.pf : Stanza Doc,
        config.lf : int 
    }
    """
    # alias
    p = config.pf
    h = config.hf
    l = config.lf
    
    # stanza dinit
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    
    # data_file_loading
    with open(data_file) as fo:
        raw_lines = fo.readlines()
        json_data = [json.loads(line) for line in raw_lines]
        
    if function_test:
        json_data = json_data[:10]
        
    if os.path.isfile(str(target)) and not force_exe:
        print("file " + str(target) + " already exist")
        print("if u still want to procceed, add force_exe=True in function arg")
        print("exiting")
        return None
    else:
        print("creating file " + str(target) + " to save result")
        print("executing")
        
    # dependency parsing and jsonl saving
    with jsonl.open(target, mode='w') as writer:
        parsed_data = []
        for data in tqdm(json_data):
            # only add those who have gold labels
            if(data[l] not in config.label_to_id.keys()):
                continue
            pdata = process_one_example(data, nlp)
            parsed_data.append(pdata)
            writer.write(pdata)
        
    return parsed_data