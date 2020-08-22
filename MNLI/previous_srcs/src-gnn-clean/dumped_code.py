

"""

# old version parse + build graph at the same time
def text2dep(s, nlp, word2idx=None):
    2020/8/3 18:30
    input (str:s, StanzaPipieline: nlp), s is of len l
    output (PytorchGeoData : G)
    G = {
     x: id tensor
     edge_idx : edges size = (2, l-1)
     edge_attr: (u, v, edge_type in str)
     node_attr: text
    }
    doc = nlp(s)
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
        x = torch.tensor(list(range(doc.num_tokens+len(doc.sentences))))
    else:
        x = torch.tensor([ word2idx[token] if token in word2idx.keys() else word2idx["[UNK]"] for token in node_info])
    e = torch.tensor(e)
    G = Data(x=x, edge_index=e, edge_attr=edge_info, node_attr=node_info)
    return G
"""

"""
import networkx as nx
from torch_geometric.utils.convert import to_networkx
def draw(data):
    G = to_networkx(data)
    nx.draw(G)
    plt.savefig("path.png")
    plt.show()
"""


"""
depdency data generator

def draw(data, node_size=1000, font_size=12):
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
    plt.savefig("path.png")
    
    plt.show()

def text2dep(s, nlp):
    doc = nlp(s)
    # add root token for each sentences
    x = torch.tensor(list(range(doc.num_tokens+len(doc.sentences))))
    #y = torch.tensor(list(range(doc.num_tokens+len(doc.sentences))))
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
        node_info.append("<ROOT>")
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
    # done building edges and nodes
    e = torch.tensor(e)
    G = Data(x=x, edge_index=e, edge_attr=edge_info, node_attr=node_info)
    return G

dep = text2dep(json_data[0][config.hf], nlp)
print("dep's node attr = ? : ", dep.node_attr)
print(dep)
draw(dep)
"""