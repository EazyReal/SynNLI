# internal
import config
import utils

# external
import json
import torch
from torch_geometric.data.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
from stanza.models.common.doc import Document
from tqdm import tqdm as tqdm
from tqdm import tqdm_notebook as tqdm_nb

# load json data from preprocessed file
def load_jdata(data_file, function_test=False):
    # data_file_loading
    with open(data_file) as fo:
        raw_lines = fo.readlines()
        jdata = [json.loads(line) for line in raw_lines]
    return jdata

# bulid GraohData
class GraphData(Data):
    """
    dependences: utils.doc2graph 
    input data:json, word2idx: dict(word->token_id), to_lower?
    data is a raw json object of parsed result
    {
    config.hf: parsed hypothesis
    config.pf: parsed premise
    config.lf: label as one hot tensor of size (1, num_classes)
    config.idf: problem id, string
    }
    output a GraphData type object
    """
    def __init__(self, data, word2idx, tolower=True):
        super(GraphData, self).__init__()
        g_p = utils.doc2graph(Document(data[config.pf]))
        g_h = utils.doc2graph(Document(data[config.hf]))
        self.edge_index_p = g_p.edge_index
        self.edge_index_h = g_h.edge_index
        #print(g_p.node_attr)
        # care [ROOT] [UNK] should not get lower!!!
        if tolower == True:
            self.x_p = torch.tensor([ word2idx[w.lower() if w[0] != "[" or w[-1] != "]" else w] for w in g_p.node_attr], dtype=torch.long)
            self.x_h = torch.tensor([ word2idx[w.lower() if w[0] != "[" or w[-1] != "]" else w] for w in g_h.node_attr], dtype=torch.long)
        else:
            print("not to lower")
            self.x_p = torch.tensor([ word2idx[w] for w in g_p.node_attr], dtype=torch.long)
            self.x_h = torch.tensor([ word2idx[w] for w in g_h.node_attr], dtype=torch.long)
        label_onehot = torch.zeros([1, config.NUM_CLASSES])
        label_onehot[0][data[config.lf]] = 1
        #label_onehot = label_onehot.squeeze()
        #print(label_onehot.size())
        self.label = label_onehot.to(dtype=torch.float)
        self.pid = data[config.idf]
    def __inc__(self, key, value):
        if key == 'edge_index_p':
            return self.x_p.size(0)
        if key == 'edge_index_h':
            return self.x_h.size(0)
        else:
            return super(GraphData, self).__inc__(key, value)
"""
# collate_fn is implemented by pytorch geo
# we need to add follow_batch=[] to handle batch
# usage  = Loader
loader = DataLoader(dev_data_set, batch_size=3, follow_batch=['x_p', 'x_h', 'label'])
batch = next(iter(loader))
"""


# class of dataset
class GraphDataset(InMemoryDataset):
    """
    input: data_file : str/Path, word2idx: defaultdict
    """
    def __init__(self, data_file, word2idx):
        super(GraphDataset, self).__init__()
        self.jdata = []
        self.data = []
        with open(data_file) as fo:
            raw_lines = fo.readlines()
            for line in tqdm(raw_lines):
                self.jdata.append(json.loads(line))
                self.data.append(GraphData(self.jdata[-1], word2idx, tolower=True))
        return
        
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]