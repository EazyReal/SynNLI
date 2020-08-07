# Dependencies
import torch
import torch.nn as nn
import math
import logging
#from transformers import BertModel

import config
from config import nli_config
import utils
#from torch.nn import (LSTM)

"""
implement baseline first
glove
GAT
cross att
local comparison(F(h;p;h-p;h*p))
agrregate by mean and max
prediction
"""


# work
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree


# conv, GAT
"""
GATConv(in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, bias: bool = True, **kwargs)
"""

# conv, do later
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

# apply l layers of GraphConv
class GraphEncoder(nn.Module):
    def __init__(self, conv="gat", input_d=config.EMBEDDING_D, output_d = config.EMBEDDING_D, num_heads=config.NUM_CONV_ATT_HEADS, num_layers=config.NUM_CONV_LAYERS):
        super().__init__()
        self.dropout = nn.Dropout(p=config.DROUP_OUT_PROB)
        self.activation = nn.ReLU(inplace=True)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d = input_d
        assert(self.d%self.num_heads == 0)
        if conv == "hggcn":
            self.conv = None
        elif conv == "gat":
            # negative_slope is slope of LeakyRelu
            self.conv = GATConv(in_channels = self.d,
                                out_channels = self.d//self.num_heads,
                                heads = self.num_heads, concat = True,
                                negative_slope = 0.2,
                                dropout = 0.0,
                                add_self_loops= True,
                                bias = True)
    def forward(self, x, edge_index):
        # print(self.conv)
        for l in range(self.num_layers):
            x, (edge, att) = self.conv(x, edge_index, return_attention_weights=True)
            # print(x.size(), att.size())
        return x
        
        
class CrossAttentionLayer(nn.Module):
    """
    cross attention, similar to Decomp-Att
    but no fowrad nn, use Wk Wq Wv
    input: query vector(b*n*d), content vector(b*m*d)
    ouput: sof aligned content vector to query vector(b*n*d)
    """
    def __init__(self, input_d, output_d, hidden_d, number_of_heads=1):
        super().__init__()
        self.dropout = nn.Dropout(p=config.DROUP_OUT_PROB)
        self.activation = nn.ReLU(inplace=True)
        self.input_d = input_d
        self.output_d = output_d
        self.hidden_d = hidden_d
        self.number_of_heads = number_of_heads
        # params
        #self.Wq = nn.Parameter(torch.Tensor(input_d, hidden_d))
        self.Wq = nn.Linear(input_d, hidden_d, bias=False)
        self.Wk = nn.Linear(input_d, hidden_d, bias=False)
        self.Wv = nn.Linear(input_d, output_d, bias=False)
        #self.Wo = nn.Parameter(torch.Tensor(input_d, output_d))
        #nn.init.xavier_uniform_(self.Wk, gain=nn.init.calculate_gain('linear'))
        #nn.init.xavier_uniform_(self.Wq, gain=nn.init.calculate_gain('linear'))
        #nn.init.xavier_uniform_(self.Wv, gain=nn.init.calculate_gain('linear'))
        #nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('relu'))
        
    def forward(self, h1, h2, mask=None):
        #Q = torch.matmul(h1, self.Wq)
        Q = self.Wq(h1)
        K = self.Wk(h2)
        V = self.Wv(h2)
        E = torch.einsum("bnd,bmd->bnm", [Q, K]) # batch, n/m, dimension
        #print(E.size())
        if mask is not None:
            E = E.masked_fill(mask==0, float(-1e10))
        A = torch.softmax(E / (math.sqrt(self.hidden_d)), dim=2) #soft max dim = 2
        # attention shape: (N, heads, query_len, key_len)
        aligned_2_for_1 = torch.einsum("bnm,bmd->bnd", [A, V])
            
        return aligned_2_for_1
        
class SynNLI_Model(nn.Module):
    """
    word embedding (glove/bilstm/elmo/bert) + SRL embedding
    graph encoder (GAT/HGAT/HetGT...)
    cross attention allignment (CrossAtt)
    local comparison(F(h;p;h-p;h*p))
    aggregation ((tree-)LSTM?)
    prediction (FeedForward)
    """
    def __init__(self, nli_config=config.nli_config, pretrained_embedding_tensor=None):
        super().__init__()
        self.config = nli_config
        d = self.config.hidden_size
        # dropouts
        self.dropout = nn.Dropout(p=config.DROUP_OUT_PROB)
        self.activation = nn.ReLU(inplace=True)
        # embedding
        if(self.config.embedding == "glove300d"):
            #pretrained_embedding_tensor = utils.load_glove_vector() should not be here
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding_tensor)
        else:
            self.embedding = nn.Embedding(config.GLOVE_VOCAB_SIZE, config.GLOVE_DIMENSION)
        # bi-lstm for contextualized embedding and for local comp
        self.bilstm_emd = nn.LSTM(input_size=d, hidden_size=d*2, num_layers=2, bidirectional=True, batch_first=True, dropout=0)
        self.bilstm_cmp = nn.LSTM(input_size=d, hidden_size=d*2, num_layers=2, bidirectional=True, batch_first=True, dropout=0)
        # encoder
        if self.config.encoder == None:
            self.encoder = None
        else:
            self.encoder = GraphEncoder(conv=self.config.encoder, input_d=d, output_d=d, num_heads=config.NUM_CONV_ATT_HEADS, num_layers=config.NUM_CONV_LAYERS)
        # cross_att
        if(self.config.cross_att == "scaled_dot"):
            self.cross_att = CrossAttentionLayer(input_d=d, output_d=d, hidden_d=d, number_of_heads=1)
        # local comp fnn h, p^, h-p^, h*p^
        self.local_cmp = nn.Sequential(nn.Linear(4*d, d), nn.ReLU(), nn.Linear(d,d))
        # aggregation is max
        # self.aggr = (partial)torch.max(dim=1, keep_dim=False)
        # cls
        self.classifier = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d,config.NUM_CLASSES))
        self.criterion = nn.BCEWithLogitsLoss()
    
    def get_batch_tensor(self, p, x_p_batch):
        """
        input: p : (n*d), x_p_batch : (n*1)
        ouput: batch dense version and corresponding mask : (batch*max_l*d)
        example in 
            [e1, e2, e3..., e5]
            [0, 0, 1, 2, 2]
        example out
            [[e1,e2], [e3, <pad>], [e4, e5]]
            [[1,1], [1, 0], [1, 1]]
        """
        d = self.config.hidden_size
        # remember device
        cuda_check = p.is_cuda
        if cuda_check:
            device_id = p.get_device()
            device = "cuda"
        #print(cuda_check, device)
        batch_size = x_p_batch[-1].item() + 1
        len_bp = torch.unique_consecutive(x_p_batch, return_counts=True)
        max_len_p = torch.max(len_bp[1]).item()
        #print(len_bp, max_len_p, sep='\n')
        bp = torch.zeros([batch_size, max_len_p, d])
        maskp = torch.zeros([batch_size, max_len_p], dtype=torch.long)
        ti = 0
        for bi in range(batch_size):
            for li in range(max_len_p):
                if li < len_bp[1][bi]:
                    bp[bi][li] = p[ti]
                    maskp[bi][li] = 1
                    ti += 1
                else:
                    continue
        if cuda_check:
            bp = bp.to(device)
            maskp = maskp.to(device)
        return bp, maskp
    
    def forward_nn(self, batch):
        """
        G(graph by edge list): batch.edge_index_p, batch.edge_index_h
        X(input token id): batch.x_p, batch.x_h
        B(batch info): batch.x_p_batch, batch.x_h_batch
        L(label): batch.label
        ID(index of problem): batch.pid
        """
        # alias 
        d = self.config.hidden_size
        # id to embedding
        w_p = self.embedding(batch.x_p)
        w_h = self.embedding(batch.x_h)
        #logging.debug(w_p.size())
        # get graph contextualized embedding
        p = self.encoder(w_p, batch.edge_index_p)
        h = self.encoder(w_h, batch.edge_index_h)
        #logging.debug(p.size())
        
        # rebuild batched by follow_batch , p->bp
        p, maskp = self.get_batch_tensor(p, batch.x_p_batch)
        #print(maskp)
        h, maskh = self.get_batch_tensor(h, batch.x_h_batch)
        #logging.debug(p.size(), maskp.size(), sep='\n')
            
        # soft alignment, not considering mask...
        maskhp = torch.einsum("bn, bm->bmn", maskp, maskh) #maskp = b*n, maskh = b*m, maskhp = b*m*n
        #logging.debug(maskhp[0])
        p_hat = self.cross_att(h, p, maskhp) 
        
        # comparison stage
        # (b, l_h, d)
        cmp_hp = self.local_cmp(torch.cat((p_hat, h, p_hat-h, p_hat*h), dim=2))
        #cmp_ph = self.fnn(torch.cat((aligned_h_for_p, hp), dim=2))
        
        # aggregatoin stage (mean + max for h part IMO)
        # (b, d)
        sent_hp_max = torch.max(cmp_hp, dim=1, keepdim=False)[0] # maxpool
        sent_hp_mean = torch.mean(cmp_hp, dim=1, keepdim=False) # meanpool
        
        #sent_ph = torch.sum(cmp_ph, dim=1, keepdim=False)
        # prediction get
        #logits = self.classifier(torch.cat((sent_hp, sent_ph), dim=1))
        logits = self.classifier(torch.cat((sent_hp_max, sent_hp_mean), dim=1))
        logits = logits.squeeze(-1)
        return logits
    
    # the nn.Module method
    def forward(self, batch):
        logits = self.forward_nn(batch)
        #print(batch.label.size())
        loss = self.criterion(logits, batch.label.view([-1, config.NUM_CLASSES]))
        return loss, logits
    
    # return sigmoded score
    def _predict_score(self, batch):
        logits = self.forward_nn(batch)
        scores = torch.sigmoid(logits)
        scores = scores.detach().cpu()
        return scores
    
    # return True False based on score + threshold
    def _predict(self, batch, threshold=0.5):
        scores = self._predict_score(batch)
        return torch.argmax(scores, dim=1) # return highest