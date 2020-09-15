# Dependencies
import torch.nn as nn
from transformers import BertModel
import torch
import math

import config

# v1 sum pool

# v2 local use p, h, p-h, p+h
# aggregation use avg pool + max pool 

class CrossBERTModel(nn.Module):
    """
    bert cross attention model
    h, p go through bert and get their contexulized embedding saparately
    and do soft alignment and prediction as in decomp-att paper
    this is a embedding enhanced version of decomp-att
    """
    def __init__(self, bert_encoder=None, cross_attention_hidden=config.CROSS_ATTENTION_HIDDEN_SIZE):
        super().__init__()
        #bert encoder
        if bert_encoder == None or not isinstance(bert_encoder, BertModel):
            print("unkown bert model choice, init with config.BERT_EMBEDDING")
            bert_encoder = BertModel.from_pretrained(config.BERT_EMBEDDING)
        self.bert_encoder = bert_encoder
        # dropouts
        self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
        self.activation = nn.ReLU(inplace=True)
        # linear layers for cross attention, with biased?
        self.cross_attention_hidden = cross_attention_hidden
        self.Wq = nn.Parameter(torch.Tensor(bert_encoder.config.hidden_size, self.cross_attention_hidden))
        self.Wk = nn.Parameter(torch.Tensor(bert_encoder.config.hidden_size, self.cross_attention_hidden))
        self.Wv = nn.Parameter(torch.Tensor(bert_encoder.config.hidden_size, bert_encoder.config.hidden_size))
        self.Wo = nn.Parameter(torch.Tensor(bert_encoder.config.hidden_size, bert_encoder.config.hidden_size))
        nn.init.xavier_uniform_(self.Wk, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.Wq, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.Wv, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('relu'))
        
        ## cls
        self.classifier = nn.Linear(2*bert_encoder.config.hidden_size, config.NUM_CLASSES)
        
        forward_expansion = 1 # can change
        #compare stage fnn 2*d=>d
        self.fnn = nn.Sequential(
            nn.Linear(4*bert_encoder.config.hidden_size, forward_expansion*bert_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*bert_encoder.config.hidden_size, bert_encoder.config.hidden_size),
        )
        # critrion
        self.criterion = nn.BCEWithLogitsLoss()
    
    """
    cross attention, similar to Decomp-Att
    but no fowrad nn, use Wk Wq Wv
    input: query vector(b*n*d), content vector(b*m*d)
    ouput: sof aligned content vector to query vector(b*n*d)
    """
    def cross_attention(self, h1, h2, mask=None):
        Q = torch.matmul(h1, self.Wq)
        #K = torch.matmul(h2, self.Wk)
        K = torch.einsum("bnx,xy->bny", [h2, self.Wk])
        V = torch.matmul(h2, self.Wv)
        #Kt = torch.matmul(h2, self.Wk).permute(0,2,1)
        #E = torch.matmul(Q, Kt)
        E = torch.einsum("bnd,bmd->bnm", [Q, K]) # batch, n/m, dimension
        if mask is not None:
            E = E.masked_fill(mask==0, float(-1e7))
        A = torch.softmax(E / (math.sqrt(self.cross_attention_hidden)), dim=2) #soft max dim = 2
        # attention shape: (N, heads, query_len, key_len)
        aligned_2_for_1 = torch.einsum("bnm,bmd->bnd", [A, V])
            
        return aligned_2_for_1
    
    def forward_nn(self, batch):
        """
        'sentence1' : {'input_ids', 'token_type_ids', 'attention_mask'} batch*len*d
        'sentence2' :  {'input_ids', 'token_type_ids', 'attention_mask'}
        'gold_label' : batch*1
        """
        # get bert contextualized embedding
        hh, poolh = self.bert_encoder(input_ids=batch[config.h_field]['input_ids'],
                                         token_type_ids=batch[config.h_field]['token_type_ids'],
                                         attention_mask=batch[config.h_field]['attention_mask'])
        hp, poolp = self.bert_encoder(input_ids=batch[config.p_field]['input_ids'],
                                         token_type_ids=batch[config.p_field]['token_type_ids'],
                                         attention_mask=batch[config.p_field]['attention_mask'])
        # soft alignment, not considering mask...
        mh = attention_mask=batch[config.h_field]['attention_mask']
        mp = attention_mask=batch[config.p_field]['attention_mask']
        maskph = torch.einsum("bn,bm->bnm", [mh, mp])
        maskhp = torch.einsum("bn,bm->bnm", [mp, mh])
        # (b, l_h, d)
        p_hat = self.cross_attention(hh, hp, maskph) # b * l_h * d
        # aligned_h_for_p = self.cross_attention(hp, hh, maskhp) # b * l_p *d
        
        # comparison stage
        # (b, l_h, d)
        cmp_hp = self.fnn(torch.cat((p_hat, hh, p_hat-hh, p_hat*hh), dim=2))
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
        batch[config.label_field] = batch[config.label_field].to(dtype=torch.float)
        loss = self.criterion(logits, batch[config.label_field])
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