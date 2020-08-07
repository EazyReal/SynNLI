# Dependencies
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from transformers import BertModel
import torch
import config

# Hyper Params

n_class = 2
PRETRAINED_MODEL_NAME = config.BERT_EMBEDDING

# calc pos weight for BCE in train stage!

# no need to applied pos_weight = torch.tensor([total/true_cnt, total/(1-true_cnt)])?

class BertSERModel(nn.Module):
    """
    baseline
    naive bert by NSP stype + linear classifier applied on [CLS] last hidden
    """
    
    def __init__(self, bert_encoder=None, pos_weight=None):
        super().__init__()
        if bert_encoder == None or not isinstance(bert_encoder, BertModel):
            print("unkown bert model choice, init with PRETRAINED_MODEL_NAME")
            bert_encoder = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.bert_encoder = bert_encoder
        self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_encoder.config.hidden_size, 1)
        self.pos_weight = pos_weight
        # critrion add positive weight
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward_nn(self, batch):
        """
        batch[0] = input, shape = (batch_size ,max_len_in_batch)
        batch[1] = token_type_ids (which sent)
        batch[2] = mask for padding
        batch[3] = labels
        """
        # the _ here is the last hidden states
        # q_poolout is a 768-d vector of [CLS]
        _, q_poolout = self.bert_encoder(batch[0],
                                         token_type_ids=batch[1],
                                         attention_mask=batch[2])
        # q_poolout = self.dropout(q_poolout), MT Wu : no dropout better, without 
        logits = self.classifier(q_poolout)
        # can apply nn.module.Sigmoid here to convert to p-distribution
        # score is indeed better (and more stable)
        logits = logits.squeeze(-1)
        return logits
    
    # the nn.Module method
    def forward(self, batch):
        logits = self.forward_nn(batch)
        batch[3] = batch[3].to(dtype=torch.float)
        loss = self.criterion(logits, batch[3])
        return loss
    
    # return sigmoded score
    def _predict_score(self, batch):
        logits = self.forward_nn(batch)
        scores = torch.sigmoid(logits)
        scores = scores.detach().cpu().numpy().tolist()
        return scores
    
    # return True False based on score + threshold
    def _predict(self, batch, threshold=0.5):
        scores = self._predict_score(batch)
        return [ 1 if score >= threshold else 0 for score in scores]
    
    # return result with assigned threshold, default = 0.5
    def predict_fgc(self, q_batch, threshold=0.5):
        scores = self._predict(q_batch)

        max_i = 0
        max_score = 0
        sp = []
        for i, score in enumerate(scores):
            if score > max_score:
                max_i = i
                max_score = score
            if score >= threshold:
                sp.append(i)

        if not sp:
            sp.append(max_i)

        return {'sp': sp, 'sp_scores': scores}