import json
import torch
import pandas as pd

from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import config

from torch.nn.utils.rnn import pad_sequence
from functools import partial

##############################################################
# for batch                                                  #
##############################################################

"""
DataLoader for minibatch
in each batch
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""
def create_mini_batch(samples, tokenizer=BertTokenizer.from_pretrained(config.BERT_EMBEDDING)):
    label_ids = torch.tensor([[config.label_to_id[s[config.label_field]]] for s in samples])
    label_onehot = torch.zeros(len(samples), config.NUM_CLASSES).scatter_(1,label_ids,1)
    batch = {
        config.p_field : tokenizer([ s[config.p_field] for s in samples], padding=True, truncation=True, return_tensors="pt"),
        config.h_field : tokenizer([ s[config.h_field] for s in samples], padding=True, truncation=True, return_tensors="pt"),
        config.label_field: label_onehot
    }
    return batch

            
            
##############################################################


class MNLI_Raw_Dataset(Dataset):
    """
    MNLI set for CrossBERT baseline
    source: 
    wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip @ 2020/7/21 17:09
    self.j_data is list of jsons
    self.raw_data is list of (hyposesis, premise, gold label)
    """
    def __init__(self,
                 file_path=config.DEV_MA_FILE,
                 mode="develop",
                ):
        # super(MNLI_CrossBERT_Dataset, self).__init__()
        # decide config
        self.mode = mode
        # load raw data
        self.file_path = file_path
        with open(self.file_path) as fo:
            self.raw_lines = fo.readlines()
        # to json
        self.j_data = [json.loads(line) for line in self.raw_lines]
        self.data = [line for line in self.j_data if line[config.label_field] in config.label_to_id.keys()]
        
        return None
        
        
    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return len(self.data)
    
    
##############################################################
