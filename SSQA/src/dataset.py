# dependencies
import json
import torch
import pandas as pd

from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

from pathlib import Path

import config

# params
PRETRAINED_MODEL_NAME = config.BERT_EMBEDDING
BERT_MAX_INPUT_LEN = config.BERT_MAX_INPUT_LEN

# custom dataset for FGC data
class FGC_Dataset(Dataset):
    """
        FGC release all dev.json
        usage FGC_Dataset(file_path, mode, tokenizer)
        for tokenizer:
            PRETRAINED_MODEL_NAME = "bert-base-chinese"
            tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        for file_path:
            something like ./FGC_release_1.7.13/FGC_release_all_dev.json
        for mode:
            ["train", "develop", "test"]
    """
    # read, preprocessing
    def __init__(self, data_file_ref, mode="train", tokenizer=None):
        # load raw json
        assert mode in ["train", "develop", "test"]
        self.mode = mode
        with open(data_file_ref) as fo:
            self.raw_data = json.load(fo)
        if tokenizer == None:
                tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        self.tokenizer = tokenizer 
        self.tokenlized_pair = None
        
        # generate raw pairs of q sent s
        self.raw_pair = list()
        for instance in self.raw_data:
            q = instance["QUESTIONS"][0]["QTEXT_CN"]
            sentences = instance["SENTS"]
            for idx, sent in enumerate(sentences):
                # check if is supporting evidence
                lab = float(idx in instance["QUESTIONS"][0]["SHINT_"])
                self.raw_pair.append((q, sent["text"], lab))
        
        # generate tensors 
        self.dat = list()
        for instance in self.raw_pair:
            q, sent, label = instance
            
            if mode is not "test":
                label_tensor = torch.tensor(label)
            else:
                label_tensor = None
            
            # first sentence, use bert tokenizer to cut subwords
            subwords = ["[CLS]"]
            q_tokens = self.tokenizer.tokenize(q)
            subwords.extend(q_tokens)
            subwords.append("[SEP]")
            len_q = len(subwords)
            
            # second sentence
            sent_tokens = self.tokenizer.tokenize(sent)
            subwords.extend(sent_tokens)
            
            # truncate if > BERT_MAX_INPUT_LEN 
            if(len(subwords) > BERT_MAX_INPUT_LEN-1):
                subwords = subwords[:BERT_MAX_INPUT_LEN-1]
            
            subwords.append("[SEP]")
            len_sent = len(subwords) -len_q
            
            
            # subwords to ids, ids to torch tensor
            ids = self.tokenizer.convert_tokens_to_ids(subwords)
            tokens_tensor = torch.tensor(ids)
            
            # segments_tensor
            segments_tensor = torch.tensor([0] * len_q + [1] * len_sent, dtype=torch.long)
            self.dat.append((tokens_tensor, segments_tensor, label_tensor))
            
        return None
    
    # get one data of index idx
    def __getitem__(self, idx):
        return self.dat[idx]
    
    def __len__(self):
        return len(self.dat)
    
    # get id2qid for evalutation
    def get_id2qid(self, data_file_ref = None):
        if data_file_ref == None:
            raw_data = self.raw_data
            if not raw_data:
                print("No built raw data or data_file_ref")
                assert(False)
        else:
            with open(data_file_ref) as fo:
                raw_data = json.load(fo)
        
        id_to_qid = []
        for qidx, instance in enumerate(raw_data):
            cur_qid = instance["QUESTIONS"][0]["QID"]
            sentences = instance["SENTS"]
            for idx, sent in enumerate(sentences):
                id_to_qid.append(cur_qid)
        return id_to_qid

    
class SSQA_Dataset(Dataset):
    """
        SSQA release 0.8, training set is still in developmemt
        usage :  FGC_Dataset(file_path, mode, tokenizer)
        for tokenizer:
            PRETRAINED_MODEL_NAME = "bert-base-chinese"
            tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        for file_path:
            refer to config
        for mode:
            ["train", "develop", "test"]
    """
    # read, preprocessing
    def __init__(self, data_file_path, mode="develop", tokenizer=None):
        # load raw json
        assert mode in ["train", "develop", "test", "dev"]
        self.mode = mode
        with open(data_file_path) as fo:
            self.raw_data = json.load(fo)
        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_EMBEDDING)
        else:
            self.tokenizer = tokenizer 
        self.tokenlized_pair = None
        
        # generate raw pairs of q sent s
        self.raw_pair = list()
        for instance in self.raw_data:
            q = instance["qtext"]
            sentences = instance["paragraphs"]
            for idx, sent in enumerate(sentences):
                # check if is supporting evidence
                lab = idx in instance["supporting_paragraphs_index"]
                self.raw_pair.append((q, sent, lab))
        
        # generate tensors 
        self.dat = list()
        for instance in self.raw_pair:
            q, sent, label = instance
            
            if mode is not "test":
                label_tensor = torch.tensor(label)
            else:
                label_tensor = None
            
            # first sentence, use bert tokenizer to cut subwords
            subwords = ["[CLS]"]
            q_tokens = self.tokenizer.tokenize(q)
            subwords.extend(q_tokens)
            subwords.append("[SEP]")
            len_q = len(subwords)
            
            # second sentence
            sent_tokens = self.tokenizer.tokenize(sent)
            subwords.extend(sent_tokens)
            
            # truncate if > BERT_MAX_INPUT_LEN 
            if(len(subwords) > config.BERT_MAX_INPUT_LEN-1):
                subwords = subwords[:config.BERT_MAX_INPUT_LEN-1]
            
            subwords.append("[SEP]")
            len_sent = len(subwords) -len_q
            
            
            # subwords to ids, ids to torch tensor
            ids = self.tokenizer.convert_tokens_to_ids(subwords)
            tokens_tensor = torch.tensor(ids)
            
            # segments_tensor
            segments_tensor = torch.tensor([0] * len_q + [1] * len_sent, dtype=torch.long)
            self.dat.append((tokens_tensor, segments_tensor, label_tensor))
            
        # id to q
        self.id_to_qid = []
        for qidx, instance in enumerate(self.raw_data):
            cur_qid = instance["qid"]
            sentences = instance["paragraphs"]
            for idx, sent in enumerate(sentences):
                self.id_to_qid.append(cur_qid)
            
        return None
    
    # get one data of index idx
    def __getitem__(self, idx):
        return self.dat[idx]
    
    def __len__(self):
        return len(self.dat)
    
    
"""
DataLoader for minibatch
in each batch
- tokens_tensors  : (batch_size, max_seq_len_in_batch) 
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # use(have) label or not
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad to same length
    tokens_tensors = pad_sequence(tokens_tensors,  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,  batch_first=True)
    
    # attention masks, set none-padding part to 1 for LM to attend
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill( tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids