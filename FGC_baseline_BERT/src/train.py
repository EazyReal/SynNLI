from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import os
from collections import defaultdict

from dataset import * 
from model import * 
import config as config

TEST = True
LOG = True

# (instance/batchsize)*epcho = # batch
BATCH_SIZE = 
NUM_EPOCHS = 6
LR = 0.00001 # 1e-5
WEIGHT_DECAY = 0.01

NUM_WARMUP = 100

# Load Data using dataset.py

class SER_Trainer:
    def __init__(self, model):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def eval(self, )
    
    def train(self, model, train_set=None, dev_set=None, batch_size, collate_fn=create_mini_batch, model_file_path):
        
    
    