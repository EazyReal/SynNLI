######################################
# must change save_model_folder to designate save param path
# statistics will go to the save_model_folder too
#####################################
import os
from pathlib import Path

TRANSFORMER_NAME = "bert-base-uncased"

######################################
# class for config dict 
#####################################
class Config():
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        return

######################################
# ROOTS 
######################################
SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PROJ_ROOT = SRC_ROOT.parent #care if multi source under src

DATA_ROOT = PROJ_ROOT / "data" / "multinli_1.0"
PARAM_PATH = PROJ_ROOT / "param"
P_DATA_ROOT = PROJ_ROOT / "data" / "MNLI_Stanza" # use stanza
PARSER_ROOT = PROJ_ROOT / "parsers"

######################################
# data path
######################################
TEST_FILE = DATA_ROOT / "multinli_1.0_dev_mismatched.jsonl"
DEV_FILE = DATA_ROOT / "multinli_1.0_dev_matched.jsonl"
TRAIN_FILE = DATA_ROOT / "multinli_1.0_train.jsonl"

######################################
# parser gz
######################################
DEP_PARSER_MODEL = PARSER_ROOT / "biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
SRL_LABELER_MODEL = PARSER_ROOT / "bert-base-srl-2020.03.24.tar.gz"

######################################
# parsed data
######################################
P_TEST_FILE = P_DATA_ROOT / "pre_multinli_1.0_dev_mismatched.jsonl"
P_DEV_FILE = P_DATA_ROOT / "pre_multinli_1.0_dev_matched.jsonl"
P_TRAIN_FILE = P_DATA_ROOT / "pre_multinli_1.0_train.jsonl"
P_DEVELOPMENT_FILE = P_DATA_ROOT / "for_development.jsonl"

######################################
# save model
######################################
SAVE_MODEL_FOLDER = "SynNLIv0.1_glove_GAT3"
LOG_FILE_PATH =  PARAM_PATH / SAVE_MODEL_FOLDER / "train_log.txt"
PURE_TRAIN_STAT_PATH = PARAM_PATH / SAVE_MODEL_FOLDER / "stat.jsonl"

######################################
# Preprocssing Config / Data related
######################################
label_to_id = {
    "contradiction" : 0,
    "neutral" : 1,
    "entailment" : 2,
}
id_to_label = ["contradiction", "neutral", "entailment"]

h_field = "sentence2"
p_field = "sentence1"
label_field = "gold_label"
index_field = "pairID"
# alias
hf = "sentence2"
pf = "sentence1"
lf = "gold_label"
idf = "pairID"

######################################
# MODEL
######################################
#CROSS_ATTENTION_HIDDEN_SIZE = 392
NUM_CLASSES = 3
EMBEDDING_D = 300
HIDDEN_SIZE = 300

######################################
# Bert Enbedding
######################################
BERT_EMBEDDING = "bert-base-uncased" #cased?
BERT_MAX_INPUT_LEN = 512
BERT_EMBEDDING_D = 786

######################################
# NLI Model config class and instance
######################################

# for conv encoder
NUM_CONV_LAYERS = 3
NUM_CONV_ATT_HEADS = 3

# for big model
nli_config_dict = {
    "hidden_size" : 300,
    "embedding" : "glove300d",
    "encoder" : "gat",
    "cross_att" : "scaled_dot",
    "aggregation" : "max+mean",
    "local_cmp" : "2-layer-FNN",
    
    "prediction" : "2-layer-FNN",
    "activation" : "relu"
}
model_config = Config(nli_config_dict)

######################################
# Trainning
######################################
"""
reference for bert fine-tuning
BATCH_SIZE = # 32 / 16 / 8
NUM_EPOCHS = # 2 / 3 / 4
LR = # 5/3/2 1e-5
WEIGHT_DECAY = 0.01

reference for KAGNet
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3
"""
trainer_config_dict = {
    "batch_size" : 32,
    "num_epochs" : 5,
    "lr" : 5e-4,
    "weight_decay" : 0.01,
    "max_grad_norm" : 1.0,
    "num_warm_up" : 100,
    "drop_out_prob" : 0.1
}
trainer_config = Config(trainer_config_dict)
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 5*1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NUM_WARMUP = 100
DROUP_OUT_PROB = 0.1

######################################
# collate
######################################
follow_batch = ["x_p", "x_h", "label"]
tensor_attr_list = ["edge_index_p", "edge_index_h", "x_p", "x_h", "label"]