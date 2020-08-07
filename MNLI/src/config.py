import os
from pathlib import Path

###################
# must change
# save_model_folder
##################

# whether log when executing
DEBUG = True
LOG = True
#log file path is below

# Paths 
SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PROJ_ROOT = SRC_ROOT.parent #care if multi source

DATA_ROOT = PROJ_ROOT / "data" / "multinli_1.0"
PARAM_PATH = PROJ_ROOT / "param"

PDATA_ROOT = PROJ_ROOT / "data" / "preprocessed_MNLI"

DEV_MMA_FILE = DATA_ROOT / "multinli_1.0_dev_mismatched.jsonl"
DEV_MA_FILE = DATA_ROOT / "multinli_1.0_dev_matched.jsonl"
TRAIN_FILE = DATA_ROOT / "multinli_1.0_train.jsonl"

PDEV_MMA_FILE = DATA_ROOT / "pre_multinli_1.0_dev_mismatched.jsonl"
PDEV_MA_FILE = DATA_ROOT / "pre_multinli_1.0_dev_matched.jsonl"
PTRAIN_FILE = DATA_ROOT / "pre_multinli_1.0_train.jsonl"

SAVE_MODEL_FOLDER = "cross-bert-comp-maxpool"
LOG_FILE_PATH =  PARAM_PATH / SAVE_MODEL_FOLDER / "train_log.txt"
PURE_TRAIN_STAT_PATH = PARAM_PATH / SAVE_MODEL_FOLDER / "stat.jsonl"

# Preprocssing / Data Config
data_config = {
    "file_path" : DEV_MA_FILE # this should be a param
}
label_to_id = {
    "contradiction" : 0,
    "neutral" : 1,
    "entailment" : 2,
}
h_field = "sentence2"
p_field = "sentence1"
l_field = "gold_label"
label_field = "gold_label"

# MODEL
CROSS_ATTENTION_HIDDEN_SIZE = 392
NUM_CLASSES = 3


# Bert Enbedding
BERT_EMBEDDING = "bert-base-uncased" #cased?
BERT_MAX_INPUT_LEN = 512

# Trainning
BATCH_SIZE = 8
NUM_EPOCHS = 6
LR = 3*0.00001 # 1e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NUM_WARMUP = 100