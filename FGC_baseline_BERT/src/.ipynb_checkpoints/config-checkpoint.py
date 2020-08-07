import os
from pathlib import Path

# Paths 
SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PROJ_ROOT = SRC_ROOT.parent

DATA_ROOT = PROJ_ROOT / "data"
PARAM_PATH = PROJ_ROOT / "param"

FGC_DEV = DATA_ROOT / "FGC_release_1.7.13" / "FGC_release_all_dev.json"
FGC_TRAIN = DATA_ROOT / "FGC_release_1.7.13" / "FGC_release_all_train.json"
FGC_TEST = DATA_ROOT / "FGC_release_1.7.13" / "FGC_release_all_test.json"

# Bert Enbedding
BERT_EMBEDDING = "bert-base-chinese"
BERT_MAX_INPUT_LEN = 512


# Trainning
BATCH_SIZE = 8
NUM_EPOCHS = 6
LR = 0.00001 # 1e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0