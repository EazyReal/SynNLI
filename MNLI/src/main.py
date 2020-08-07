################################################################################
# Desciption

# author = ytlin
# decription = main program for preprocess + load model + train + eval + use
################################################################################

################################################################################
# Usage

# python -m main.py --do_train=true

"""
pip install -r requirements.txt
python -m main.py \
  --data_dir=... \
  --output_dir=... \
  --model_name=... \
  --model_file=... \
  --model_check_point=...\
  --do_train \
  --do_eval \
  --do_predict
  
visit config.py for other settings 
config.LOG control
"""
################################################################################


################################################################################
# Dependencies
################################################################################

# project
import config
from preprocess import *
from model import *
from utils import *
from train import *

# external
from transformers import BertModel
import torch.nn as nn
import math # for sqrt
import argparse


################################################################################
# ARGUMENTS
################################################################################
def init_args(arg_string=None):
    parser = argparse.ArgumentParser()
    
    # ACTION
    parser.add_argument('--do_train', type=lambda x: (x.lower() == 'true'),
                        default=False)
    #parser.add_argument('--train_file', nargs='*', type=str)
    parser.add_argument('--do_eval', type=lambda x: (x.lower() == 'true'),
                        default=False)
    #parser.add_argument('--eval_file', nargs='*', type=str)
    parser.add_argument('--do_predict', type=lambda x: (x.lower() == 'true'),
                        default=False)
    #parser.add_argument('--predict_file', nargs='*', type=str)
    # FILE PATH
    # MODEL PARAMETERS
    # DATA PREPROCESSING
    # PARAMETERS FOR TRAINING
    # PARAMETERS FOR PREDICTING

    args = parser.parse_args(arg_string)
    
    args.err_stream = sys.stdout
    args.train_stream = open(config.LOG_FILE_PATH, mode='w')

    # DEFAULT TO THE MULT-LABEL MODE
    """
    if args.do_train and len(args.train_file) == 0:
        raise ValueError('"do_train" is set but no "train_file" is given.')
    if args.do_eval and len(args.eval_file) == 0:
        raise ValueError('"do_eval" is set but no "eval_file" is given.')
    if args.do_predict and len(args.predict_file) == 0:
        raise ValueError('"do_predict" is set but no "predict_file" is given.')
        
    model_config_path = os.path.join(
        args.model_name_or_path, ARGS_FILE_NAME)
    if os.path.exists(model_config_path):
        with open(model_config_path) as f:
            model_config = json.load(f)
        for key, val in model_config.items():
            setattr(args, key, val)


    # DEVICE SETTING
    if torch.cuda.is_available() and not args.force_cpu:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # ERROR STREAM
    if args.err_to_dev_null:
        args.err_stream = open(os.path.devnull, mode='w')
    else:
        args.err_stream = sys.stderr
    """

    return args

################################################################################
# THE MAIN FUNCTION
################################################################################
def main():
    args = init_args()
    
    print('CREATING TOKENIZER...', file=args.err_stream)
    tokenizer = BertTokenizer.from_pretrained(config.BERT_EMBEDDING)

    print('CREATING MODEL...', file=args.err_stream)
    model = CrossBERTModel()
    
    if args.do_train:
        # trainer will use gpu if avaiable
        trainer = CrossBERT_Trainer(model=model)
        # None = default
        trainer.train(train_set=None,
                      dev_set=None,
                      model_file_path=config.SAVE_MODEL_FOLDER,
                      log_stream=args.train_stream)

    if args.do_eval:
        pass

    if args.do_predict:
        predict_data = pd.DataFrame()
        pass

if __name__ == "main":
    main()