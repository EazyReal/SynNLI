## util
import os
import logging
import argparse
from tqdm import tqdm_notebook as tqdmnb
from tqdm import tqdm as tqdm
import pickle
import json 
import jsonlines as jsonl
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, Union
from pathlib import Path
## Stanza
import stanza
from stanza.models.common.doc import Document as StanzaDocument
from stanza.pipeline.core import Pipeline as StanzaPipeline
# self

logger = logging.getLogger(__name__)

# ANLI
anli_p_field = "context"
anli_h_field = "hypothesis"
anli_l_field = "label"
anli_id_field = "uid"

# this is the fields (in, out) called by the program
in_fields = [
    "sentence1", # premise 
    "sentence2", # hypothesis
    "gold_label", # label here
    "pairID", # problem id here
]
out_fields = ["sentence1", "sentence2", "gold_label", "id"]
# for HANS, there is only two labels
labels = ["non-entailment", "entailment"]
# for MNLI+HANS, may have to do some trick in the dataset reader (HANSreader)
# labels = ["n", "e", "c"]

"""
defaults usage
python preprocess.py -i "in" -o "out" -m $max_num_of_instances -files "filelist" -f(if want to force exe)
ex:
python preprocess.py -i ../data/anli_v1.0/R2/ \
 -o ../data/anli_v1.0_preprocessed/R2/ \
 --files dev.jsonl test.jsonl train.jsonl \
 --force \
 -m 10
 
src# python preprocess.py -i ../data/anli_v1.0/R1/  -o ../data/anli_v1.0_preprocessed/R1/  --files dev.jsonl test.jsonl train.jsonl  --force 

"""


# process one data
def process_one(data :Dict, parser: StanzaPipeline, f: List, out_f: List) -> Dict:
    ret = {}
    ret[out_f[0]] = parser(data[f[0]]).to_dict()
    ret[out_f[1]] = parser(data[f[1]]).to_dict()
    ret[out_f[2]] = data[f[2]]
    ret[out_f[3]] = data[f[3]]
    return ret

# parse MNLI style dataset with Stanza and save the result
def process_file(data_file : Union[str, Path],
                 target_file : Union[str, Path],
                 parser : StanzaPipeline,
                 input_fields: List,
                 output_fields: List,
                 max_num_of_instances: int = None,
                 force_exe : bool=False) -> Iterable[Dict]:
    """
    input
        (data_file = str, target_file = str)
    effect
        load preprocess models
        preprocess and save data to target_file
    ouput
        preprocessed data
    
    parsed data is in jsonl (each line is a json)
    {
        "id" : id: str
        out_f[0] : p doc: StanzaDoc
        out_f[1] : h doc: StanzaDoc
        out_f[2] : label: str
    }
    """
    # alias
    in_f = input_fields
    out_f = output_fields
    # data_file_loading
    with open(data_file, "r") as fo:
        json_data = [json.loads(line) for line in fo]
        
    if max_num_of_instances is not None:
        logger.info(msg=f"max_num_of_instances is {max_num_of_instances}")
        json_data = json_data[:max_num_of_instances]
        
    # check if already processed
    if os.path.isfile(str(target_file)) and not force_exe:
        logger.critical(msg=f"file {target_file}  already exist")
        logger.critical(msg="if u still want to procceed, add force_exe=True in arg")
        return None
    else:
        logger.info(msg=f"creating {target_file} to save result")
        
    # processing and jsonl saving
    with jsonl.open(target_file, mode='w') as writer:
        parsed_data = []
        for data in tqdm(json_data):
            # only add those who have gold labels
            if(data[input_fields[2]] not in labels):
                continue
            pdata = process_one(data, parser, input_fields, output_fields)
            parsed_data.append(pdata)
            writer.write(pdata)
            
    logger.info(msg=f"parsing compplete, data saved to {target_file}")
    return parsed_data

def init_args(arg_string=None):
    parser = argparse.ArgumentParser(description='Preprocess datasets. Note that you should configure your field names and labels in this preprocess.py')
    parser.add_argument('-i', '--input_dir', dest='input_dir', type=str, default=None, required=True, help='input dir should end with / ')
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default=None, required=True)
    parser.add_argument('--files', dest='files', nargs='+', type=str, default=None, required=True)
    parser.add_argument('-m', '--max_num_of_instances', type=int, default=None)
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args(arg_string)
    return args
               
if __name__ == "__main__":
    args = init_args()
    # handle directory
    if not os.path.isdir(args.input_dir):
        raise InputDirNotExistError
    if os.path.isdir(args.output_dir) and not args.force:
        raise OutputDirExistError
    if not os.path.exists(args.output_dir):
        logger.info(f"making directory {args.output_dir}")
        os.makedirs(args.output_dir) # already checked force
    else:
        logger.info(f"{args.output_dir} exists but forcing execution")
    # parser
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    
    # process files
    for file in args.files:
        if not os.path.isdir(args.input_dir):
            raise InputFileNotExistError
        process_file(data_file=args.input_dir+file,
                     target_file=args.output_dir+file,
                     parser = nlp,
                     max_num_of_instances=args.max_num_of_instances,
                     input_fields=in_fields,
                     output_fields=out_fields,
                     force_exe=args.force)