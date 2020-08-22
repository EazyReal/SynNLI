## util
import os
import logging
from argparse import ArgumentParser
from tqdm import tqdm_notebook as tqdmnb
from tqdm import tqdm as tqdm
import pickle
import json 
import jsonlines as jsonl
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, Union
from pathlib import Path
## graph
import networkx as nx
import matplotlib.pyplot as plt
## nn
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data.data import Data
## model
from allennlp_models.structured_prediction.predictors.srl import SemanticRoleLabelerPredictor
from allennlp_models.structured_prediction.predictors.biaffine_dependency_parser import BiaffineDependencyParserPredictor
from allennlp.predictors.predictor import Predictor #
from stanza.models.common.doc import Document

## self
import config

# parse one example of MNLI style data
def process_one(data:Dict, dependency_parser:BiaffineDependencyParserPredictor) -> Dict:
    ret = {}
    ret[config.idf] = data[config.idf]
    ret[config.pf] = { "text": data[config.pf],
                       "dep": dependency_parser.predict(sentence=data[config.pf]),
                     }
    ret[config.hf] = { "text": data[config.hf],
                       "dep": dependency_parser.predict(sentence=data[config.hf]),
                     }
    ret[config.lf] = config.label_to_id[data[config.lf]]
    return ret

# parse the MNLI style data with Stanza and save the result
def process_file(data_file :  Union[str, Path],
                 target_file :  Union[str, Path],
                 srl_labeler_model :  Union[str, Path, SemanticRoleLabelerPredictor]=config.SRL_LABELER_MODEL,
                 dep_parser_model :  Union[str, Path, BiaffineDependencyParserPredictor]=config.DEP_PARSER_MODEL,
                 function_test : bool=False, force_exe : bool=False) -> Iterable[Dict]:
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
        config.idf : id(in string)
        config.p/hf : "text", "dep", "srl"
        config.lf : int 
    }
    """
    # alias
    p = config.pf
    h = config.hf
    l = config.lf
    
    # parser loading (or direct pass)
    if isinstance(srl_labeler_model, SemanticRoleLabelerPredictor):
        predictor_srl = srl_labeler_model
    else:
        predictor_srl = Predictor.from_path(srl_labeler_model)
    
    if isinstance(dep_parser_model, BiaffineDependencyParserPredictor):
        predictor_dep = dep_parser_model
    else:
        predictor_dep = Predictor.from_path(dep_parser_model)
    
    # data_file_loading
    with open(data_file, "r") as fo:
        json_data = [json.loads(line) for line in fo]
        
    if function_test:
        json_data = json_data[:10]
        
    # check if already processed
    if os.path.isfile(str(target_file)) and not force_exe:
        print("file " + str(target_file) + " already exist")
        print("if u still want to procceed, add force_exe=True in function arg")
        print("exiting")
        return None
    else:
        print("creating file " + str(target_file) + " to save result")
        print("executing")
        
    # processing and jsonl saving
    with jsonl.open(target_file, mode='w') as writer:
        parsed_data = []
        for data in tqdm(json_data):
            # only add those who have gold labels
            if(data[l] not in config.label_to_id.keys()):
                continue
            pdata = process_one(data, dependency_parser=predictor_dep, srl_parser=predictor_srl)
            parsed_data.append(pdata)
            writer.write(pdata)
        
    return parsed_data

if __name__ == "__main__":
    predictor_srl = Predictor.from_path(config.SRL_LABELER_MODEL)
    predictor_dep = Predictor.from_path(config.DEP_PARSER_MODEL)
    
    process_file(data_file=config.TRAIN_FILE,
                 target_file=config.P_TRAIN_FILE,
                 srl_labeler_model=predictor_srl,
                 dep_parser_model=predictor_dep,
                 function_test=False,
                 force_exe=True)
    process_file(data_file=config.DEV_FILE,
                 target_file=config.P_DEV_FILE,
                 srl_labeler_model=predictor_srl,
                 dep_parser_model=predictor_dep,
                 function_test=False,
                 force_exe=True)
    process_file(data_file=config.TEST_FILE,
                 target_file=config.P_TEST_FILE,
                 srl_labeler_model=predictor_srl,
                 dep_parser_model=predictor_dep,
                 function_test=False,
                 force_exe=True)