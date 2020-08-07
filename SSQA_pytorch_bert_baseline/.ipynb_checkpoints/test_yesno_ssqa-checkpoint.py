"""Codes for FGC YesNo Module"""

import argparse
import json
import math
import os
import random
import sys
from pprint import pprint
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import transformers
from transformers import AdamW, BertConfig, BertModel, BertTokenizer

#from sklearn.mertrics import precision_recall_fscore_support


ARGS_FILE_NAME = 'yesno_config.json'
MODEL_FILE_NAME = 'yesno_model.pt'


################################################################################
# ANSWER TEXT FORMS
################################################################################
# THE FIRST ONE FROM EACH LIST WILL BE USED FOR THE OUTPUT
# ALL OF THEM WILL BE USED WHEN READING TRAINING FILES
YES_ANSWERS = [1]
NO_ANSWERS = [0]


################################################################################
# ARGUMENTS
################################################################################
def init_args(arg_string=None):
    parser = argparse.ArgumentParser()

    # FILE PATHS & ACTIONS
    parser.add_argument('--model_name_or_path', type=str,
                        default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--output_dir', type=str, default='roberta_model')
    
    parser.add_argument('--do_train', type=lambda x: (x.lower() == 'true'),
                        default=False)
    parser.add_argument('--train_file', nargs='*', type=str)
    parser.add_argument('--do_eval', type=lambda x: (x.lower() == 'true'),
                        default=False)
    parser.add_argument('--eval_file', nargs='*', type=str)
    parser.add_argument('--do_predict', type=lambda x: (x.lower() == 'true'),
                        default=False)
    parser.add_argument('--predict_file', nargs='*', type=str)
    # MODEL PARAMETERS
    parser.add_argument('--max_seq_length', type=int, default=512)
    # DATA PREPROCESSING
    parser.add_argument('--sup_evidence_as_passage', action='store_true')
    parser.add_argument('--max_window_slide_dist', type=int, default=128)
    # PARAMETERS FOR TRAINING
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--train_batch_size', type=int, default=6)
    parser.add_argument('--train_epochs', type=int, default=4)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--save_epochs', type=int, default=1)
    # PARAMETERS FOR PREDICTING
    parser.add_argument('--predict_batch_size', type=int, default=6)
    # OTHERS
    parser.add_argument('--err_to_dev_null', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)

    args = parser.parse_args(arg_string)

    # DEFAULT TO THE MULT-LABEL MODE
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

    return args


################################################################################
# DATASET CLASS
################################################################################
class ssqaDataset(torch.utils.data.IterableDataset):
    def __init__(self, args, input_data, tokenizer, is_training=False):
        super(ssqaDataset, self).__init__()

        self.example_list = []
        self.feature_list = []
        for row in tqdm(input_data.iterrows(), desc='LOADING DATA', file=args.err_stream):
            qid = row[1][0]+'-'+row[1][1]+'-'+row[1][2]
            ans_label = -1
            if is_training:
                # ANSWER LABELS
                label = row[1][5]
                if label in YES_ANSWERS:
                    ans_label = 0
                elif label in NO_ANSWERS:
                    ans_label = 1
                else:
                    print(f'ssqaDataset: Question {qid} has no normal yes/no answer.', file=args.err_stream)
                    if is_training:
                        continue

            # QUESTION TEXT
            try:
                qtext = row[1][4]
            except KeyError:
                qtext = ''
                
#                 if '是否' in qtext:
#                     qtext = qtext.replace('是否', '')
#                 elif '是不是' in qtext:
#                     qtext = qtext.replace('是不是', '')
#                 elif '好不好' in qtext:
#                     qtext = qtext.replace('好不好', '好')
#                 elif '有沒有' in qtext:
#                     qtext = qtext.replace('有沒有', '')
#                 elif '吗' in qtext:
#                     qtext = qtext.replace('吗', '')
#                 else:
#                     pass

#                 qtext = qtext.replace('?', '。')
#                 qtext = qtext.replace('？', '。')
            qtext_tokens = tokenizer.tokenize(qtext)
            dtext_max_length = args.max_seq_length - len(qtext_tokens) - 3


            # PASSAGE TEXT
            try:
                dtext = row[1][3]
            except KeyError:
                dtext = ''
            dtext_tokens = tokenizer.tokenize(dtext)


            # CONCAT INPUT
            while True:
                dtext_length = min(len(dtext_tokens), dtext_max_length)
                input_tokens = ([tokenizer.cls_token]
                                + qtext_tokens
                                + [tokenizer.sep_token]
                                + dtext_tokens[0:dtext_length]
                                + [tokenizer.sep_token])
                input_tokens += [tokenizer.pad_token] * \
                    (args.max_seq_length - len(input_tokens))
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                assert len(input_ids) == args.max_seq_length

                attention_mask = [1] * (len(qtext_tokens) + dtext_length + 3)
                attention_mask += [0] * (args.max_seq_length - len(attention_mask))
                assert len(attention_mask) == args.max_seq_length

                token_type_ids = [0] * (len(qtext_tokens) + 2)
                token_type_ids += [1] * (dtext_length + 1)
                token_type_ids += [0] * \
                    (args.max_seq_length - len(token_type_ids))
                assert len(token_type_ids) == args.max_seq_length
                
                example = {'qid': qid,
                            'dtext': dtext,
                            'qtext': qtext,
                            'input_tokens': input_tokens,
                            'ans_labels': ans_label}
                self.example_list.append(example)
                feature = {'input_ids': torch.tensor(input_ids),
                            'attention_mask': torch.tensor(attention_mask),
                            'token_type_ids': torch.tensor(token_type_ids),
                            'ans_labels': torch.tensor(ans_label, dtype=torch.long)}
                self.feature_list.append(feature)

                if dtext_length < len(dtext_tokens):
                    window_slide_dist = min(
                        args.max_window_slide_dist, len(dtext_tokens) - dtext_length)
                    dtext_tokens = dtext_tokens[window_slide_dist:]
                else:
                    break

        if is_training:
            zipped_list = list(zip(self.example_list, self.feature_list))
            random.shuffle(zipped_list)
            self.example_list = [x[0] for x in zipped_list]
            self.feature_list = [x[1] for x in zipped_list]

    def __iter__(self):
        
        return iter(self.feature_list)


################################################################################
# THE MODEL
################################################################################
class YesNoModel(nn.Module):
    def __init__(self, args):
        super(YesNoModel, self).__init__()

        yesno_weight_path = os.path.join(
            args.model_name_or_path, MODEL_FILE_NAME)
        if os.path.exists(yesno_weight_path):
            bert_config = BertConfig.from_pretrained(args.model_name_or_path)
            bert_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
            self.bert = BertModel(bert_config)
        else:
            self.bert = BertModel.from_pretrained(args.model_name_or_path)
            self.bert.config.attention_probs_dropout_prob = args.attention_probs_dropout_prob 
        hidden_size = self.bert.config.hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        if os.path.exists(yesno_weight_path):
            self.load_state_dict(torch.load(yesno_weight_path, map_location=args.device),
                                 strict=False)

    def forward(self, features):
        bert_output = self.bert(input_ids=features['input_ids'],
                                attention_mask=features['attention_mask'],
                                token_type_ids=features['token_type_ids'])
        bert_cls_hidden = bert_output[1]

        output = self.fc1(bert_cls_hidden)
        output = self.fc2(output)

        return output


################################################################################
# TRAIN
################################################################################
def train(args, model, tokenizer, input_data):
    train_dataset = ssqaDataset(args, input_data, tokenizer, is_training=True)
    data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.train_batch_size,
                                              pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    steps_per_epoch = math.ceil(
        len(train_dataset.feature_list) / args.train_batch_size)
    num_training_steps = steps_per_epoch * args.train_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    model.train()
    for epoch in trange(args.train_epochs, desc='EPOCHS', file=args.err_stream):
        #train_loss = 0.0
        
        for data_batch in tqdm(data_loader, desc='STEPS', total=steps_per_epoch, leave=False, file=args.err_stream):
            for key, feature in data_batch.items():
                data_batch[key] = feature.to(args.device)
            optimizer.zero_grad()

            output = model(data_batch)
            loss = F.cross_entropy(output, data_batch['ans_labels'])

            loss.backward()
            optimizer.step()
            scheduler.step()
            #train_loss += loss.item()*len(data_batch)
        #train_loss = train_loss/11125
        #print('train_loss: ', train_loss)

        if (args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0) or (epoch + 1 == args.train_epochs):
            save_dir = os.path.join(args.output_dir, f'epoch_{epoch + 1:02d}')
            save_model(args, model, tokenizer, save_dir)

    return model


def save_model(args, model, tokenizer, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, MODEL_FILE_NAME))
    model.bert.config.to_json_file(os.path.join(save_dir, transformers.CONFIG_NAME))
    tokenizer.save_vocabulary(save_dir)

    param_list = ['max_seq_length']
    config_json = dict()
    for param in param_list:
        try:
            config_json[param] = getattr(args, param)
        except AttributeError as e:
            print('save_model: ', str(e), file=args.err_stream)
    with open(os.path.join(save_dir, ARGS_FILE_NAME), mode='w') as f:
        json.dump(config_json, f, indent=4)


################################################################################
# PREDICT
################################################################################
def predict(args, model, tokenizer, input_data):
    predict_dataset = ssqaDataset(
        args, input_data, tokenizer, is_training=False)
    data_loader = torch.utils.data.DataLoader(predict_dataset,
                                              batch_size=args.predict_batch_size,
                                              pin_memory=True)
    model.eval()

    steps_per_epoch = math.ceil(
        len(predict_dataset.feature_list) / args.predict_batch_size)
    prediction_list = []
    with torch.no_grad():
        for data_batch in tqdm(data_loader, desc='STEPS', total=steps_per_epoch, file=args.err_stream):
            for key, feature in data_batch.items():
                data_batch[key] = feature.to(args.device)

            output = model(data_batch)
            output = F.softmax(output, dim=1)

            prediction_list += output.tolist()


    # PREDICTION POST-PROCESSING
    prediction_dict = dict()
    for example, prediction in zip(predict_dataset.example_list, prediction_list):
        qid = example['qid']
        prediction_dict.setdefault(qid, [])
        prediction_dict[qid].append(prediction)
        

    final_predictions = dict()
    for qid in tqdm(prediction_dict.keys(), desc='PROCESSING PREDICTIONS', file=args.err_stream):
        mean_prob = [math.fsum([x[0] for x in prediction_dict[qid]]) / len(prediction_dict[qid]),
                     math.fsum([x[1] for x in prediction_dict[qid]]) / len(prediction_dict[qid])]
        final_predictions[qid] = [
            {'ATEXT': YES_ANSWERS[0],
             'score': mean_prob[0],
             'start_score': 0.0,
             'end_score': 0.0,
             'AMODULE': "YesNo"},
            {'ATEXT': NO_ANSWERS[0],
             'score': mean_prob[1],
             'start_score': 0.0,
             'end_score': 0.0,
             'AMODULE': "YesNo"}]

    return final_predictions


################################################################################
# EVALUATE
################################################################################
def eval(args, model, tokenizer, input_data):
       
    final_predictions = predict(args, model, tokenizer, input_data)
    
    question_count = 0
    correct_count = 0
    all_qid = []
    correct_qid = []
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for row in input_data.iterrows():
        question_count += 1
        qid = row[1][0]+'-'+row[1][1]+'-'+row[1][2]
        all_qid.append(qid)
        top_prediction = max(final_predictions[qid], key=lambda x: x['score'])
        label = row[1][5]
        #for answer_text_set in [YES_ANSWERS, NO_ANSWERS]:
        if (top_prediction['ATEXT'] in YES_ANSWERS and label in YES_ANSWERS):
            correct_count += 1
            correct_qid.append(qid)
            TP += 1
        elif (top_prediction['ATEXT'] in YES_ANSWERS and label in NO_ANSWERS):
            FP += 1
        elif (top_prediction['ATEXT'] in NO_ANSWERS and label in YES_ANSWERS):
            FN += 1
        elif (top_prediction['ATEXT'] in NO_ANSWERS and label in NO_ANSWERS):
            correct_count += 1
            correct_qid.append(qid)
            TN += 1
    
    
    precision =  0.0 if TP == 0 else TP/(TP+FP) 
    recall =  0.0 if TP == 0 else TP/(TP+FN) 
    #beta = 1
    f1 = 0.0 if TP == 0 else 2*precision*recall / (precision + recall)
    
    #modified by ytlin + precision, recall, F1, EM is accuracy
    eval_result = {'question_count': question_count,
                   'correct_count': correct_count,
                   'accuracy': correct_count / question_count,
                   'precision': precision,
                   'recall': recall,
                   'f1': f1,
                  }
    return eval_result, list(set(all_qid) - set(correct_qid))
    #return eval_result, correct_qid, list(set(all_qid) - set(correct_qid))
    #return eval_result
        

################################################################################
# THE MAIN FUNCTION
################################################################################
def main():
    args = init_args()

    print('CREATING TOKENIZER...', file=args.err_stream)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    print('CREATING MODEL...', file=args.err_stream)
    model = YesNoModel(args)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device=args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        train_data = pd.DataFrame()

        if args.train_file[0].endswith(".tsv"):
            train_data = pd.read_csv(args.train_file[0], sep='\t', header=0, index_col='id')
        elif args.train_file[0].endswith(".csv"):
            train_data = pd.read_csv(args.train_file[0], sep=',', header=0, index_col='id')
        
        print('TRAINING...', file=args.err_stream)
        train(args, model, tokenizer, train_data)

    if args.do_eval:
        eval_data = pd.DataFrame()

        if args.eval_file[0].endswith(".tsv"):
            eval_data = pd.read_csv(args.eval_file[0], sep='\t', header=0, index_col='id')
        elif args.eval_file[0].endswith(".csv"):
            eval_data = pd.read_csv(args.eval_file[0], sep=',', header=0, index_col='id')
        
        print('EVALUATING...', file=args.err_stream)
        #eval_result, correct_qid, error_qid= eval(args, model, tokenizer, eval_data)
        eval_result, error_qid= eval(args, model, tokenizer, eval_data)
        #eval_result = eval(args, model, tokenizer, eval_data)
        
        print(eval_result)
        #print("correct: ", correct_qid)
        print("error: ", error_qid)


    if args.do_predict:
        predict_data = pd.DataFrame()

        if args.predict_file[0].endswith(".tsv"):
            predict_data = pd.read_csv(args.predict_file[0], sep='\t', header=0, index_col='id')
        elif args.predict_file[0].endswith(".csv"):
            predict_data = pd.read_csv(args.predict_file[0], sep=',', header=0, index_col='id')
        
        print('PREDICTING...', file=args.err_stream)
        final_predictions = predict(args, model, tokenizer, predict_data)

        print('WRITING PREDICTIONS...', file=args.err_stream)
        prediction_file_path = os.path.join(
            args.output_dir, 'predictions.json')
        with open(prediction_file_path, mode='w') as f: #modified by ytlin 
            json.dump(final_predictions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
