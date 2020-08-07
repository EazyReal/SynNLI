"""
past arg parse

def init_args(arg_string=None):
    parser = argparse.ArgumentParser()

    # FILE PATHS
    parser.add_argument('--model_check_point', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='roberta_model')
    
    # ACTION
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
"""



"""
main starter
python -m main.py \
  --data_dir=... \
  --output_dir=... \
  --model_name=... \
  --model_file=... \
  --model_check_point=...\
  --model_config_file=... \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length=128 \
  --optimizer=adamw \
  --task_name=MNLI \
  --warmup_step=1000 \
  --learning_rate=3e-5 \
  --train_step=10000 \
  --save_checkpoints_steps=100 \
  --train_batch_size=128


if args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device=args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
        
    if args.do_train:
        # trainer will use gpu if avaiable
        trainer = CrossBERT_Trainer(model=model)

    if args.do_eval:
        pass

    if args.do_predict:
        predict_data = pd.DataFrame()
        pass
"""



"""
old collate

for f in [config.h_field, config.p_field]:
    tokens_tensors = [s[f] for s in samples]
    segments_tensors = [torch.tensor([0] * len(s[f]), dtype=torch.long) for s in samples]
    # zero pad to same length
    tokens_tensors = pad_sequence(tokens_tensors,  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,  batch_first=True)
    # attention masks, set none-padding part to 1 for LM to attend
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill( tokens_tensors != 0, 1)
    batch[f] = {
        "tokens_tensors" : tokens_tensors,
        "segments_tensors" :  segments_tensors,
        "masks_tensors" : masks_tensors
    }
"""

"""
debug tensor
if config.DEBUG:
    print("WK.size is ", self.Wk.size())
    print("WK is ", self.Wk)
    print("h2.size is ", h2.size())
    print("h2 is ", h2)
    print("K.size is ", K.size())
    print("K is ", K)
"""

##############################################################
# for bert embedding, self tokenized                                         #
##############################################################

# note, this should be called after tokenizer is assigned
def sent_to_tensor(s, tokenizer):
    assert(tokenizer is not None)
    
    tokens = tokenizer.tokenize(s)
    if(len(tokens) > config.BERT_MAX_INPUT_LEN-2):
        if config.DEBUG:
            print("a sentence: \n" + s + "\n is truncated to fit max bert len input size.")
        tokens = tokens[:BERT_MAX_INPUT_LEN-2]
    tokens.insert(0, "[CLS]")
    tokens.append("SEP")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    tensor = torch.tensor(ids)
    return tensor

# for visualization 
def tensor_to_sent(t, tokenizer):
    assert(tokenizer is not None)
    tokens = tokenizer.convert_ids_to_tokens(t)
    sent = " ".join(tokens)
    return sent

# for CrossBERT dataset
def CrossBERT_preprocess(raw_data, tokenizer=None):
    #tokenizer
    if tokenizer == None:
        tokenizer = BertTokenizer.from_pretrained(config.BERT_EMBEDDING)
    else:
        tokenizer = tokenizer 
    
    # filed alias
    pf = config.p_field
    hf = config.h_field
    lf = config.label_field
    
    processed_data = list()
    maxlen = {pf: 0, hf : 0}
    
    ## to tensor and get maxlen
    for instance in raw_data:
        # label
        l = instance[lf]
        if(l not in config.label_to_id.keys()):
            continue
        # storage
        processed_instance = {
            pf: sent_to_tensor(instance[pf], tokenizer),
            hf: sent_to_tensor(instance[hf], tokenizer),
            lf: torch.tensor(config.label_to_id[l], dtype=torch.long)
        }
        processed_data.append(processed_instance)
    
    # padding no here
    return processed_data


class MNLI_CrossBERT_Dataset(Dataset):
    """
    MNLI set for CrossBERT baseline
    source: 
    wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip @ 2020/7/21 17:09
    self.j_data is list of jsons
    self.raw_data is list of (hyposesis, premise, gold label)
    # self.tensor_data is list of tensored data (generate by sent_to_tensor for bert)
    """
    def __init__(self,
                 file_path=config.DEV_MA_FILE,
                 mode="develop",
                 process_fn=CrossBERT_preprocess,
                 tokenizer=None,
                 data_config=config.data_config,
                 save=False):
        # super(MNLI_CrossBERT_Dataset, self).__init__()
        # decide config
        self.mode = mode
        
        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_EMBEDDING)
        else:
            self.tokenizer = tokenizer 
        
        # load raw data
        self.file_path = file_path
        with open(self.file_path) as fo:
            self.raw_lines = fo.readlines()
        # to json
        self.j_data = [json.loads(line) for line in self.raw_lines]
        self.tensor_data = process_fn(self.j_data, self.tokenizer)
        return None
        
        
    def __getitem__(self, index):
        return self.tensor_data[index]
        
    def __len__(self):
        return len(self.tensor_data)
    
##############################################################
