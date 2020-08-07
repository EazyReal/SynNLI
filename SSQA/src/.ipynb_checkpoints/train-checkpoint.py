from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import os
from collections import defaultdict

from dataset import * 
from model import * 
import config as config

# print debug message or general message?
TEST = True
LOG = True

# training params  in config

# Load Data using dataset.py

class SER_Trainer:
    # init trainer, if not specify model, will create one with weighted BCE with dataset when call train
    def __init__(self, model=None):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def eval_model_when_training(self, model, dev_set, id2qid):
        model.eval()
        
        dev_loader = DataLoader(dev_set, batch_size=config.BATCH_SIZE, collate_fn=create_mini_batch)
        pred = []
        label = []
        for batch_i, batch in enumerate(dev_loader):
            label.extend(batch[3].cpu()) # batch[3] is label
            batch = [data.to(self.device) for data in batch]
            pred_batch = model._predict(batch)
            pred.extend(pred_batch)
            
        stat = defaultdict(list)
        for idx in range(len(dev_set)):
            qid = id2qid[idx]
            state = label[idx].item()*2 + pred[idx]
            stat[qid].append(state)

        stat2 = list()
        sum_stat =  {
                "precision" : 0.0,
                "recall" : 0.0,
                "F1" : 0.0,
                "accuracy" : 0.0
            }

        for k, v in stat.items():
            TP = sum([1 if ins == 3 else 0 for ins in v])
            TN = sum([1 if ins == 0 else 0 for ins in v])
            FN = sum([1 if ins == 2 else 0 for ins in v])
            FP = sum([1 if ins == 1 else 0 for ins in v])
            
            precision = TP / (TP + FP) if TP+FP > 0 else 0.0
            recall = TP / (TP + FN) if TP+FN > 0 else 0.0
            f1 = 2 * recall * precision / (recall + precision) if  (recall + precision) > 0 else 0.0
            acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

            cur = {
                "precision" : precision,
                "recall" : recall,
                "F1" : f1,
                "accuracy" : acc
            }
            stat2.append(cur)
            for k in cur:
                sum_stat[k] += cur[k]

        # each question weight the sum
        # remain to do is get max if SE is all negetive
        for k in sum_stat:
            sum_stat[k] /= len(stat2)

        return stat2, sum_stat
    
    def train(self, train_set=None, dev_set=None):
        
        # train set loading
        if train_set == None:
            train_set = SSQA_Dataset(config.SSQA_TRAIN, mode="train", tokenizer=None)
        train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, collate_fn=create_mini_batch)
        
        # dev set loading, id2qid
        if dev_set == None:
            dev_set = SSQA_Dataset(config.SSQA_DEV)
            dev_loader = DataLoader(dev_set, batch_size=config.BATCH_SIZE, collate_fn=create_mini_batch)
        id2qid = dev_set.id_to_qid
        
        # build weight for BCE error if required
        if config.DEFAULT_USE_WEIGHTED_BCE is True and self.model == None:
            total = 0
            true_cnt = 0
            for instance in train_set:
                if(instance[-1] == True):
                    true_cnt += 1
                total += 1
            # to increase the value of recall in the model's criterion
            # binary classification => (pos), the past format (pos, 1) was wrong but some how not used in FGC training
            pos_weight = torch.tensor([(total-true_cnt)/true_cnt])
        
        bert_encoder = BertModel.from_pretrained(config.BERT_EMBEDDING)
        
        # model initialization if = None, will aplly pos_weight to BCE
        if self.model == None:
            if config.DEFAULT_USE_WEIGHTED_BCE is True:
                if TEST or LOG:
                    print("apllying BCE error with weight, ", pos_weight)
                self.model = BertSERModel(bert_encoder=bert_encoder, pos_weight=pos_weight)
            else:
                self.model = BertSERModel(bert_encoder=bert_encoder, pos_weight=None)
            
        self.model.to(self.device)
        
        if TEST or LOG:
            print("model in cuda?", next(self.model.parameters()).is_cuda)
            
        # set saving directory
        model_file_path = "baseline_SSQA"
        save_model_path = config.PARAM_PATH / model_file_path

        if not os.path.exists(save_model_path):
            if LOG:
                print("directory {} doesn't exist, creating...".format(save_model_path))
            os.mkdir(save_model_path)
            
        # optimizer
        
        # weight decay scheduler setting
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.WEIGHT_DECAY},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = len(train_loader) * config.NUM_EPOCHS
        
        # AdamW
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.LR)
        # scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=config.NUM_WARMUP,
                                                    num_training_steps=num_train_optimization_steps)
        
        # Check that input does not exist MAX_BERT_LEN
        for data in train_set:
            # print(data[0])
            assert(data[1].shape[0] <= config.BERT_MAX_INPUT_LEN)
            
        
        # alias when migrating code
        model = self.model
        device = self.device
        
        if LOG:
            print('start training ... ')

        stat = {
            "precision" : [],
            "recall" : [],
            "F1" :[],
            "accuracy" : []
        }

        for epoch_i in range(config.NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for batch_i, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                batch = [data.to(device) for data in batch] # batch[0] = ids, batch[1] = ...
                loss = model(batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM) 
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                
            learning_rate_scalar = scheduler.get_lr()[0]
            print('lr = %f' % learning_rate_scalar)
            print('epoch %d train_loss: %.3f' % (epoch_i, running_loss / len(train_loader)))

            eval_epoch_frequency = 1
            if epoch_i % eval_epoch_frequency == 0:
                ecopch_stat_by_questions, epoch_stat = self.eval_model_when_training(model, dev_set, id2qid)
                for k in stat:
                    stat[k].append(epoch_stat[k])
                print('epoch %d eval_recall: %.3f eval_f1: %.3f' % 
                      (epoch_i, epoch_stat['recall'], epoch_stat['F1']))
                model_to_save = model
                torch.save(model_to_save.state_dict(),
                            str(save_model_path / "model_epoch{0}_precision:{1:.3f}_recall:{2:.3f}_f1:{3:.3f}_acc:{4:.3f}.m".
                                   format(epoch_i, epoch_stat['precision'], epoch_stat['recall'], epoch_stat['F1'],
                                          epoch_stat['accuracy'])))
        model.eval()
        return model