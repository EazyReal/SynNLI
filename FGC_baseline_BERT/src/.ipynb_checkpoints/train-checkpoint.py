from tqdm import tqdm_notebook as tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import os

from dataset import * 
from model import * 
import config as config

TEST = True
LOG = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# (instance/batchsize)*epcho = # batch
BATCH_SIZE = 8
NUM_EPOCHS = 6
LR = 0.00001 # 1e-5
WEIGHT_DECAY = 0.01

NUM_WARMUP = 100

# Load Data using dataset.py

class SER_Trainer:
    def __init__(self, model):
        self.model = model
    
    def eval(self, )
    
    def train(self, model, train_set=None, dev_set=None, batch_size, collate_fn=create_mini_batch, model_file_path):
        if train_set == None:
            train_set = FGC_Dataset(config.FGC_TRAIN, mode="train")
        if dev_set == None:
            dev_set = FGC_Dataset(config.FGC_DEV, mode="develop")
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
        
        # calc pos weight for BCE
        if WEIGHTED_BCE:
        total = 0
        true_cnt = 0
        for instance in train_set:
            if(instance[-1] == True):
                true_cnt += 1
            total += 1
        print(true_cnt)
        print(total)
        # to increase the value of recall in the model's criterion
        pos_weight = print(torch.tensor([(total-true_cnt)/true_cnt, 1]))
        print(pos_weight)
        # no need to applied pos_weight = torch.tensor([total/true_cnt, total/(1-true_cnt)])?
        
        bert_encoder = BertModel.from_pretrained(config.BERT_EMBEDDING)
        
        model = BertSERModel(bert_encoder=bert_encoder, pos_weight=pos_weight)
        model.to(device) # means model = model.to(device)
        if LOG:
            print("model in cuda?", next(model.parameters()).is_cuda)
        
        # saving directory
        model_file_path = "baseline"
        save_model_path = config.PARAM_PATH / model_file_path
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
            
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        # optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
        num_train_optimization_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=NUM_WARMUP,
                                                    num_training_steps=num_train_optimization_steps)
        
        
        print('start training ... ')

        for epoch_i in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for batch_i, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                batch = [data.to(device) for data in batch]

                loss = model(batch)

                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
            learning_rate_scalar = scheduler.get_lr()[0]
            print('lr = %f' % learning_rate_scalar)
            print('epoch %d train_loss: %.3f' % (epoch_i, running_loss / len(dataloader_train)))
            if epoch_i % eval_epoch_frequency == 0:
                self.eval(dev_batches, epoch_i, trained_model_path)
        return 
    
    
    