# GNN source v0

## Overview
- the src folder (src-gnn) is no longer updating (too dirty and cannot use BERT as base embedding easily)
- the way to reproduce the result in this folder is in the next sections
- refer to src_oop for new model in allennlp style

## Usage 

### Trained Model Probing
- follow the execution will do
- probing.ipynb

### Training from scratch
- follow the last sections in testbed.ipynb
- it contains training and testing code

## notebooks
- GNL_play : play ground
- SynNLI-data.ipynb : stage1 code for data preprocessing and data visualization(text2graph) developmemnt
- testbed.ipynb: stage2 code for batch model and training code development (GraphData, collate by follow_batch, model, and trainer)
- probing.ipynb: loading and probing/evaluating trained model

## src orginization
- main.py: todo, the execution code
- config.py: settings(all in here)
- utils.py : from changers, preprocess
- model.py : model
- data.py : dataset definition
- train.py : for training code
- dunmped_code: previous versions of useful functions

## todo
- change label view style code to real batch code
    - study DataLoader
- cut the pipeline more clearly
- config orginizing
    - using objects
- config.PURE_STAT should be filename, filepath should be able to be determine by train function


---

# Implementation of a Smaller Model

## Implementation of a Smaller Model - model
<!-- .slide: style="font-size: 26px;" -->
- model
    - Input Layer: Stanza dependecy parsing(tokenize is done in this step)
    - Embedding Layer: GLOVE embedding (300d)
    - Graph Encoder Layer: dependency + Graph encoder(GAT * 3 layers)
    - Alignment Layer: Cross Attention 
    - Local Comparison Layer: vector cmp$(h, p, h-p, h\odot p)$ -> feed forward(2 layer)
    - Final Judgement Layer: mean and max poolings + feed forward(2 layer)
- code available at [github repo folder](https://github.com/EazyReal/2020-IIS-internship/tree/master/MNLI/src-gnn)
    - with a brief instruction to execute

----

## Implementation of a Smaller Model - data 1
<!-- .slide: style="font-size: 26px;" -->

![](https://i.imgur.com/9UwLSlv.png)


----

## Implementation of a Smaller Model - data 2
<!-- .slide: style="font-size: 26px;" -->

![](https://i.imgur.com/6CnWfb2.png)


----

## Implementation of a Smaller Model - training details
<!-- .slide: style="font-size: 26px;" -->
- dataset - MNLI
    - 433 k examples in total 
    - overall balanced in 3 labels
- training details
    - Optimizer = transformers AdamW
    - BATCH_SIZE = 32
    - NUM_EPOCHS = 5
    - LR = 5e-4
    - WEIGHT_DECAY = 0.01
    - MAX_GRAD_NORM = 1.0
    - NUM_WARMUP = 100
    - DROUP_OUT_PROB = 0.1

----

## Implementation of a Smaller Model - performance
<!-- .slide: style="font-size: 26px;" -->
- train once 
- ~ 1 hr/epoch, 5 epcho in toal
- todo: add param size and inference speed
- macro f1 ~= acc in my model

| model\acc | Train  | Matched | MisMatched |
| -------------- | ------ | ------- | ---------- |
| RoBERTa        | -      | 90.8    | 90.2       |
| aESIM          | -      | 73.9    | 73.9       |
| Glove-GAT3     | 0.6323 | 0.6116  | 0.6252     |
| random/majority     | ~0.33 | ~0.33  |~0.33     |


----

## Implementation of a Smaller Model - probing 1
<!-- .slide: style="font-size: 26px;" -->

- [My Model Probing](/itHxmpLSTYqjobgc4PFKVw)
- naive
    - negation - $\text{ A is B v.s. A is not B}$
        - $\text{ A is not B v.s. A is not B}$
        - isn't is tested OK too
    - double negation - $\text{ A is B v.s. A is not not B}$
    - syntax relation - $\text{ from A to B v.s. from B to A}$
- +noise
    - long - $\text{ C is D C is D A is B  |  C is D C is D A is not B}$
    - change order - $\text{ A is B C is D C is D  |  C is D C is D A is not B}$
        - win roberta
    - extra twitch
        - not happy vs not unhappy

----

## Implementation of a Smaller Model - probing 2
<!-- .slide: style="font-size: 26px;" -->
| purpose\variation    | Naive | + Long Seq | + Change Order | +Extra Twitch |
| -------------------- | ----- | ---------- | -------------- | ------------- |
| Negation             | Y     | Y          | Y              | N             |
| Double Negation      | N     | -          | -              | -             |
| Syntax Relation      | Y     | N          | -              | -             |
| Over Causuality      | Y     | Y          | Y              | -             |
| Nane Entity Mismatch | N     | -          | -              | -             |


----

## Implementation of a Smaller Model - discussion (performance)
<!-- .slide: style="font-size: 24px;" -->
- performance 
    - training set acc ~= development set acc
        - catastrophic forgetting (training set acc is low)
        - model generize well 
    - both acc are low
        - main reason - model complexity is low
        - only use 300d glove without biGRU contextulization
            - use 3 layer or "directed GAT" instead...
        - ESIM and following model apply biLSTM for local comparison 
    - consider
        - use BERT encoder as base embedding
        - add SRL for semantic role info
        - use LSTM(in sentence order) or tree lstm by dependency to do local compare

----

## Implementation of a Smaller Model - discussion (probing)
<!-- .slide: style="font-size: 24px;" -->
- probing
    - model can indeed catch syntatic issue!
        - dependency can help
    - model can catch negation! but not double negation
        - consider SRL info (negation tag?), adding Q-edge(Edge for Quantifier), math
    - model cannot catch adj => neutral
    - model errors most come from word sense
        - consider using bert
    - but still judge by similarity and can not determine syntax relation if sequence is long (noisy)
        - consider maxpool + lstm(gating)

---