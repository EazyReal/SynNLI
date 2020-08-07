# GNN source v0

## Alert
- I have stopped developing this src folder
- plz refer to another folder for the newest results
- the way to reproduce the result in this folder is in the next section


## Execution
- follow the last sections in testbed.ipynb
- it contains training and testing code

## notebooks
- GNL_play : play ground
- SynNLI-data.ipynb : stage1 code for data preprocessing and data visualization(text2graph) developmemnt
- testbed.ipynb: stage2 code for batch model and training code development (GraphData, collate by follow_batch, model, and trainer)

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