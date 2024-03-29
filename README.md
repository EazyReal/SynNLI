# SynNLI - Syntax-aware Natural Language Inference

## Description
- this repo uses allennlp as base repo
- see `README_allen_nlp_guide.md` for traning and running

## Custom Classes and Operations
- `GraphPair2VecEncoder`
    - 'gen', 'gmn'
- `Graph2GraphEncoder`
    - known as `graph convolution layer` in `pytorch_geometric` 
- `GraphPair2GraphPairEncoder`
    - for graph matching in sparse batch
    - tf.dynamic_partition + normal attention
- `NodeUpdater`
    - A wrapper over `RNN`s
- `Graph2VecEncoder`
    - known as `global pooling layer` in `pytorch_geometric` 
    - 'global_attention'
- `SynNLIModel(base=Model)`
    - use `Embedder` to embed input
    - use `GraphPair2VecEncoder` to get compare vector for classifier to make final decision
- `tensor_op.py`
    - batch conversion between normal model and graph model
        - sparse2dense
        - dense2sparse
- `SparseAdjacencyField`
    - cooperate with `pytorch_geometric` to get sparce graph batch
    - see `batch_tensors()` and `as_tensor()` for the key of implementation
- `NLIGraphReader`
    - read graph input (parsed by `Stanza`)
- `preprocess.py`
    - see the `Preprocess` section for detail
- `configs`
    - can be found in `src/training`
    - for allennlp train

## Usage (2020)
- ./install_dependencies.sh 
- download NLI style data set to data
    - and specify path in jsonnet
- parse data (see Parse Data section)
    - and specify path in jsonnet
- train model (see Training Area)
    - with jsonnet
- in 2023, this will not work properly (see `2023install.md`)


## Parse Data with Stanza
- Stanza will be loaded in preprocess.py
    - the parser version is the one @ 2020/8/22
- use preprocess.py
```
python preprocess.py -i <raw_data_path> \
 -o <target_path> \
 --files <file_names> \
 --force(if activated, force execution when <target_path exists>) \
 -m 10(if provided, maximum instances to process is set, this is mainly for testing)
```
```
# example
python preprocess.py -i ../data/anli_v1.0/R2/ \
 -o ../data/anli_v1.0_preprocessed/R2/ \
 --files dev.jsonl test.jsonl train.jsonl \
 --force \
 -m 10
```
- if want to use allennlp (less recommended)
    - download allennlp dependency parser and SRL labeler from path

## Training
- refer to "the config.jsonnet"
```
allennlp train "./src_gmn/training_config.jsonnet" -s "./param/testv1"   --include-package "package_v1" --force
```

## Future Supported Usage
- pip install -r requirements
- + add configs folder for various config
- note that should take lemmatized as node attr if use word level embedding(or + char embedding to ease)
- root to spetial token
- use MLP prjection

