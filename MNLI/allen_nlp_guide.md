---
descroption: "also presentation for NLU lab biweely meeting"
author: "yt lin"
license: "apache"
---


# Modularized Model with AllenNLP

---

## Overview
<!-- .slide: style="font-size: 25px;" -->
- The repo can be viewed as an extention to Allennlp to support some graph NN modules
- Why?
  - abstraction between layers
  - dependency injection to do experiments

![](https://i.imgur.com/QtS2DLJ.png)

----

### Example (config.jsonnet)
![](https://i.imgur.com/yZCApK1.png)

----

### Example (training)
- `allennlp train <config> -s <log_dir> --include-package=<src>`

---

## Custom Modules
![](https://i.imgur.com/IS035wX.png)

----

## With Module Name

![](https://i.imgur.com/pfo5vfx.png =50%x)


----

### SynNLI Model
<!-- .slide: style="font-size: 30px;" -->
- `Embedder`
    - map tokens to embedding space
- `Dense2Sparse`
    - graph NN and normal NN uses different way to handle minibatching
- `GraphPair2VecEncoder`
    - encode graph comparison information into one vector
- `FeedForward`
    - final clasifier

Note:
InferSent

----

### Data Type
<!-- .slide: style="font-size: 30px;" -->
- Store graphs and tokens
- `SparseAdjacencyField`

----

### Embedder
<!-- .slide: style="font-size: 30px;" -->
- Give nodes inital embedding
- built in `PretrainedTransformerMismatchedEmbedder`

----

### GraphPair2VecEncoder
<!-- .slide: style="font-size: 30px;" -->
- Take 2 graphs and project them to vector space for comparison
- `Graph2GraphEncoder`
    - Relational Graph Convolution
    - `RGCNConv`
- `GraphPairAttention`
    - Attention Between Graphs
- `Graph2VecEncoder`
    - Attentive Pooling to Keep Important Info.
    - `GlobalAttention`

---

## Usage
<!-- .slide: style="font-size: 30px;" -->
- Install dependencies
    - `./install_dependencies.sh `
- Download NLI style data set to data
    - and specify path in configuration
- Parse data with Stanza
    - `python preprocess.py -i <raw_dir>  -o <target_dir> --files <list_of_files> --force`
- Train model
    - `allennlp train <config> -s <log_dir> --include-package=<src>`

----

## Future Features
<!-- .slide: style="font-size: 30px;" -->
- `pip install -r requirements`
- note that should take lemmatized as node attr if use work embedding
- `<ROOT>` to special token (now use `$`)

---
