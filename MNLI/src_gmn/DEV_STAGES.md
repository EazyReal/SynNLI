# Stage 1 - Code Study Allennlp + DataReader, SparseAdjacencyField
- 8/?-8/?
- code study allennlp (across stages)
    - study tools and resources
        - allennlp guide
        - allennlp doc
        - allennlp github (source code)
        - google
    - data
        - fields
        - instances
        - batch
        - dataset
        - vocabulary
    - data operator
        - reader(to_instance)
        - vocab.from_instances
        - token_indexer(index_with => token_id)
        - dataloader(call batch_tensors in fields)
        - embedder(token_id2vec)
        - encoder(model part)
    - trainer
        - trainer
        - tensorboard_writer
        - config file composition and jsonnet
    - general work flow conclusion
        - rawdata =reader=> instance (with fields)
        - instance =Vocab=> vocab (with namespaces)
        - instance =IndexWith=> indexed_instances (TensorDict)
        - instances =dataloader(batch_tensors function)=> batch_tensor (TensorDict)
        - batch_tensor =model=> logits
    - allennlp conclusion
        - OOP + dependency injection
        - a good coding style
        - several robust off-the-shelf models
- implementation of datareader
    - use utils.doc2graph
    - graph2instance
- implementation of SparseAdjacencyField
    - sparse version of origin AdjacencyField
    - modify code (almost all) of allennlp AdjacencyField
    - implementation of PytorchGeoData Batching

# Stage 2 - Train a naive model (BagofWordPooling) with allennlp train
- 8/?-8/?
- (start this note when 2->3)
- mismatched BERT (use defualt mean)
    - use PretrainedTransformerMismatchedIndexer + PretrainedTransformerMismatchedEmbedder
    - note that here use BERT without special token
    - also "[ROOT]" in dependency graph is not special token to BERT is a potential issue
- sparse2dense, dense2sparse in tensorop.py
    - naive implementation works well without tensor
    - fix gradient issue
        - learn about leaf node in computatino graph
        - inplace operation
        - tensor properties
        - torch.sparse.Tensor.to_dense() as tf.scatter_nd
    - 2020/8/21, can actually use pytorch_scatter, pytorch_sparse...
- allennlp train can work with my modules
- advanced training with tensorboard, optimizer, shedelur settings by looking at source
    
    
# (Now) State 3 - Train A HGNN model (het graph embedding w/o interaction)
- due 8/22
- add Graph2VecEncoder Registrable
- implement HGEN

# Stage 4 - Train A HGMN model (het graph matching network (may be final))
- due 8/31
- add GraphPair2VecEncoder Registrable
- implement HGMN

# Stage 5 - Validation on ANLI/Q-Test/HAN, Experiments
- due 9/15
- parse ANLI/HAN
- Q-Test generator(This may be required earlier)

# Stage 6 - Paper Fixing (due 9/19, EACL due 9/20)