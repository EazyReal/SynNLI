# Training Configurations

## Usage
- at MNLI directory
- `allennlp train "./src/training/config_gmn.jsonnet" -s "./param/GMN_BERT_GAT_300d"   --include-package "src" --force`
- `allennlp find-lr ./src/training/config_gmn.jsonnet --include-package "src" -s ./find_lr --start-lr 0.00001 --end-lr 0.01 --num-batches 50 --force`

## flag
- test 
    - if set, reader will have max_instances set to 100