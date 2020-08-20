//the file path should be relative to the path that call `allennlp train`
//usage: `allennlp train "./src_gmn/training_config.jsonnet" -s "./param/testv1"   --include-package "package_v1" --force`
local bert_model = "bert-base-uncased";
local train_data_path = "./data/MNLI_Stanza/pre_multinli_1.0_dev_matched.jsonl";
local validation_data_path = "./data/MNLI_Stanza/pre_multinli_1.0_dev_mismatched.jsonl";

{
    "dataset_reader" : {
        "type": "nli-graph",
        "wordpiece_tokenizer": {
            "type" : "pretrained_transformer",
            "model_name" : bert_model
        }
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "model": {
        "type": "simple_model",
        "embedder": {
            "type": "pretrained_transformer_mismatched",
            "model_name" : bert_model
        },
        "pooler": {
            "type": "boe", //bag_of_embeddings
            "embedding_dim": 768
        }
    },
    "data_loader": {
        "batch_size": 4,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 1
    }
}

