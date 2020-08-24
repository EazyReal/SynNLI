//the file path should be relative to the path that call `allennlp train`
//usage: `allennlp train "./src_gmn/training_config.jsonnet" -s "./param/testv1"   --include-package "package_v1" --force`
//use amp

local bert_model = "bert-base-uncased";

local data_root = "/work/2020-IIS-NLU-internship/MNLI/data";
local train_data_path = data_root + "/MNLI_Stanza/pre_multinli_1.0_train.jsonl";
local validation_data_path = data_root + "/MNLI_Stanza/pre_multinli_1.0_dev_matched.jsonl";
local cache_data_dir = data_root + "/MNLI_instance_cache";

local BATCH_SIZE = 32;
local EPOCH = 20;

local input_fields = ["sentence1", "sentence2", "gold_label"];
local num_edge_labels = 20;

local dim_embedder = 768;
local dim_encoder = 300;
local num_labels = 3;

// care vocabulary of edge labels, this is related to model

{
    "dataset_reader" : {
        "type": "nli-graph",
        "wordpiece_tokenizer": {
            "type" : "pretrained_transformer",
            "model_name" : bert_model
        },
        "token_indexers" : {
             "tokens": {
                 // need "indexer" ?
                 "type": "pretrained_transformer_mismatched",
                 "model_name" : bert_model,
             }
        },
        "input_parsed" : true, //use parsed data
        "input_fields" : input_fields, //use default
        "max_instances" : 10, // to exp, simply use 10 here
        "cache_directory": cache_data_dir,
        "lazy": null,
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "vocabulary": {
        "type": "from_instances",
        "min_count": {
            "edge_labels": 1000,
        },
        "max_vocab_size": {
            "edge_labels": num_edge_labels,
        },
        "non_padded_namespaces": ["labels"] // default = ["**labels", "**tags"]
        "oov_token": "[UNK]",  
        "padding_token": "[PAD]",
    },
    "model": {
        "type": "graph-nli",
        "embedder": {
            "type": "pretrained_transformer_mismatched",
            "model_name" : bert_model,
            "max_length" : null, //use cut and concat 
            "train_parameters" : true,
            "last_layer_only" : true, 
            "gradient_checkpointing" : null //study 
        },
        "projector": {
            "input_dim": dim_embedder,
            "num_layers": 1,
            "hidden_dims": dim_encoder,
            "activations": {
                "type": "leaky_relu",
                "negative_slope": 0.2
            }, 
            "dropout": 0.1,
        },
        "encoder": { //this is of type GraphPair2VecEncoder
            "type": "rgcn",
            "dim": dim_encoder,
        },
        "classifier": {
            "input_dim": 4*dim_ecoder,
            "num_layers": 1,
            "hidden_dims": num_labels,
            "activations": {
                "type": "leaky_relu",
                "negative_slope": 0.2
            }, 
            "dropout": 0.1,
        },
    },
    "data_loader": {
        "batch_size": BATCH_SIZE,
        "shuffle": true
    },
    "trainer": {
        "type": "gradient_descent",
        "optimizer": "huggingface_adamw",
        "patience": 5,
        "validation_metric": "-loss",
        "num_epochs": 10,
        "checkpointer": null, //use default
        "cuda_device": 0, // use cuda:0
        //"grad_norm": None, gradient norm rescaled to have max norm of this value
        "grad_clipping": 1.0, //gradient clipping
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            //"num_epochs": EPOCH, this will be passed by **extras
            "warmup_steps": 100,
            //optimizer: torch.optim.Optimizer, (should this be passed after train command?)
            //num_steps_per_epoch: int = None,
            //last_epoch: int = -1,
        },
        "tensorboard_writer": {
            //"summary_interval" : 100, //default=100
            "histogram_interval": 1000, //default= No Log
            //"batch_size_interval": null, (bug?)
            "should_log_parameter_statistics": true,
            "should_log_learning_rate": true, //default if False
            //get_batch_num_total: Callable[[], int] = None, passed from Trainer
        },
        //moving_average: Optional[MovingAverage] = None,
        //batch_callbacks: List[BatchCallback] = None,
        //epoch_callbacks: List[EpochCallback] = None,
        //distributed: bool = False,
        //local_rank: int = 0,
        //world_size: int = 1,
        //num_gradient_accumulation_steps: int = 1, // this can be enable if want to use batchsize = 1 
        "use_amp": true,
    }
}

