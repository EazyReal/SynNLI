//usage:
//@MNLI
// `allennlp train "./src/training/cgconv.jsonnet" -s "./param/GMN_CG_AD_300d_MNLI"   --include-package "src" --force`
//usage2: `allennlp find-lr ./src/training/cgconv.jsonnet --include-package "src" -s ./find_lr --start-lr 0.00001 --end-lr 0.01 --num-batches 50 --force`

local test = false;
local gpu = 0; //use cuda:0
//local gpu = null; //test with cpu

local max_instances = if test then 100 else null;

//data, dirs
local data_root = "/work/2020-IIS-NLU-internship/MNLI/data";
//local train_data_path = data_root + "/hans_preprocessed/train.jsonl";
//local validation_data_path = data_root + "/hans_preprocessed/dev.jsonl";
local train_data_path = data_root + "/MNLI_Stanza/pre_multinli_1.0_train.jsonl";
local validation_data_path = data_root + "/MNLI_Stanza/pre_multinli_1.0_dev_matched.jsonl";
local cache_data_dir = null;
local input_fields = ["sentence1", "sentence2", "gold_label"];

// dimensions and models param
local bert_model = "bert-base-uncased";
local bert_trainable = false;
local dim_embedder = 768;
local dim_encoder = 300;
local dim_edge = 50;
local dim_match = 300;
local num_labels = 3;
local num_edge_labels = 100; //if use cg conv like, no need to be small



//training
local BATCH_SIZE = 16;
local EPOCH = if test then 2 else 20;
local LR = 0.0001;
local WD = 0.1; //L2 norm
local patience = null; 
local use_amp = false; //no scatter for Half

local pooler_setting = {
    "type": "global_attention",
    "gate_nn": {
        "input_dim": dim_encoder,
        "num_layers": 1,
        "hidden_dims": 1,
        "activations": {
            "type": "linear",
            //"negative_slop": 0.2,
        }, 
        "dropout": 0.0,
    },
    "nn" : {
        "input_dim": dim_encoder,
        "num_layers": 1,
        "hidden_dims": dim_encoder,
        "activations": {
            "type": "linear",
            //"negative_slop": 0.2,
        }, 
        "dropout": 0.0,
    },
};

local updater_setting = {
    "type": "gru",
    "input_size" : dim_encoder + dim_match,
    "hidden_size" : dim_encoder,
};


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
                 "type": "pretrained_transformer_mismatched",
                 "model_name" : bert_model,
             }
        },
        "input_parsed" : true, //use parsed data
        "input_fields" : input_fields, //use default
        "max_instances" : max_instances, // to exp, simply use 10 here
        "cache_directory": cache_data_dir,
        //"lazy": null, //let it passed by upper
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "vocabulary": {
        "type": "from_instances",
        "min_count": {
            "edge_labels": 0, //set to 0 if use cg conv like 
        },
        "max_vocab_size": {
            "edge_labels": num_edge_labels,
        },
        "non_padded_namespaces": ["labels"], // default = ["**labels", "**tags"]
        "oov_token": "[UNK]",  
        "padding_token": "[PAD]",
    },
    "model": {
        "type": "graph-nli-ee",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name" : bert_model,
                    "max_length" : null, //use cut and concat 
                    "train_parameters" : bert_trainable,
                    "last_layer_only" : true, 
                    "gradient_checkpointing" : null //study 
                },
            },
        },
        "edge_embedder": { //use passthrough if is not cg conv like
            "type": "embedding",
            "embedding_dim" : dim_edge,
            "num_embeddings" : num_edge_labels,
            "projection_dim" : null,
            "weight" : null,
            "padding_index" : null,
            "trainable" : true,
            "max_norm" : null,
            "norm_type" : 2.0,
            "scale_grad_by_freq" : true,
            "sparse" : false,
            "vocab_namespace" : "edge_labels",
            "pretrained_file" : null,
            //vocab : vocab,
        },
        "projector": {
            "input_dim": dim_embedder,
            "num_layers": 1,
            "hidden_dims": dim_encoder,
            "activations": {
                "type": "linear",
                //"negative_slope": 0.2
            }, 
            "dropout": 0.0,
        },
        "encoder": { //this is of type GraphPair2VecEncoder
            "type": "graph_matching_net",
            "num_layers": 3,
            "convs": {
                "type": "cg",
                "channels": dim_encoder,
                "dim": dim_edge,
                "aggr": "add",
                "batch_norm": false,
                "bias": true,
            },
            "atts": {
                "type": "att_diff",
                "att":{
                    "type": "cosine",
                },
                "dim": dim_match,
            },
            "updaters": [updater_setting, updater_setting],
            "poolers": [pooler_setting, pooler_setting],
        },
        "classifier": {
            "input_dim": 4*dim_encoder,
            "num_layers": 2,
            "hidden_dims": [dim_encoder, num_labels],
            "activations": {
                "type": "leaky_relu",
                "negative_slope": 0.2
            }, 
            "dropout": 0.1,
        },
    },
    "data_loader": {
        "batch_size": BATCH_SIZE,
        "shuffle": true,
    },
    "trainer": {
        "type": "gradient_descent",
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": LR,
            "weight_decay": WD,
        },
        "patience": patience, # early stopping if no improvement in 0 epoch
        "validation_metric": "+accuracy",
        "num_epochs": EPOCH,
        "checkpointer": null, //use default
        "cuda_device": gpu, // use cuda:0
        //"grad_norm": 10, //gradient norm rescaled to have max norm of this value
        "grad_clipping": 1.0, //gradient clipping
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps": 100,
        },
        "tensorboard_writer": {
            "summary_interval" : 100, //default=100
            "histogram_interval": 1000, //default= No Log
            "should_log_parameter_statistics": true,
            "should_log_learning_rate": true, //default if False
            //"batch_size_interval": null,
            //get_batch_num_total: Callable[[], int] = None, passed from Trainer
        },
        //moving_average: Optional[MovingAverage] = None,
        //batch_callbacks: List[BatchCallback] = None,
        //epoch_callbacks: List[EpochCallback] = None,
        //distributed: bool = False,
        //local_rank: int = 0,
        //world_size: int = 1,
        //num_gradient_accumulation_steps: int = 1, // this can be enable if want to use batchsize = 1 
        "use_amp": use_amp,
    }
}

