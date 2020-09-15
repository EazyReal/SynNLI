//usage:
//@MNLI
// `allennlp train "./src/training/elmo_rgcn.jsonnet" -s "./param/GMN_ELMO_RGCN_MPM_300d"   --include-package "src" --force`
// find lr: `allennlp find-lr ./src/training/elmo_rgcn.jsonnet --include-package "src" -s ./find_lr_elmo --start-lr 0.00001 --end-lr 0.01 --num-batches 50 --force`
//usage evaluate: `allennlp evaluate ./param/GMN_ELMO_RGCN_MPM_300d/model.tar.gz ./data/MNLI_Stanza/pre_multinli_1.0_dev_mismatched.jsonl --output-file ./evaluation_results/GMN_ELMO_RGCN_MPM_300d_Mismatched.txt --batch-size 16 --cuda-device -1 --include-package src`
//allennlp evaluate ./param/GMN_ELMO_RGCN_MPM_300d/model.tar.gz ./data/MNLI_Stanza/pre_multinli_1.0_dev_mismatched.jsonl --output-file ./param/ELMO_MNLI_TEST --include-package src --batch-size=16

//test on mismatched
// --evaluate_on_test 

// main changes bert => elmo, weight decay 0.1=> 0.01, lr 0.0001 => 0.0005

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
local test_data_path = data_root + "/MNLI_Stanza/pre_multinli_1.0_dev_mismatched.jsonl";
local cache_data_dir = null;
local input_fields = ["sentence1", "sentence2", "gold_label"];

// dimensions and models param
local dim_embedder = 1024; //elmo = 512*2
local dim_encoder = 300;
local dim_edge = 50;
local dim_match = 44; //300 for AttDiff, 44 for BiMPM
local num_labels = 3;
local num_edge_labels = 20; //cg => 100, rgcn => 20?

// bert related, unused with elmo
local bert_model = "bert-base-uncased";
local bert_trainable = false;

// elmo related
local elmo_weights = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5";
local elmo_options = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json";

//training
local BATCH_SIZE = 16;
local EPOCH = if test then 2 else 20;
local LR = 0.0005;
local WD = 0.01; //L2 norm
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
        "wordpiece_tokenizer": null,
        "token_indexers" : {
             "tokens": {
                 "type": "elmo_characters_with_mask",
             }
        },
        "input_parsed" : true, //use parsed data
        "input_fields" : input_fields, //use default
        "max_instances" : max_instances, // to exp, simply use 10 here
        "cache_directory": cache_data_dir,
    },
    "evaluate_on_test": true,
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "test_data_path": test_data_path,
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
                    "type": "elmo_token_embedder_with_mask",
                     options_file: elmo_options,
                     weight_file: elmo_weights,
                     do_layer_norm: false,
                     dropout: 0.0, //default = 0.5
                     requires_grad: false,
                     projection_dim: null,
                     vocab_to_cache: null,
                     scalar_mix_parameters: null,
                },
            },
        },
        "edge_embedder": {
            "type": "pass_through",
            "hidden_dim": 1, // use type, not embedding
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
                "type": "rgcn",
                "in_channels": dim_encoder,
                "out_channels": dim_encoder,
                "num_relations": num_edge_labels,
                "root_weight": false,
                "bias": false,
            },
            "atts": {
                "type": "bimpm",
                "bimpm":{
                    "hidden_dim" : 300,
                    "num_perspectives" : 10,
                    "share_weights_between_directions" : false,
                    "with_full_match" : false,
                    "with_maxpool_match" :  true,
                    "with_attentive_match" : true,
                    "with_max_attentive_match" : true,
                },
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

