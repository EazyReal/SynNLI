# by ytlin

# IF YOU HAVE MULTIPLE GPUS, USE THIS TO SET WHICH ONE TO USE.
# Single GPU => 0
export CUDA_VISIBLE_DEVICES=0

# OUTPUT DIRECTORY, "$1" IS THE FIRST ARGUMENT WHEN YOU RUN THIS SCRIPT
#OUTPUT_DIR=$1

PROJ_ROOT=/work/2020-IIS-NLU-internship/SSQA_pytorch_bert_baseline
PARAM_ROOT=$PROJ_ROOT/param
DATA_ROOT=$PROJ_ROOT/dataset/dataset_processed_by_DrLin


############################################
# where to save model, but not used
OUTPUT_DIR=$PARAM_ROOT/bert-baseline
# save pred stats
PRED_ROOT=$PROJ_ROOT/prediction/best-baseline-0716
############################################


INIT_MODEL=bert-base-chinese

TRAIN_FILE=$DATA_ROOT/train.tsv
DEV_FILE=$DATA_ROOT/dev.tsv
TEST_FILE=$DATA_ROOT/test.tsv

# best model is here
############################################
EPOCH_DIR=${OUTPUT_DIR}/epoch_27
############################################

# testing python is here
############################################
TESTING_PY=$PROJ_ROOT/test_yesno_ssqa.py
############################################


# savins are here
############################################
RESULT_FILE=$PRED_ROOT/best_baseline.txt
############################################

echo $EPOCH_DIR
echo $EPOCH_DIR >> $RESULT_FILE


echo $DEV_FILE >> $RESULT_FILE
python $TESTING_PY \
    --model_name_or_path $EPOCH_DIR \
    --do_eval True \
    --eval_file $DEV_FILE \
    1>> $RESULT_FILE


echo $TEST_FILE >> $RESULT_FILE
python $TESTING_PY \
    --model_name_or_path $EPOCH_DIR \
    --do_eval True \
    --eval_file $TEST_FILE \
    1>> $RESULT_FILE

    echo >> $RESULT_FILE
    
    # FOR GETTING PREDICTIONS
python $TESTING_PY \
    --model_name_or_path $EPOCH_DIR \
    --output_dir $PRED_ROOT \
    --do_predict True \
    --predict_file $DEV_FILE

    # FOR GETTING PREDICTIONS
python $TESTING_PY \
    --model_name_or_path $EPOCH_DIR \
    --output_dir $PRED_ROOT \
    --do_predict True \
    --predict_file $TEST_FILE
      
done