# the project is from tzumwu, slightly modified by yt lin
# the parts about path and cuda_device is modified

# IF YOU HAVE MULTIPLE GPUS, USE THIS TO SET WHICH ONE TO USE.
# Single GPU => 0
export CUDA_VISIBLE_DEVICES=0

# OUTPUT DIRECTORY, "$1" IS THE FIRST ARGUMENT WHEN YOU RUN THIS SCRIPT
#OUTPUT_DIR=$1

#AA=/work
#BB=$AA/dev
#echo $BB
PROJ_ROOT=/work/2020-IIS-NLU-internship/SSQA_pytorch_bert_baseline
PARAM_ROOT=$PROJ_ROOT/param
DATA_ROOT=$PROJ_ROOT/dataset/dataset_processed_by_DrLin
#PRED_ROOT=$PROJ_ROOT/prediction

############################################
# where to save model
OUTPUT_DIR=$PARAM_ROOT/bert-baseline
############################################

#USE "hfl/chinese-roberta-wwm-ext-large" FOR THE LARGE PRE-TRAINED MODEL
#INIT_MODEL=hfl/chinese-roberta-wwm-ext
INIT_MODEL=bert-base-chinese

#TRAIN_FILE=/home/tzumwu/_myPython/SSQA_YN/raw_data/SSQA/case_original/train.tsv
#DEV_FILE=/home/tzumwu/_myPython/SSQA_YN/raw_data/SSQA/case_original/dev.tsv
#TEST_FILE=/home/tzumwu/_myPython/SSQA_YN/raw_data/SSQA/case_original/test.tsv

TRAIN_FILE=$DATA_ROOT/train.tsv
DEV_FILE=$DATA_ROOT/dev.tsv
TEST_FILE=$DATA_ROOT/test.tsv

#RESULT_FILE=result.txt, the stats
RESULT_FILE=$OUTPUT_DIR/bert_output_original_ssqa.txt

# TESTING 
   for EPOCH_DIR in ${OUTPUT_DIR}/epoch_* ; do
       echo $EPOCH_DIR
       echo $EPOCH_DIR >> $RESULT_FILE
   
#      # FOR EVALUATION
       python run_yesno_ssqa.py \
           --model_name_or_path $EPOCH_DIR \
           --do_eval True \
           --eval_file $TRAIN_FILE \
           1>> $RESULT_FILE

        
       python run_yesno_ssqa.py \
           --model_name_or_path $EPOCH_DIR \
           --do_eval True \
           --eval_file $DEV_FILE \
           1>> $RESULT_FILE


       python run_yesno_ssqa.py \
           --model_name_or_path $EPOCH_DIR \
           --do_eval True \
           --eval_file $TEST_FILE \
           1>> $RESULT_FILE

       echo >> $RESULT_FILE
    
    # FOR GETTING PREDICTIONS
    python run_yesno_ssqa.py \
        --model_name_or_path $EPOCH_DIR \
        --output_dir $EPOCH_DIR \
        --do_predict True \
        --predict_file $DEV_FILE

    echo

    # FOR GETTING PREDICTIONS
      python run_yesno_ssqa.py \
          --model_name_or_path $EPOCH_DIR \
          --output_dir $EPOCH_DIR \
          --do_predict True \
          --predict_file $TEST_FILE
      echo
      
 done