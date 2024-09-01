#!/bin/bash

# Step 1: Prepare dataset
bash dataset/prepare_ljspeech.sh --stage -1 --stop-stage 3

# Define the save directory
exp_dir=saved_files/version_1

# Variables for AR model training
MAX_DURATION_AR=80
FILTER_MIN_DURATION_AR=0.5
FILTER_MAX_DURATION_AR=14
TRAIN_STAGE_AR=1
NUM_BUCKETS_AR=6
DTYPE_AR="bfloat16"
SAVE_EVERY_N_AR=10000
VALID_INTERVAL_AR=20000
MODEL_NAME_AR="valle"
SHARE_EMBEDDING_AR=true
NORM_FIRST_AR=true
ADD_PRENET_AR=false
DECODER_DIM_AR=1024
NHEAD_AR=16
NUM_DECODER_LAYERS_AR=12
PREFIX_MODE_AR=1
BASE_LR_AR=0.05
WARMUP_STEPS_AR=200
AVERAGE_PERIOD_AR=0
NUM_EPOCHS_AR=2
START_EPOCH_AR=1
START_BATCH_AR=0
ACCUMULATE_GRAD_STEPS_AR=4

# Variables for NAR model training
MAX_DURATION_NAR=40
FILTER_MIN_DURATION_NAR=0.5
FILTER_MAX_DURATION_NAR=14
TRAIN_STAGE_NAR=2
NUM_BUCKETS_NAR=6
DTYPE_NAR="float32"
SAVE_EVERY_N_NAR=10000
VALID_INTERVAL_NAR=20000
MODEL_NAME_NAR="valle"
SHARE_EMBEDDING_NAR=true
NORM_FIRST_NAR=true
ADD_PRENET_NAR=false
DECODER_DIM_NAR=1024
NHEAD_NAR=16
NUM_DECODER_LAYERS_NAR=12
PREFIX_MODE_NAR=1
BASE_LR_NAR=0.05
WARMUP_STEPS_NAR=200
AVERAGE_PERIOD_NAR=0
NUM_EPOCHS_NAR=1
START_EPOCH_NAR=3
START_BATCH_NAR=0
ACCUMULATE_GRAD_STEPS_NAR=4

# Step 2: Train AR model
python3 bin/trainer.py \
  --max-duration "$MAX_DURATION_AR" \
  --filter-min-duration "$FILTER_MIN_DURATION_AR" \
  --filter-max-duration "$FILTER_MAX_DURATION_AR" \
  --train-stage "$TRAIN_STAGE_AR" \
  --num-buckets "$NUM_BUCKETS_AR" \
  --dtype "$DTYPE_AR" \
  --save-every-n "$SAVE_EVERY_N_AR" \
  --valid-interval "$VALID_INTERVAL_AR" \
  --model-name "$MODEL_NAME_AR" \
  --share-embedding "$SHARE_EMBEDDING_AR" \
  --norm-first "$NORM_FIRST_AR" \
  --add-prenet "$ADD_PRENET_AR" \
  --decoder-dim "$DECODER_DIM_AR" \
  --nhead "$NHEAD_AR" \
  --num-decoder-layers "$NUM_DECODER_LAYERS_AR" \
  --prefix-mode "$PREFIX_MODE_AR" \
  --base-lr "$BASE_LR_AR" \
  --warmup-steps "$WARMUP_STEPS_AR" \
  --average-period "$AVERAGE_PERIOD_AR" \
  --num-epochs "$NUM_EPOCHS_AR" \
  --start-epoch "$START_EPOCH_AR" \
  --start-batch "$START_BATCH_AR" \
  --accumulate-grad-steps "$ACCUMULATE_GRAD_STEPS_AR"

# Step 3: Train NAR model
python3 bin/trainer.py \
  --max-duration "$MAX_DURATION_NAR" \
  --filter-min-duration "$FILTER_MIN_DURATION_NAR" \
  --filter-max-duration "$FILTER_MAX_DURATION_NAR" \
  --train-stage "$TRAIN_STAGE_NAR" \
  --num-buckets "$NUM_BUCKETS_NAR" \
  --dtype "$DTYPE_NAR" \
  --save-every-n "$SAVE_EVERY_N_NAR" \
  --valid-interval "$VALID_INTERVAL_NAR" \
  --model-name "$MODEL_NAME_NAR" \
  --share-embedding "$SHARE_EMBEDDING_NAR" \
  --norm-first "$NORM_FIRST_NAR" \
  --add-prenet "$ADD_PRENET_NAR" \
  --decoder-dim "$DECODER_DIM_NAR" \
  --nhead "$NHEAD_NAR" \
  --num-decoder-layers "$NUM_DECODER_LAYERS_NAR" \
  --prefix-mode "$PREFIX_MODE_NAR" \
  --base-lr "$BASE_LR_NAR" \
  --warmup-steps "$WARMUP_STEPS_NAR" \
  --average-period "$AVERAGE_PERIOD_NAR" \
  --num-epochs "$NUM_EPOCHS_NAR" \
  --start-epoch "$START_EPOCH_NAR" \
  --start-batch "$START_BATCH_NAR" \
  --accumulate-grad-steps "$ACCUMULATE_GRAD_STEPS_NAR" 

# Step 4: (optional) Perform inference
python3 bin/infer.py --output-dir ${exp_dir}/infer/demos \
    --checkpoint=${exp_dir}/epoch-2.pt \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts dataset/prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below." \
    --ckpt-dir ../speech-token-modified/saved_files/combined_0.2_0.8/ # path to checkpoint directory of trained tokenizer.
