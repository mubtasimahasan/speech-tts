#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0 #--------------> 0 = AR and 1 = NAR
#SBATCH --ntasks=1 
#SBATCH -t 24:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 
#SBATCH --cpus-per-task=32 
#SBATCH --output=saved_files/slurm/tts_train_E01/log-%A-%a.log 
#SBATCH -J tts_speech
PORT=$((29500 + SLURM_JOB_ID % 1000))

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

module purge
module load apptainer

EXPORT_DIRECTORY=saved_files/version_1

# Arrays for model training variables
MAX_DURATION=(80 40) # Maximum pooled recordings duration (seconds) in a single batch. Reduce it if it causes CUDA OOM.
FILTER_MIN_DURATION=(0.5 0.5) # Keep only utterances with duration > this.
FILTER_MAX_DURATION=(14 14) # Keep only utterances with duration < this.
TRAIN_STAGE=(1 2) # 0: train all modules, 1: AR Decoder, 2: NAR Decoder(s)
NUM_BUCKETS=(6 6) # The number of buckets for the DynamicBucketingSampler.
DTYPE=("bfloat16" "float32") # Training dtype: float32 bfloat16 float16.
SAVE_EVERY_N=(10000 10000) # Save checkpoint after processing this number of batches.
VALID_INTERVAL=(20000 20000) # Run validation if batch_idx %% valid_interval is 0.
MODEL_NAME=("valle" "valle") # Model
SHARE_EMBEDDING=(true true) # Share the parameters of the output projection layer with the parameters of the acoustic embedding.
NORM_FIRST=(true true) # Pre or Post Normalization.
ADD_PRENET=(false false) # Whether add PreNet after Inputs.
DECODER_DIM=(1024 1024) # Embedding dimension in the decoder model.
NHEAD=(16 16) # Number of attention heads in the Decoder layers.
NUM_DECODER_LAYERS=(12 12) # Number of Decoder layers.
PREFIX_MODE=(1 1) # How to prefix VALL-E NAR Decoder, 0: no prefix, 1: 0 to random, 2: random to random, 4: chunk of pre or post utterance.
BASE_LR=(0.05 0.05) # The base learning rate.
WARMUP_STEPS=(200 200) # The warm-up steps of learning rate.
AVERAGE_PERIOD=(0 0) # Update the averaged model after processing this number of batches.
NUM_EPOCHS=(100 100) # Number of epochs to train.
START_EPOCH=(1 3) # Resume training from this epoch. If larger than 1, it will load checkpoint.
START_BATCH=(0 0) # If > 0, it loads the checkpoint.
ACCUMULATE_GRAD_STEPS=(4 4) # Update gradient when batch_idx_train %% accumulate_grad_steps == 0.

# Selecting parameters based on the array task ID (0 for AR, 1 for NAR)
IDX=${SLURM_ARRAY_TASK_ID}

# Processing step for NAR (Array ID 1)
if [ "$IDX" -eq 1 ]; then
  cp ${EXPORT_DIRECTORY}/best-valid-loss.pt ${EXPORT_DIRECTORY}/epoch-2.pt  # Copy AR model checkpoint to start NAR training
fi

# Run the trainer.py script with the selected parameters
apptainer exec --nv /scratch/mi8uu/mub/speech-tts_latest.sif python3 bin/trainer.py \
  --max-duration "${MAX_DURATION[$IDX]}" \
  --filter-min-duration "${FILTER_MIN_DURATION[$IDX]}" \
  --filter-max-duration "${FILTER_MAX_DURATION[$IDX]}" \
  --train-stage "${TRAIN_STAGE[$IDX]}" \
  --num-buckets "${NUM_BUCKETS[$IDX]}" \
  --dtype "${DTYPE[$IDX]}" \
  --save-every-n "${SAVE_EVERY_N[$IDX]}" \
  --valid-interval "${VALID_INTERVAL[$IDX]}" \
  --model-name "${MODEL_NAME[$IDX]}" \
  --share-embedding "${SHARE_EMBEDDING[$IDX]}" \
  --norm-first "${NORM_FIRST[$IDX]}" \
  --add-prenet "${ADD_PRENET[$IDX]}" \
  --decoder-dim "${DECODER_DIM[$IDX]}" \
  --nhead "${NHEAD[$IDX]}" \
  --num-decoder-layers "${NUM_DECODER_LAYERS[$IDX]}" \
  --prefix-mode "${PREFIX_MODE[$IDX]}" \
  --base-lr "${BASE_LR[$IDX]}" \
  --warmup-steps "${WARMUP_STEPS[$IDX]}" \
  --average-period "${AVERAGE_PERIOD[$IDX]}" \
  --num-epochs "${NUM_EPOCHS[$IDX]}" \
  --start-epoch "${START_EPOCH[$IDX]}" \
  --start-batch "${START_BATCH[$IDX]}" \
  --accumulate-grad-steps "${ACCUMULATE_GRAD_STEPS[$IDX]}" \
  --exp-dir ${EXPORT_DIRECTORY}
