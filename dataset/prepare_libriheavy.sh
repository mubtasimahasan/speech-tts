#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=4

echo "Preparing the Libriheavy dataset..."

audio_extractor="speechtokenizer"
audio_feats_dir=data/tokenized

dataset_parts="small medium large dev"  # all
# dataset_parts="debug"  # for debugging

# Set path where model checkpoint and config file is 
checkpoint_path='../speech-token-modified/saved_files_v1/combined_0.2_0.8/'

. dataset/shared/parse_options.sh || exit 1

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: Downloading audio file."
  mkdir -p download/librilight
  for subset in small medium large; do
    log "Downloading ${subset} subset."
    if [ ! -d download/librilight/${subset} ]; then
      wget -P download/librilight -c https://dl.fbaipublicfiles.com/librilight/data/${subset}.tar 
      tar xf download/librilight/${subset}.tar -C download/librilight
    else
      log "Skipping download, ${subset} subset exists."
    fi
  done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Downloading manifests from huggingface."
  for subset in small medium large dev test_clean test_other test_clean_large test_other_large; do
    if [ ! -e data/manifests/libriheavy_cuts_${subset}.jsonl.gz ]; then
      log "Downloading ${subset} subset."
      wget -P data/manifests -c https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_${subset}.jsonl.gz
    else
      log "Skipping download, ${subset} subset exists."
    fi
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Split For Debugging"

  if [ "${dataset_parts}" == "debug" ];then
    if [ ! -e data/manifests/libriheavy_cuts_debug.jsonl.gz ]; then
      lhotse subset --first 5000 data/manifests/libriheavy_cuts_small.jsonl.gz \
        data/manifests/libriheavy_cuts_debug.jsonl.gz || exit 1
    fi
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Tokenize libriheavy"
  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.libriheavy.tokenize.done ]; then
    python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" --prefix "libriheavy" \
        --audio-extractor ${audio_extractor} \
        --batch-duration 400 \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}" \
        --ckpt-dir "${checkpoint_path}"
  else
    log "Manifests were tokenized, tokenization skipped."
  fi
  touch ${audio_feats_dir}/.libriheavy.tokenize.done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare Libriheavy train/dev "
  if [ ! -e ${audio_feats_dir}/.libriheavy.train.done ]; then
    if [ "${dataset_parts}" != "debug" ];then
      # train
      lhotse combine \
        ${audio_feats_dir}/libriheavy_cuts_large.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_medium.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_small.jsonl.gz \
        ${audio_feats_dir}/cuts_train.jsonl.gz

      # dev
      lhotse copy \
        ${audio_feats_dir}/libriheavy_cuts_dev.jsonl.gz \
        ${audio_feats_dir}/cuts_dev.jsonl.gz
    else  # debug
      # train
      lhotse copy \
        ${audio_feats_dir}/libriheavy_cuts_debug.jsonl.gz \
        ${audio_feats_dir}/cuts_train.jsonl.gz
      # dev
      lhotse subset --first 500 \
        ${audio_feats_dir}/libriheavy_cuts_debug.jsonl.gz \
        ${audio_feats_dir}/cuts_dev.jsonl.gz
    fi

    touch ${audio_feats_dir}/.libriheavy.train.done
  fi
fi

python3 bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}

