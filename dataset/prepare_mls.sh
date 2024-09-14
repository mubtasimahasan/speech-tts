#!/usr/bin/env bash

set -eou pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nj=16
stage=-1
stop_stage=3

echo "Downloading the Multilingual LibriSpeech dataset..."

dl_dir=$PWD/download

audio_extractor="speechtokenizer"  # or Fbank
audio_feats_dir=data/tokenized

# Set path where model checkpoint and config file is 
checkpoint_path='../speech-token-modified/saved_files_v1/combined_0.2_0.8/'

. dataset/shared/parse_options.sh || exit 1

mkdir -p data
mkdir -p $dl_dir

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Download and Extract data
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "dl_dir: $dl_dir"
  log "Stage 0: Download and Extract data"

  if [ ! -f "$dl_dir/mls_english.tar.gz" ]; then
    urls=(
      # "https://dl.fbaipublicfiles.com/mls/mls_polish.tar.gz" # for debugging with small data
      "https://dl.fbaipublicfiles.com/mls/mls_english.tar.gz" # English size (2.4T)     
    )
    # Download each file
    for url in "${urls[@]}"; do
      file_name=$(basename $url)
      output_file="$dl_dir/$file_name"
      log "Downloading $file_name"
      wget $url -O $output_file
    done 
  else
    log "File already exists, skipping download."
  fi

  # Extract the downloaded data
  if [ ! -d $dl_dir/multils ]; then
    log "Extracting .tar file to $dl_dir/multils"
    mkdir -p $dl_dir/multils
    tar -xf "$dl_dir/mls_english.tar.gz" -C "$dl_dir/multils"
    log "Extraction complete: $dl_dir/multils"
  else
    log "File was extracted, extraction skipped."
  fi
fi

# Prepare Manifest
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare manifest"
  
  mkdir -p data/manifests
  if [ ! -e data/manifests/.mls.done ]; then
    lhotse prepare mls $dl_dir/multils data/manifests --flac --num-jobs 40
    touch data/manifests/.mls.done
  else
    log "Manifests were prepared, stage skipped."
  fi
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 2: Tokenize mls"
  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.mls.tokenize.done ]; then
    python3 bin/tokenizer.py --dataset-parts "train test dev" --prefix "mls-english" \
        --audio-extractor ${audio_extractor} \
        --batch-duration 400 \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}" \
        --ckpt-dir "${checkpoint_path}"
  else
    log "Manifests were tokenized, tokenization skipped."
  fi
  cd ${audio_feats_dir}
  ln -sf mls-english_cuts_train.jsonl.gz cuts_train.jsonl.gz
  ln -sf mls-english_cuts_dev.jsonl.gz cuts_dev.jsonl.gz
  ln -sf mls-english_cuts_test.jsonl.gz cuts_test.jsonl.gz
  cd -
  touch ${audio_feats_dir}/.mls.tokenize.done
fi

python3 bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}
