#!/usr/bin/env bash

set -eou pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nj=16
stage=-1
stop_stage=3

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriTTS
#      You can download LibriTTS from https://www.openslr.org/60/
# After downloading tar.gz files, you should extract them into dl_dir/LibriTTS.
# Ignoring *.tar.gz files, which you can download into anywhere, the structure of $dl_dir should look like below
# 
# dl_dir
# ├── dev-clean.tar.gz
# ├── dev-other.tar.gz
# ├── LibriTTS
# │   ├── BOOKS.txt
# │   ├── CHAPTERS.txt
# │   ├── dev-clean
# │   ├── dev-other
# │   ├── eval_sentences10.tsv
# │   ├── LICENSE.txt
# │   ├── NOTE.txt
# │   ├── reader_book.tsv
# │   ├── README_librispeech.txt
# │   ├── README_libritts.txt
# │   ├── speakers.tsv
# │   ├── SPEAKERS.txt
# │   ├── test-clean
# │   ├── test-other
# │   ├── train-clean-100
# │   ├── train-clean-360
# │   └── train-other-500
# ├── test-clean.tar.gz
# ├── test-other.tar.gz
# ├── train-clean-100.tar.gz
# ├── train-clean-360.tar.gz
# └── train-other-500.tar.gz

echo "Downloading the LibriTTS dataset..."

dl_dir=$PWD/download

# dataset_parts="-p dev-clean -p test-clean"  # debug
dataset_parts="--dataset-parts all"  # all

audio_extractor="speechtokenizer"  # or Fbank
audio_feats_dir=data/tokenized

# Set path where model checkpoint and config file is 
checkpoint_path='../speech-token-modified/saved_files/combined_0.2_0.8/'

. dataset/shared/parse_options.sh || exit 1

mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "dl_dir: $dl_dir"
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LibriTTS,
  # you can create a symlink
  #
  #   ln -sfv /path/to/LibriTTS $dl_dir/LibriTTS
  #
  if [ ! -d $dl_dir/LibriTTS/dev-other ]; then
    # lhotse download libritts $dl_dir
    lhotse download libritts ${dataset_parts} $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LibriTTS manifest"
  # We assume that you have downloaded the LibriTTS corpus
  # to $dl_dir/LibriTTS
  mkdir -p data/manifests
  if [ ! -e data/manifests/.libritts.done ]; then
    lhotse prepare libritts ${dataset_parts} -j $nj $dl_dir/LibriTTS data/manifests
    touch data/manifests/.libritts.done
  fi
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Tokenize/Fbank LibriTTS"
  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.libritts.tokenize.done ]; then
    python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" \
        --audio-extractor ${audio_extractor} \
        --batch-duration 400 \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}" \
        --ckpt-dir "${checkpoint_path}"
  fi
  touch ${audio_feats_dir}/.libritts.tokenize.done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare LibriTTS train/dev/test"
  if [ ! -e ${audio_feats_dir}/.libritts.train.done ]; then
    if [ "${dataset_parts}" == "--dataset-parts all" ];then
      # train
      lhotse combine \
        ${audio_feats_dir}/libritts_cuts_train-clean-100.jsonl.gz \
        ${audio_feats_dir}/libritts_cuts_train-clean-360.jsonl.gz \
        ${audio_feats_dir}/libritts_cuts_train-other-500.jsonl.gz \
        ${audio_feats_dir}/cuts_train.jsonl.gz

      # dev
      lhotse copy \
        ${audio_feats_dir}/libritts_cuts_dev-clean.jsonl.gz \
        ${audio_feats_dir}/cuts_dev.jsonl.gz
    else  # debug
      # train
      lhotse copy \
        ${audio_feats_dir}/libritts_cuts_dev-clean.jsonl.gz \
        ${audio_feats_dir}/cuts_train.jsonl.gz
      # dev
      lhotse subset --first 400 \
        ${audio_feats_dir}/libritts_cuts_test-clean.jsonl.gz \
        ${audio_feats_dir}/cuts_dev.jsonl.gz
    fi

    # test
    lhotse copy \
      ${audio_feats_dir}/libritts_cuts_test-clean.jsonl.gz \
      ${audio_feats_dir}/cuts_test.jsonl.gz

    touch ${audio_feats_dir}/.libritts.train.done
  fi
fi

python3 bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}
