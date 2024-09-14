#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=0

echo "Preparing the Libriheavy dataset..."

audio_extractor="speechtokenizer"
audio_feats_dir=data/tokenized
manifests_dir=data/manifests

dataset_parts="large"  # large
# dataset_parts="debug"  # for debugging

# Set path where model checkpoint and config file is 
checkpoint_path='../speech-token-modified/saved_files_v1/combined_0.2_0.8/'

. dataset/shared/parse_options.sh || exit 1

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Splitting libriheavy large"
    lhotse split-lazy data/manifests/libriheavy_cuts_${dataset_parts}.jsonl.gz \
            "${manifests_dir}" 1115694 || exit 1

    log "Done Splitting libriheavy large"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Tokenize libriheavy large parts"
    for parts in 00000000 00000001; do
        python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" --prefix "libriheavy" --suffix "${parts}.jsonl.gz" \
            --audio-extractor ${audio_extractor} \
            --batch-duration 400 \
            --src-dir "${manifests_dir}" \
            --output-dir "${audio_feats_dir}" \
            --ckpt-dir "${checkpoint_path}"

        log "Done Tokenize libriheavy large "${parts}" part"
    log "Done Tokenize libriheavy large all part"
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Tokenize libriheavy large parts"
    for parts in 00000002 00000003; do
        python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" --prefix "libriheavy" --suffix "${parts}.jsonl.gz" \
            --audio-extractor ${audio_extractor} \
            --batch-duration 400 \
            --src-dir "${manifests_dir}" \
            --output-dir "${audio_feats_dir}" \
            --ckpt-dir "${checkpoint_path}"

        log "Done Tokenize libriheavy large "${parts}" part"
    log "Done Tokenize libriheavy large all part"
    done
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Tokenize libriheavy large parts"
    for parts in 00000004 00000005; do
        python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" --prefix "libriheavy" --suffix "${parts}.jsonl.gz" \
            --audio-extractor ${audio_extractor} \
            --batch-duration 400 \
            --src-dir "${manifests_dir}" \
            --output-dir "${audio_feats_dir}" \
            --ckpt-dir "${checkpoint_path}"

        log "Done Tokenize libriheavy large "${parts}" part"
    log "Done Tokenize libriheavy large all part"
    done
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Tokenize libriheavy large parts"
    for parts in 00000006 00000007; do
        python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" --prefix "libriheavy" --suffix "${parts}.jsonl.gz" \
            --audio-extractor ${audio_extractor} \
            --batch-duration 400 \
            --src-dir "${manifests_dir}" \
            --output-dir "${audio_feats_dir}" \
            --ckpt-dir "${checkpoint_path}"

        log "Done Tokenize libriheavy large "${parts}" part"
    log "Done Tokenize libriheavy large all part"
    done
fi


if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Tokenize libriheavy large parts"
    for parts in 00000008 00000009; do
        python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" --prefix "libriheavy" --suffix "${parts}.jsonl.gz" \
            --audio-extractor ${audio_extractor} \
            --batch-duration 400 \
            --src-dir "${manifests_dir}" \
            --output-dir "${audio_feats_dir}" \
            --ckpt-dir "${checkpoint_path}"

        log "Done Tokenize libriheavy large "${parts}" part"
    log "Done Tokenize libriheavy large all part"
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Combine libriheavy large parts"
    
    if [ "${dataset_parts}" != "debug" ];then
      lhotse combine \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000000.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000001.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000002.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000003.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000004.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000005.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000006.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000007.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000008.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000009.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.jsonl.gz

    else  # debug
      lhotse combine \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000000.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.00000001.jsonl.gz \
        ${audio_feats_dir}/libriheavy_cuts_${dataset_parts}.jsonl.gz     
    fi
    
fi
