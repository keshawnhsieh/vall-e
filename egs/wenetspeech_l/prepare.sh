#!/usr/bin/env bash

set -eou pipefail

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nj=16
stage=-1
stop_stage=4

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/aishell
#      You can download aishell from https://www.openslr.org/33/
#

dl_dir=/home/keshawnhsieh/data

dataset_parts="-p train -p dev -p test"  # debug
#dataset_parts="-p L"

text_extractor="pypinyin_initials_finals"
audio_extractor="Encodec"  # or Fbank
audio_feats_dir=data/tokenized

. shared/parse_options.sh || exit 1


# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "dl_dir: $dl_dir"
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/aishell,
  # you can create a symlink
  #
  #   ln -sfv /path/to/aishell $dl_dir/aishell
  #
  if [ ! -d $dl_dir/aishell/dev ]; then
    lhotse download aishell $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare aishell manifest"
  # We assume that you have downloaded the aishell corpus
  # to $dl_dir/aishell
  mkdir -p data/manifests
  if [ ! -e data/manifests/.aishell.done ]; then
    lhotse prepare aishell $dl_dir/wenetspeech_l_wav data/manifests
#    touch data/manifests/.aishell.done
  fi
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Tokenize/Fbank aishell"
  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.aishell.tokenize.done ]; then
    python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" \
        --text-extractor ${text_extractor} \
        --audio-extractor ${audio_extractor} \
        --batch-duration 400 \
        --prefix "wenetspeech" \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}"
  fi
#  touch ${audio_feats_dir}/.aishell.tokenize.done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare aishell train/dev/test"
  if [ ! -e ${audio_feats_dir}/.aishell.train.done ]; then
    # train
    lhotse copy \
        ${audio_feats_dir}/aishell_cuts_train.jsonl.gz \
        ${audio_feats_dir}/cuts_train.jsonl.gz

    # dev
    lhotse copy \
        ${audio_feats_dir}/aishell_cuts_dev.jsonl.gz \
        ${audio_feats_dir}/cuts_dev.jsonl.gz

    # test
    lhotse copy \
        ${audio_feats_dir}/aishell_cuts_test.jsonl.gz \
        ${audio_feats_dir}/cuts_test.jsonl.gz

#    touch ${audio_feats_dir}/.aishell.train.done
  fi
fi

python3 ./bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}
