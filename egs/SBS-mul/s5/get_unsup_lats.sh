#!/bin/bash

# Copyright 2015-2016  University of Illinois (Author: Amit Das)
# Apache 2.0
#

# Prepares unsupervised data scp, and decodes them using a DNN; assumes features are (LDA+MLLT or delta+delta-delta)
# + fMLLR (probably with SAT models).
# It first computes an alignment with the SAT model final.alimdl (or the final.mdl if final.alimdl
# is not present), then does 2 iterations of fMLLR transform estimation of the speakers in the unsupervised audio.
# Once the fMLLR transforms are ready, it generates the fMLLR features. Finally, using these fMLLR features,
# the unsupervised data is decoded (i.e. lattices generated) using a reasonably well-trained DNN.

set -euo pipefail

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Set the location of the SBS speech
SBS_CORPUS=${SBS_DATADIR}/audio

# Options
LANG="SW"   # Target language
nutts=4000
unsup_dir_tag=
feats_nj=40
train_nj=20
decode_nj=5
stage=-100 # resume training with --stage=N

# Decode Config:
acwt=0.2
parallel_opts="--num-threads 6"

# End of config.

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

if [ $# != 6 ]; then
  echo "Num args reqd is 6." 
  exit 1;
fi

# ./get_unsup_lats.sh --nutts 1000 "SW" exp/tri3cpt_ali/SW  data-fmllr-tri3c/SW/SW  exp/tri3c/SW/graph_text_G_SW
# exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax/decode_dev_text_G_SW exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax/decode_unsup_1000_SW
LANG=$1
gmmdir=$2     # SAT GMM used to estimate fMLLR transforms for unlabeled data
featdir=$3    # Dir where fMMLR features will be saved
graphdir=$4   # HCLG graph dir. "H" should be based on $gmmdir/final.mdl
dnndir=$5     # DNN directory used to decode unsup data. This should be reasonably well-trained to get meaningful hypotheses lattices
dir=$6        # Dir to save the decoded lattices

[ -z $unsup_dir_tag ] && unsup_dir_tag="unsup_$nutts"
echo "fMLLR transforms of the unsup data will be saved in $gmmdir/decode_${unsup_dir_tag}_$LANG"
echo "fMMLR features of the unsup data will be saved in $featdir"
echo "Decoding lattices of the unsup data will be saved in $dir"
 
#Prepare unsupervised data scp
if [ $stage -le -4 ]; then
  if [ ! -f data/$LANG/unsup/spk2utt -o ! -f data/$LANG/unsup/utt2spk -o ! -f data/$LANG/unsup/wav.scp ]; then    
    local/sbs_gen_data_dir.sh --corpus-dir=$SBS_CORPUS \
    --lang-map=conf/lang_codes.txt $LANG || exit 1 ;    
  fi
fi

# Prepare features for a subset $nutts utterances.
L=$LANG
if [ $stage -le -3 ]; then
  mfccdir=mfcc/$L
  if [ ! -f data/$LANG/unsup/feats.scp ]; then
    rm -rf data/$L/$unsup_dir_tag exp/make_mfcc/$L/unsup exp/make_mfcc/$L/$unsup_dir_tag
    steps/make_mfcc.sh --nj $feats_nj --cmd "$train_cmd" data/$L/unsup exp/make_mfcc/$L/unsup $mfccdir || exit 1
    
    # If wav.scp has m files and m < nutts, then we have only m utterances available to us although our target is to have nutts.
    # Hence, create a subset data dir with m utterances.
    nutts_wavscp=$(cat data/$L/unsup/wav.scp|wc -l)
    nutts_final=$(($nutts_wavscp < $nutts?$nutts_wavscp:$nutts))    
    echo "User requested $nutts unsup files. Corpus has $nutts_final files. Will use $nutts_final files."

    utils/subset_data_dir.sh data/$L/unsup $nutts_final data/$L/$unsup_dir_tag || exit 1
    steps/compute_cmvn_stats.sh data/$L/$unsup_dir_tag exp/make_mfcc/$L/$unsup_dir_tag $mfccdir || exit 1
    
    # delete unsup wav files which are not in the subset (saves lots of disk space)
    rm -rf data/$L/unsup
    while read line; do
     rm $line; 
    done < <(comm -23 <(ls -1 data/$L/wav/unsup/* | sort ) <(awk '{print $2}' data/$L/$unsup_dir_tag/wav.scp|sort))
  fi
fi

# Decode unlabeled data using GMM (used to estimate fMLLR speaker transform)
if [ $stage -le -2 ]; then
  rm -rf $gmmdir/decode_${unsup_dir_tag}_$L $gmmdir/decode_${unsup_dir_tag}_${L}.si
  steps/decode_fmllr.sh $parallel_opts --nj $decode_nj --cmd "$decode_cmd" \
    --skip-scoring true --acwt $acwt \
    $graphdir data/$L/$unsup_dir_tag $gmmdir/decode_${unsup_dir_tag}_$L || exit 1
fi

# Create fMLLR features of the unlabeled data
#featdir=$data_fmllr/${unsup_dir_tag}_$L
if [ $stage -le -1 ]; then
  rm -rf $featdir
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
    --transform-dir $gmmdir/decode_${unsup_dir_tag}_$L \
    $featdir data/$L/$unsup_dir_tag $gmmdir $featdir/log $featdir/data  || exit 1
    
  steps/compute_cmvn_stats.sh $featdir $featdir/log $featdir/data || exit 1;
  utils/validate_data_dir.sh --no-text $featdir
fi

# Decode unlabeled data using DNN
if [ $stage -le 0 ]; then
  rm -rf $dir
  steps/nnet/decode.sh --num-threads 3 --nj $decode_nj --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --acwt $acwt --skip-scoring true --srcdir $dnndir \
    $graphdir $featdir $dir || exit 1
fi

exit 0;

