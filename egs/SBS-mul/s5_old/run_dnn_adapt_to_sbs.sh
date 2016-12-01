#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script takes an existing pre-trained dbn or dnn, renews its softmax layer to create a new DNN,
# and fine-tunes the new DNN according to the SBS data present in data/train.
# The training is done in 3 stages,
#

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Usage: $0 --precomp-dnn "/export/ws15-pt-data/amitdas/kaldi-trunk/egs/babel/s5cp_ad/exp/dnn4e-fmllr_multisoftmax/final.nnet" 
# Usage: $0 --precomp-dbn "/export/ws15-pt-data/amitdas/kaldi-trunk/egs/multilingualdbn/s5/exp/dnn4_pretrain-dbn/6.dbn"

stage=0 # resume training with --stage=N
feats_nj=4
train_nj=8
decode_nj=4
precomp_dbn=
precomp_dnn= #/export/ws15-pt-data/amitdas/kaldi-trunk/egs/babel/s5cp_ad/exp/dnn4e-fmllr_multisoftmax/final.nnet
train_iters=20
use_delta=false
SBS_LANG="SW MD AR DT HG UR"
# End of config.

. utils/parse_options.sh || exit 1;

gmmdir=exp/tri3b
data_fmllr=data-fmllr-tri3b

[[ ! -z ${precomp_dnn} ]] && use_dbn=false || use_dbn=true 
$use_dbn && echo "Using a pre-trained DBN to init target DNN" || echo "Using pre-trained DNN to init target DNN"

echo ==========================
if [ $stage -le 0 ]; then
steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
  data/train data/lang $gmmdir ${gmmdir}_ali 2>&1 | tee ${gmmdir}_ali/align.log
fi
echo ==========================

if [ $stage -le 1 ]; then
  # Store fMLLR features, so we can train on them easily,
  
  # eval
  for lang in $SBS_LANG; do
	dir=$data_fmllr/$lang/eval
	steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
		--transform-dir $gmmdir/decode_eval_$lang \
		$dir data/$lang/eval $gmmdir $dir/log $dir/data || exit 1
	steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
	utils/validate_data_dir.sh --no-text $dir	
  done
  
  # dev
  for lang in $SBS_LANG; do
  dir=$data_fmllr/$lang/dev
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev_$lang \
     $dir data/$lang/dev $gmmdir $dir/log $dir/data || exit 1
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  utils/validate_data_dir.sh --no-text $dir  
  done   
  
  # train
  dir=$data_fmllr/train
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir ${gmmdir} \
     $dir data/train $gmmdir $dir/log $dir/data || exit 1
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  utils/validate_data_dir.sh --no-text $dir   
  
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

if [ $stage -le 2 ]; then
# First check for pre-computed DBN dir. Then try pre-computed DNN dir. If both fail, generate DBN now.
  if [[ ! -z ${precomp_dbn} ]]; then
  echo "using pre-computed dbn ${precomp_dbn}"
  initdir=exp/dnn4_pretrain-outdbn  #out-of-domain dbn (dbn learned from a corpus different than target corpus), ${post_fix}
  mkdir -p $initdir
  # copy the dbn and feat xform from dbn dir
  #ln -s  ${precomp_dbn}  $dir
  cp -r ${precomp_dbn} $initdir 
  # Compute feature xform estmn from the adaptation data (SBS)
  #cp $(dirname ${precomp_dbn})/final.feature_transform $dir
  #feature_transform=$dir/final.feature_transform
  #feature_transform_opt=$(echo "--feature-transform $feature_transform")  
  elif [[ ! -z ${precomp_dnn} ]]; then
  echo "using pre-computed dnn ${precomp_dnn}"
  initdir=exp/dnn4_pretrain-outdnn  #out-of-domain dnn (dnn learned from a corpus different than  target corpus), ${post_fix}
  mkdir -p $initdir
  # replace the softmax layer of the precomp dnn with a random init layer
  nnet_init=$initdir/nnet.init
  rm -rf ${nnet_init}
  perl local/utils/nnet/renew_nnet_softmax.sh $gmmdir/final.mdl ${precomp_dnn} ${nnet_init}  
  # Compute feature xform estmn from the adaptation data (SBS)
  # copy the feat xform from dnn dir
  #cp $(dirname ${precomp_dnn})/final.feature_transform $dir  
  #feature_transform=$dir/final.feature_transform
  #feature_transform_opt=$(echo "--feature-transform $feature_transform")
  else
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  initdir=exp/dnn4_pretrain-indbn #in-domain dbn (dbn learned from the target corpus), ${post_fix}
  (tail --pid=$$ -F $initdir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $initdir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --nn-depth 6 --hid-dim 1024 \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    --rbm-iter 20 $data_fmllr/train $initdir || exit 1;  
  fi
fi

if [ $stage -le 3 ]; then
  # Train the DNN optimizing per-frame cross-entropy.  
  ali=${gmmdir}_ali
  feature_transform=
  dir=${initdir}_dnn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log    
  # Train
  if ${use_dbn}; then
  # Initialize NN training with a DBN
  dbn=${initdir}/6.dbn
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh  --dbn $dbn --hid-layers 0 \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    --learn-rate 0.008 \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 data/lang $ali $ali $dir || exit 1;
  else
  nnet_init=${initdir}/nnet.init
  # Initialize NN training with the hidden layers of a DNN
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --nnet-init ${nnet_init} --hid-layers 0 \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    --learn-rate 0.008 \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 data/lang $ali $ali $dir || exit 1;
  fi
  
  exit 0; # run lang dependent LM decoding - use ./run_oracle_LG.sh  
   
fi

exit 0;

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
