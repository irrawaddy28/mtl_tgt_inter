#!/bin/bash -e

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0 # resume training with --stage=N
feats_nj=4
train_nj=8
decode_nj=4
# End of config.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "usage: $0 [options] <lang code> gmm-dir> <feat-dir> <out-dir>"
   echo "e.g.:  steps/align_fmllr.sh data/train data/lang exp/tri1 exp/tri1_ali"   
   exit 1;
fi

TEST_LANG=$1
gmmdir=$2        # exp/tri3b/$lang
alidir=$3        # exp/tri3b_ali/$lang
data_fmllr=$4    # data-fmllr-tri3b/$lang
dbndir=$5		 # exp/dnn4_pretrain-dbn/$lang/indbn
nnetdir=$6       # exp/dnn4_pretrain-dbn_dnn/$lang/monosoftmax_dt

lang=$TEST_LANG
echo ==========================
if [ $stage -le 0 ]; then
steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
  data/$lang/train data/$lang/lang $gmmdir $alidir
fi
echo ==========================

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
     
  # test
  dir=$data_fmllr/eval
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval \
     $dir data/$lang/eval $gmmdir $dir/log $dir/data
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  utils/validate_data_dir.sh --no-text $dir

  # dev
  dir=$data_fmllr/dev
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev \
     $dir data/$lang/dev $gmmdir $dir/log $dir/data
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  utils/validate_data_dir.sh --no-text $dir   
     
  # train
  dir=$data_fmllr/train
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir $alidir \
     $dir data/$lang/train $gmmdir $dir/log $dir/data
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  utils/validate_data_dir.sh --no-text $dir   

  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=$dbndir #exp/dnn4_pretrain-dbn/indbn/$lang
  # (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log

  $cuda_cmd $dir/log/pretrain_dbn.log \
  steps/nnet/pretrain_dbn.sh --nn-depth 6 --hid-dim 1024 \
	--cmvn-opts "--norm-means=true --norm-vars=true" \
	--delta-opts "--delta-order=2" --splice 5 \
	--rbm-iter 20 $data_fmllr/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=$nnetdir  # exp/dnn4_pretrain-dbn_dnn/$lang
  ali=$alidir   # ${gmmdir}_ali
  feature_transform=$dbndir/final.feature_transform
  dbn=$dbndir/6.dbn
  # (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 data/$lang/lang $ali $ali $dir

  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --num-threads 6 --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    $gmmdir/graph $data_fmllr/dev $dir/decode_dev &
  steps/nnet/decode.sh --num-threads 6 --nj $decode_nj --cmd "$decode_cmd" --acwt 0.2 \
    $gmmdir/graph $data_fmllr/eval $dir/decode_eval &
  wait    
fi

# # Sequence training using sMBR criterion, we do Stochastic-GD 
# # with per-utterance updates. We use usually good acwt 0.1
# dir=exp/$L/dnn4_pretrain-dbn_dnn_smbr
# srcdir=exp/$L/dnn4_pretrain-dbn_dnn
# acwt=0.2
# 
# if [ $stage -le 3 ]; then
#   # First we generate lattices and alignments:
#   steps/nnet/align.sh --nj $train_nj --cmd "$train_cmd" \
#     $data_fmllr/train data/$L/lang $srcdir ${srcdir}_ali || exit 1;
#   steps/nnet/make_denlats.sh --nj $train_nj --cmd "$decode_cmd" --acwt $acwt \
#     --lattice-beam 10.0 --beam 18.0 \
#     $data_fmllr/train data/$L/lang $srcdir ${srcdir}_denlats || exit 1;
# fi
# 
# if [ $stage -le 4 ]; then
#   # Re-train the DNN by 6 iterations of sMBR 
#   steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt \
#     --do-smbr true \
#     $data_fmllr/train data/$L/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
#   # Decode
#   for ITER in 1 6; do
#     steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
#       --nnet $dir/${ITER}.nnet --acwt $acwt \
#       $gmmdir/graph $data_fmllr/eval $dir/decode_eval_it${ITER} || exit 1
#   done 
# fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
