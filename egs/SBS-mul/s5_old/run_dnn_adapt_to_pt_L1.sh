#!/bin/bash

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

# ./run_dnn_adapt_to_pt_L1.sh --stage 2 --replace-softmax "false" "SW" exp/tri3b_map_SW_pt  exp/dnn4_pretrain-indbn_dnn/nnet_SW_pt/nnet.init data-fmllr-tri3b exp/dnn4_pretrain-indbn_dnn/nnet_SW_pt
# ./run_dnn_adapt_to_pt_L1.sh --stage 2 --replace-softmax "true"  "SW" exp/tri3b_map_SW_pt  exp/dnn4_pretrain-indbn_dnn/nnet_SW_pt/nnet.init data-fmllr-tri3b exp/dnn4_pretrain-indbn_dnn/nnet_SW_pt
#                                                              
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0 # resume training with --stage=N
feats_nj=4
train_nj=8
decode_nj=4
train_iters=20
precomp_dnn=exp/dnn4_pretrain-outdbn_dnn/final.nnet
replace_softmax=
lang="SW" # we want the DNN to fine-tune to this language using the probabilistic transcriptions from this language

gmmdir=exp/tri3b_map_${lang}_pt
data_fmllr=data-fmllr-tri3b
nnet_dir=exp/dnn4_pretrain-outdbn_dnn
# End of config.

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: $0 [options] <lang code> <gmmdir> <precomputed dnn> <fmllr fea-dir> <nnet output dir>" 
   echo "e.g.: $0 --replace-softmax true SW exp/tri3b_map_SW_pt exp/pretrain-dnn/final.nnet data-fmllr-tri3b exp/dnn"
   echo ""
fi

lang=$1
gmmdir=$2
precomp_dnn=$3
data_fmllr=$4
nnet_dir=$5

for f in $gmmdir/final.mdl $gmmdir/post.*.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  graph_dir=$gmmdir/graph_oracle_LG      
  [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$lang/lang_test_oracle_LG $gmmdir $graph_dir >& $graph_dir/mkgraph.log; }  
fi

if [ $stage -le 1 ]; then
  # Store fMLLR features, so we can train and test on them easily
    
  # eval
  #for lang in $SBS_LANG; do
	dir=$data_fmllr/${lang}_pt/eval
	steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
	    --transform-dir $gmmdir/decode_eval_oracle_LG_${lang} \
		$dir data/$lang/eval $gmmdir $dir/log $dir/data || exit 1
	steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
	utils/validate_data_dir.sh --no-text $dir	
  #done
  
  # dev
  #for lang in $SBS_LANG; do
  dir=$data_fmllr/${lang}_pt/dev
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
	   --transform-dir $gmmdir/decode_dev_oracle_LG_${lang} \
       $dir data/$lang/dev $gmmdir $dir/log $dir/data || exit 1
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  utils/validate_data_dir.sh --no-text $dir  
  #done
  
  # train
  #for lang in $SBS_LANG; do
  dir=$data_fmllr/${lang}_pt/train
  steps/nnet/make_lda_feats.sh --nj $feats_nj --cmd "$train_cmd" \
    --transform-dir $gmmdir \
	$dir data/$lang/train $gmmdir $dir/log $dir/data || exit 1
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  utils/validate_data_dir.sh --no-text $dir  
  
  # split the train data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
  #done 
  
fi

labels_trf="\"ark:gunzip -c ${gmmdir}/post.*.gz| post-to-pdf-post $gmmdir/final.mdl ark:- ark:- |\" "
labels_cvf=${labels_trf}
echo "soft-labels = ${labels_trf}"

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy. 
  ali=$gmmdir
  feature_transform=  # calculate unsupervised feat xform based on the adaptation data in steps/nnet/train_pt.sh
  for train_iters in {10..10..1}; do  #{1..5..1}
    for l1_penalty_fac in -11 -10.5 -10 -9.5 -9.0 -8.5 -8.0 -7.5 -7.0; do  #{1..5..1}    
	  dir=${nnet_dir}_xentit_${train_iters}_l1_${l1_penalty_fac}; mkdir -p $dir 
	  nnet_init=$dir/nnet.init
	  l1_penalty=`perl -e 'print STDOUT exp('$l1_penalty_fac');'`;  
	  if [[ ${replace_softmax} == "true" ]]; then 
		perl utils/nnet/renew_nnet_softmax.sh $gmmdir/final.mdl ${precomp_dnn} ${nnet_init}
	  else
		ln -s $PWD/${precomp_dnn} ${nnet_init}
	  fi  
	  echo "nnet_init = ${nnet_init}, train_iters = ${train_iters}, l1 = ${l1_penalty}"
	  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
	   #Initialize NN training with the hidden layers of a DNN
	  $cuda_cmd $dir/log/train_nnet.log \
		steps/nnet/train_pt.sh --nnet-init ${nnet_init} --hid-layers 0 \
		--cmvn-opts "--norm-means=true --norm-vars=true" \
		--delta-opts "--delta-order=2" --splice 5 \
		--learn-rate 0.008 \
		--labels-trainf  ${labels_trf} \
		--labels-crossvf ${labels_cvf} \
		--copy-feats "false" \
		--train-iters ${train_iters} \
		--train-opts "--l1-penalty ${l1_penalty}" \
		$data_fmllr/${lang}_pt/train_tr90 $data_fmllr/${lang}_pt/train_cv10 dummy_lang $ali $ali $dir || exit 1; 
		
	  echo "Done training nnet in: $dir"	  
	  ./run_oracle_LG.sh --stage 7 --nnet-dir $dir --data-fmllr $data_fmllr/${lang}_pt &	   
	done  
  done
  wait
  exit 0; # run lang dependent LM decoding - use ./run_oracle_LG.sh  
  
  #for lang in $SBS_LANG; do  
  ## Decode (reuse HCLG graph)
  #steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    #$gmmdir/graph $data_fmllr/$lang/dev $dir/decode_dev_$lang || exit 1;
  #steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    #$gmmdir/graph $data_fmllr/$lang/eval $dir/decode_eval_$lang || exit 1;
  #done  
fi

exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done