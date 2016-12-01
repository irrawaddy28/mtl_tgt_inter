#!/bin/bash

# Copyright 2015-2016  University of Illinois (Author: Amit Das)
# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script creates a multilingual nnet. The training is done in 3 stages:
# 1. FMLLR features: It generates fmllr features from the multilingual training data.
# 2. DBN Pre-training: To initialize the nnet, it can 
#    a) train a dbn using the multilingual fmllr features or
#    b) use an existing pre-trained dbn or dnn from the user
# 3. DNN cross-entropy training: It fine-tunes the initialized nnet using 
#    the multilingual training data (deterministic transcripts).
#

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0 # resume training with --stage=N
feats_nj=4
train_nj=8
decode_nj=4
train_iters=20
l2_penalty=0
transform_dir_train=
splice=5         # temporal splicing
splice_step=1    # stepsize of the splicing (1 == no gap between frames)
hid_dim=
hid_layers=
precomp_dbn=
bn_layer=
bn_dim=
replace_softmax=true

# Frame weighting options
threshold=0.7   # If provided, use frame thresholding -- keep only frames whose
                # best path posterior is above this value.  
use_soft_counts=true    # Use soft-posteriors as targets for PT data
disable_upper_cap=true
acwt=0.2
parallel_opts="--num-threads 6"
# End of config.

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 [options] <lang code> <gmmdir> <ptlatdir> <precomputed dnn> <fmllr fea-dir> <nnet output dir>" 
   echo "e.g.: $0 --replace-softmax true SW exp/tri3b_map_SW_pt exp/pretrain-dnn/final.nnet data-fmllr-tri3b exp/dnn"
   echo ""
   exit 1;
fi


TEST_LANG=$1     # "SW" (we want the DNN to fine-tune to this language using its PT's)
gmmdir=$2        # exp/tri3cpt_ali/${TEST_LANG}
ptlatdir=$3      # exp/tri3cpt_ali/${TEST_LANG}/decode_train
precomp_dnn=$4   # exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/indbn/final.nnet
data_fmllr=$5    # data-fmllr-tri3c/${TEST_LANG}
nnet_dir=$6      # exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/indbn_pt

[ -z $transform_dir_train ] && transform_dir_train=$gmmdir

for f in $gmmdir/final.mdl $ptlatdir/lat.1.gz ; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  graph_dir=$gmmdir/graph_text_G_${TEST_LANG}
  [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/${TEST_LANG}/lang_test_text_G $gmmdir $graph_dir >& $graph_dir/mkgraph.log; }  
fi

if [ $stage -le 1 ]; then
  # Store fMLLR features, so we can train on them easily
    
  # eval
  for lang in ${TEST_LANG}; do
	dir=$data_fmllr/$lang/eval
	steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
		--transform-dir $gmmdir/decode_eval_$lang \
		$dir data/$lang/eval $gmmdir $dir/log $dir/data || exit 1
	steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
	utils/validate_data_dir.sh --no-text $dir	
  done
  
  # dev
  for lang in ${TEST_LANG}; do
   dir=$data_fmllr/$lang/dev
   steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev_$lang \
     $dir data/$lang/dev $gmmdir $dir/log $dir/data || exit 1
   steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
   utils/validate_data_dir.sh --no-text $dir  
  done
  
  # train
  for lang in ${TEST_LANG}; do
   dir=$data_fmllr/$lang/train
   steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir ${transform_dir_train} \
     $dir data/$lang/train $gmmdir $dir/log $dir/data || exit 1
   steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
   utils/validate_data_dir.sh --no-text $dir
  done
  
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

decode_dir=$ptlatdir  # this is where pt lattices are expected lat.*.gz. Usually in $gmmdir/decode_train
best_path_dir=$nnet_dir/bestpath_ali 
postdir=$nnet_dir/post_train_thresh${threshold:+_$threshold}  
# Get frame posteriors of the best path in the PT lattice
if [ $stage -le 2 ]; then
  
  local/posts_and_best_path_weights.sh --acwt $acwt --threshold $threshold \
  --use-soft-counts $use_soft_counts --disable-upper-cap $disable_upper_cap \
  $gmmdir $decode_dir $best_path_dir $postdir  
fi

labels_trf="scp:$postdir/post.scp"
labels_cvf=$labels_trf
frame_weights="scp:$postdir/frame_weights.scp"
dir=$nnet_dir
# Start DNN training
if [ $stage -le 3 ]; then
  # Train the DNN optimizing per-frame cross-entropy. 
  ali=$gmmdir # used only to provide tree and transition model to the nnet
  feature_transform=  # calculate unsupervised feat xform of the nnet based on the adaptation data
  mkdir -p $nnet_dir
  nnet_init=$dir/nnet.init  
  # Train
  if [[ -f $precomp_dbn ]]; then
  # Initialize NN training with a DBN
  echo "using DBN to start DNN training"
  dbn=${precomp_dbn}
  $cuda_cmd $dir/log/train_nnet.log \
    local/nnet/train_pt.sh  --dbn $dbn --hid-layers 0 \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
    --learn-rate 0.008 \
    --labels-trainf  ${labels_trf} \
	--labels-crossvf ${labels_cvf} \
	--frame-weights  ${frame_weights} \
	--copy-feats "false" \
	--train-iters ${train_iters} \
	--train-opts "--l2-penalty ${l2_penalty}" \
    $data_fmllr/${TEST_LANG}/train_tr90 $data_fmllr/${TEST_LANG}/train_cv10 data/${TEST_LANG}/lang $ali $ali $dir || exit 1;
  elif [[ -f $precomp_dnn ]]; then	  	
	  if [[ ${replace_softmax} == "true" ]]; then
	   echo "replacing the softmax layer of $precomp_dnn"
	   perl local/nnet/renew_nnet_softmax.sh $gmmdir/final.mdl ${precomp_dnn} ${nnet_init}
	  else
	   echo "not replacing the softmax layer of $precomp_dnn"
	   cp ${precomp_dnn} ${nnet_init}
	  fi
	  echo "nnet_init = ${nnet_init}"	  
	  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
	  #Initialize NN training with the hidden layers of a DNN
	  $cuda_cmd $dir/log/train_nnet.log \
	  local/nnet/train_pt.sh --nnet-init ${nnet_init} --hid-layers 0 \
		--cmvn-opts "--norm-means=true --norm-vars=true" \
		--delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
		--learn-rate 0.008 \
		--labels-trainf  ${labels_trf} \
		--labels-crossvf ${labels_cvf} \
		--frame-weights  ${frame_weights} \
		--copy-feats "false" \
		--train-iters ${train_iters} \
		--train-opts "--l2-penalty ${l2_penalty}" \
	  $data_fmllr/${TEST_LANG}/train_tr90 $data_fmllr/${TEST_LANG}/train_cv10 dummy_lang $ali $ali $dir || exit 1;
  else # sometimes we may not have precomputed DNN in which case let train_pt.sh initialize it
	  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
	  #Initialize NN training with the hidden layers of a DNN
	  $cuda_cmd $dir/log/train_nnet.log \
	  local/nnet/train_pt.sh --hid-layers $hid_layers --hid-dim  $hid_dim \
		${bn_dim:+ --bn-dim $bn_dim} \
		--cmvn-opts "--norm-means=true --norm-vars=true" \
		--delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
		--learn-rate 0.008 \
		--labels-trainf  ${labels_trf} \
		--labels-crossvf ${labels_cvf} \
		--frame-weights  ${frame_weights} \
		--copy-feats "false" \
		--train-iters ${train_iters} \
		--train-opts "--l2-penalty ${l2_penalty}" \
	  $data_fmllr/${TEST_LANG}/train_tr90 $data_fmllr/${TEST_LANG}/train_cv10 dummy_lang $ali $ali $dir || exit 1; 
  fi  
  echo "Done training nnet in: $nnet_dir"  
fi

# Decode
if [ $stage -le 4 ]; then
  # Nnet decode:
  exp_dir=$gmmdir
  dir=$nnet_dir
  for L in ${TEST_LANG}; do
    echo "Decoding $L"    
    graph_dir=${exp_dir}/graph_text_G_$L
    [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_text_G $exp_dir $graph_dir || exit 1; }
  
    (steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
	  $graph_dir $data_fmllr/$L/dev $dir/decode_dev_text_G_$L || exit 1;) &
    (steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
      $graph_dir $data_fmllr/$L/eval $dir/decode_eval_text_G_$L || exit 1;) &     
    (cd $dir; ln -s  decode_dev_text_G_$L decode_dev_$L; ln -s decode_eval_text_G_$L decode_eval_$L)    
  done
fi

echo "Done: `date`"

exit 0;

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
