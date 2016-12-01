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

# /run_dnn_adapt_to_mono_pt.sh "SW" exp/tri3c_map/SW exp/tri3cpt_ali/SW exp/dnn4_pretrain-dbn_dnn/SW/indbn/final.nnet data-fmllr-tri3c_map/SW exp/dnn4_pretrain-dbn_dnn/SW/indbn_pt

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0 # resume training with --stage=N
feats_nj=4
train_nj=8
decode_nj=4
train_iters=20
l2_penalty=0
splice=5         # temporal splicing
splice_step=1    # stepsize of the splicing (1 == no gap between frames)
hid_dim=
hid_layers=
precomp_dbn=
precomp_dnn=
bn_dim=
replace_softmax=true
train_dbn=false
# End of config.

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

if [ $# != 7 ]; then
   echo "Usage: $0 [options] <train lang code> <test lang code> <gmmdir> <precomputed dnn> <fmllr fea-dir> <nnet output dir>" 
   echo "e.g.: $0 --replace-softmax true  \"AR CA HG MD UR\" \"SW\" exp/tri3b_map_SW_pt exp/pretrain-dnn/final.nnet data-fmllr-tri3b exp/dnn"
   echo ""
   exit 1;
fi

TRAIN_LANG=$1
TEST_LANG=$2     # "SW" (we want the DNN to fine-tune to this language using its PT's)
gmmdir=$3        # exp/tri3c/${TEST_LANG}
alidir=$4        # exp/tri3cpt_ali/${TEST_LANG}
data_fmllr=$5    # data-fmllr-tri3c/${TEST_LANG}
nnetinitdir=$6   # exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}
nnetoutdir=$7      # exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}


UNILANG_CODE=$(echo $TRAIN_LANG |sed 's/ /_/g')
[[ ! -z ${precomp_dnn} ]] && train_dbn=false
[[ ! -z ${precomp_dbn} ]] && train_dbn=true
$train_dbn && echo "Will use a DBN to init target DNN" || echo "Will either use - a) randomly initialized DNN or b) supplied DNN - to init target DNN"

for f in $gmmdir/final.mdl $alidir/post.*.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

#if [ $stage -le 0 ]; then
  #graph_dir=$gmmdir/graph_oracle_LG      
  #[[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$lang/lang_test_oracle_LG $gmmdir $graph_dir >& $graph_dir/mkgraph.log; }  
#fi

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
     --transform-dir ${gmmdir} \
     $dir data/$lang/train $gmmdir $dir/log $dir/data || exit 1
   steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
   utils/validate_data_dir.sh --no-text $dir
  done
  
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

if [ $stage -le 2 ]; then
	if $train_dbn; then
	# First check for pre-computed DBN dir. Then try pre-computed DNN dir. If both fail, generate DBN now.
	  mkdir -p $nnetinitdir
	  if [[ ! -z ${precomp_dbn} ]]; then
		echo "using pre-computed dbn ${precomp_dbn}"
		
		# copy the dbn and feat xform from dbn dir	
		cp -r ${precomp_dbn} $nnetinitdir 
		
		# Comment lines below if you want to compute feature xform estmn from the adaptation data (SBS)
		#cp $(dirname ${precomp_dbn})/final.feature_transform $dir
		#feature_transform=$dir/final.feature_transform
		#feature_transform_opt=$(echo "--feature-transform $feature_transform")	  
	  else
	    echo "train with a randomly initialized DBN"
	    
		# Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)	
		(tail --pid=$$ -F $nnetinitdir/log/pretrain_dbn.log 2>/dev/null)& # forward log
	
	    if [[ ! -z $bn_layer ]]; then
	      $cuda_cmd $nnetinitdir/log/pretrain_dbn.log \
		  local/nnet/pretrain_dbn.sh --nn-depth $hid_layers --hid-dim $hid_dim \
		    --bn-layer $bn_layer --bn-dim $bn_dim		\
		    --cmvn-opts "--norm-means=true --norm-vars=true" \
		    --delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
		    --rbm-iter 20 $data_fmllr/${UNILANG_CODE}/train $nnetinitdir || exit 1;
	    else
		  $cuda_cmd $nnetinitdir/log/pretrain_dbn.log \
		  steps/nnet/pretrain_dbn.sh --nn-depth $hid_layers --hid-dim $hid_dim \
		    --cmvn-opts "--norm-means=true --norm-vars=true" \
			--delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
			--rbm-iter 20 $data_fmllr/${UNILANG_CODE}/train $nnetinitdir || exit 1;
	    fi  
	   fi
	fi
fi

labels_trf="\"ark:gunzip -c ${alidir}/post.*.gz| post-to-pdf-post $alidir/final.mdl ark:- ark:- |\" "
labels_cvf=${labels_trf}
echo "soft-labels = ${labels_trf}"
dir=$nnetoutdir
if [ $stage -le 3 ]; then
  # Train the DNN optimizing per-frame cross-entropy. 
  ali=$alidir
  feature_transform=  # calculate unsupervised feat xform based on the adaptation data in steps/nnet/train_pt.sh  
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  if $train_dbn; then
	  # Initialize NN training with a DBN
	  echo "using DBN to start DNN training"
	  dbn=${nnetinitdir}/${hid_layers}.dbn
	  $cuda_cmd $dir/log/train_nnet.log \
	    local/nnet/train_pt.sh  --dbn $dbn --hid-layers 0 \
	    --cmvn-opts "--norm-means=true --norm-vars=true" \
	    --delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
	    --learn-rate 0.008 \
	    --labels-trainf  ${labels_trf} \
		--labels-crossvf ${labels_cvf} \
		--copy-feats "false" \
		--train-iters ${train_iters} \
		--train-opts "--l2-penalty ${l2_penalty}" \
	    $data_fmllr/${TEST_LANG}/train_tr90 $data_fmllr/${TEST_LANG}/train_cv10 data/${TEST_LANG}/lang $ali $ali $dir || exit 1;
  elif [[ -f $precomp_dnn ]]; then
      echo "using pre-computed dnn ${precomp_dnn} to start DNN training"		
	  # replace the softmax layer of the precomp dnn with a random init layer
	  [[ ! -d $nnetinitdir ]] && mkdir -p $nnetinitdir 
	  nnet_init=$nnetinitdir/nnet.init
	  if [[ ${replace_softmax} == "true" ]]; then
	   echo "replacing the softmax layer of $precomp_dnn"
	   perl local/nnet/renew_nnet_softmax.sh $gmmdir/final.mdl ${precomp_dnn} ${nnet_init}
	  else
	   echo "not replacing the softmax layer of $precomp_dnn"
	   cp ${precomp_dnn} ${nnet_init}
	  fi
	  echo "nnet_init = ${nnet_init}"	  
	  #Initialize NN training with the hidden layers of a DNN
	  $cuda_cmd $dir/log/train_nnet.log \
	  local/nnet/train_pt.sh --nnet-init ${nnet_init} --hid-layers 0 \
		--cmvn-opts "--norm-means=true --norm-vars=true" \
		--delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
		--learn-rate 0.008 \
		--labels-trainf  ${labels_trf} \
		--labels-crossvf ${labels_cvf} \
		--copy-feats "false" \
		--train-iters ${train_iters} \
		--train-opts "--l2-penalty ${l2_penalty}" \
	  $data_fmllr/${TEST_LANG}/train_tr90 $data_fmllr/${TEST_LANG}/train_cv10 dummy_lang $ali $ali $dir || exit 1;
  else # sometimes we may not have precomputed DNN in which case let train_pt.sh initialize it	  
	  #Initialize NN training with the hidden layers of a DNN
	  $cuda_cmd $dir/log/train_nnet.log \
	  local/nnet/train_pt.sh --hid-layers $hid_layers --hid-dim  $hid_dim \
	    ${bn_dim:+ --bn-dim $bn_dim} \
		--cmvn-opts "--norm-means=true --norm-vars=true" \
		--delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
		--learn-rate 0.008 \
		--labels-trainf  ${labels_trf} \
		--labels-crossvf ${labels_cvf} \
		--copy-feats "false" \
		--train-iters ${train_iters} \
		--train-opts "--l2-penalty ${l2_penalty}" \
	  $data_fmllr/${TEST_LANG}/train_tr90 $data_fmllr/${TEST_LANG}/train_cv10 dummy_lang $ali $ali $dir || exit 1; 
  fi
  echo "Done training nnet in: $nnetoutdir"
fi

# Decode
if [ $stage -le 4 ]; then
  # Nnet decode:
  exp_dir=$gmmdir
  dir=$nnetoutdir

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
