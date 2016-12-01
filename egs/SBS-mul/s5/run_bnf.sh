#!/bin/bash
# Generate the bn feats by doing a single feed-fwd pass on the BN-DNN.
# Then append the regular feats to bn feats

. ./path.sh
. ./cmd.sh

set -e
set -o pipefail
set -u

use_gpu=yes
remove_last_components=4 # the number of components to remove from the top to get to the bottleneck layer

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

TRAIN_LANG=$1
TEST_LANG=$2
data_fmllr=$3    # data-fmllr-tri3c/${TEST_LANG} (this is where std feats is present)
nnetbndir=$4     # exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/monosoftmax_dt_bn
data_bn=$5		 # data-bn/${TEST_LANG}/monosoftmax_dt_bn  (this is where we save bn feats)
data_fmllr_bn=$6 # data-fmllr-tri3c-bn/${TEST_LANG}/monosoftmax_dt_bn (this is where we save the std+bn feats)

UNILANG_CODE=$(echo $TRAIN_LANG |sed 's/ /_/g')
rm -rf $data_bn $data_fmllr_bn

 # Generate the bn feats by doing a single feed-fwd pass
 # eval
 for lang in ${TRAIN_LANG} ${TEST_LANG}; do
    srcdatadir=$data_fmllr/$lang/eval
    dir=$data_bn/$lang/eval    
    steps/nnet/make_bn_feats.sh --nj 1 --use-gpu $use_gpu --cmd "$train_cmd" --remove-last-components $remove_last_components \
		$dir $srcdatadir $nnetbndir $dir/{log,data} || exit 1;	
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    utils/validate_data_dir.sh --no-text $dir
  done
    
  # dev
  for lang in ${TRAIN_LANG} ${TEST_LANG}; do
    srcdatadir=$data_fmllr/$lang/dev
    dir=$data_bn/$lang/dev
    steps/nnet/make_bn_feats.sh --nj 1 --use-gpu $use_gpu --cmd "$train_cmd" --remove-last-components $remove_last_components \
		$dir $srcdatadir $nnetbndir $dir/{log,data} || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    utils/validate_data_dir.sh --no-text $dir
  done
    
  # train
  for lang in ${UNILANG_CODE} ${TEST_LANG}; do
    srcdatadir=$data_fmllr/$lang/train
    dir=$data_bn/$lang/train
    steps/nnet/make_bn_feats.sh --nj 1 --use-gpu $use_gpu --cmd "$train_cmd" --remove-last-components $remove_last_components \
		$dir $srcdatadir $nnetbndir $dir/{log,data} || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    utils/validate_data_dir.sh --no-text $dir
  done
  
  # Append the bn feats to std feats 
  # eval
  for lang in ${TRAIN_LANG} ${TEST_LANG}; do
    srcdatadir1=$data_fmllr/$lang/eval
    srcdatadir2=$data_bn/$lang/eval
    dir=$data_fmllr_bn/$lang/eval
    #steps/append_feats.sh --cmd "$train_cmd" --nj 4 data-bnf/SW/train/ data-fmllr-tri3c/SW/SW/train  data-fmllr-tri3c-bnf/SW/SW/train data-fmllr-tri3c-bnf/SW/SW/train/{log,data}
    steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
		$srcdatadir1 $srcdatadir2 $dir \
		$dir/{log,data} || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    utils/validate_data_dir.sh --no-text $dir    
  done
  
  # dev
  for lang in ${TRAIN_LANG} ${TEST_LANG}; do
    srcdatadir1=$data_fmllr/$lang/dev
    srcdatadir2=$data_bn/$lang/dev
    dir=$data_fmllr_bn/$lang/dev
    steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
		$srcdatadir1 $srcdatadir2 $dir \
		$dir/{log,data} || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    utils/validate_data_dir.sh --no-text $dir
  done
  
  # train
  for lang in ${UNILANG_CODE} ${TEST_LANG}; do
    echo "creating training data for $lang"   
    srcdatadir1=$data_fmllr/$lang/train
    srcdatadir2=$data_bn/$lang/train
    dir=$data_fmllr_bn/$lang/train
    steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
		$srcdatadir1 $srcdatadir2 $dir \
		$dir/{log,data} || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    utils/validate_data_dir.sh --no-text $dir
    
    # split the data : 90% train 10% cross-validation (held-out)
    utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
  done  

exit 0;
