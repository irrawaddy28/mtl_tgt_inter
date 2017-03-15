#!/bin/bash

# Copyright 2015  University of Illinois (Author: Amit Das)
# Copyright 2012-2015  Brno University of Technology (Author: Karel Vesely)

# Apache 2.0

# This example script trains Multi-lingual DNN with <BlockSoftmax> output, using FBANK features.
# The network is trained on multiple languages simultaneously, creating a separate softmax layer
# per language while sharing hidden layers across all languages.
# The script supports arbitrary number of languages.

# Notes:
# 1. The test language should always be in the first block (first element of the comma separated lists)
# 2. Assumes the fMLLR features for each block are already available. This script only combines them.

. ./cmd.sh
. ./path.sh

stage=0

# dnn options
nnet_type=dnn_small # dnn_small | dnn | bn
remove_last_components=2
dnn_init=
hid_layers=
hid_dim=
cmvn_opts=   # speaker specific cmvn for i/p features. For mean+var normalization, use "--norm-means=true --norm-vars=true"
delta_order=0
splice=5         # temporal splicing
splice_step=1    # stepsize of the splicing (1 == no gap between frames)
test_block_csl=1

# Multi-task training options
objective_csl="xent:xent:xent"
lang_weight_csl="1.0:1.0:1.0"
lat_dir_csl="-:-:-"    # pt or semisup lattices
data_type_csl="dt:dt:dt"
label_type_csl="s:s:s" # s = senone, p = monophone, f = feature (for unsup data, feats acts as labels)
dup_and_merge_csl="0>>1:0>>2:0>>3" # x>>y means: create x copies of data and use them to train the task in block y
renew_nnet_type="parallel"  # can be "parallel", "blocksoftmax", or an empty string (single softmax for all tasks)
renew_nnet_opts=            # options for the renew nnnet. Details about these options in local/nnet/renew_nnet_*.sh scripts. For e.g., for parallel nnet, refer to the examples in local/nnet/renew_nnet_parallel.sh;
parallel_nhl_opts=  # no. of hidden layers in the MTL tasks, Default 0 hidden layers for all tasks
parallel_nhn_opts=  # no. of hidden neurons in the MTL tasks, Default 1024 neurons for all tasks
randomizer_size=32768  # Maximum number of samples we want to have in memory at once
minibatch_size=256     # num samples per mini-batch
use_gpu="yes"

# Frame weighting options
threshold_default=0.7
threshold_csl=   # If provided, use frame thresholding -- keep only frames whose
                    # best path posterior is above this value.
use_soft_counts=true    # Use soft-posteriors as targets for PT data
disable_upper_cap=true
acwt=0.2
parallel_opts="--num-threads 6"
# End of config.

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

set -euo pipefail

# The first language in csl should always be the test language
# ./run_multilingual.sh --dnn-init "exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax/decode_dev_text_G_SW/final.nnet" --data-type-csl "pt:dt:unsup" --lang-weight-csl "1.0:1.0:0.0" --threshold-csl "0.7:1.0:0.8" --lat-dir-csl "exp/tri3cpt_ali/SW/decode_train:-:exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax/decode_unsup_4k_SW" --dup-and-merge-csl "4>>1:0>>2:1>>3" "SW:AR_CA_HG_MD_UR:SW" "exp/tri3cpt_ali/SW:exp/tri3b_ali/SW:exp/tri3cpt_ali/SW" "data-fmllr-tri3c/SW/SW/train:data-fmllr-tri3b/SW/AR_CA_HG_MD_UR/train:data-fmllr-tri3c/SW/SW/unsup_4k_SW" "data-fmllr-tri3c_map/SW/combined" "exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax2c"

if [ $# != 5 ]; then
   echo "Usage: $0 [options] <lang code csl> <ali dir csl> <data dir csl> <data output dir> <nnet output dir>" 
   echo "e.g.: $0 --dnn-init nnet.init --lang-weight-csl 1.0:1.0 SW:HG exp/tri3b_ali/SW:exp/tri3b_ali/HG data-fmllr-tri3b/SW/train:data-fmllr-tri3b/HG/train data-fmllr-tri3b/combined exp/dnn "
   echo ""
fi

lang_code_csl=$1 # AR_CA_HG_MD_UR:SW:SW
ali_dir_csl=$2   # exp/tri3b_ali/SW:exp/tri3cpt_ali/SW:
data_dir_csl=$3  # data-fmllr-tri3c/SW/SW/train:data-fmllr-tri3c/SW/AR_CA_HG_MD_UR/train
data=$4          # this is where we'll save the combined data for multisoftmax training 
nnet_dir=$5      # o/p dir of multisoftmax nnet

# Convert csl string to bash array (accept separators ',' ':'),
lang_code=($(echo $lang_code_csl | tr ',:' ' ')) 
ali_dir=($(echo $ali_dir_csl | tr ',:' ' '))
data_dir=($(echo $data_dir_csl | tr ',:' ' '))

# Make sure we have same number of items in the lists,
! [ ${#lang_code[@]} -eq ${#ali_dir[@]} -a ${#lang_code[@]} -eq ${#data_dir[@]} ] && \
  echo "Non-matching number of 'csl' items: lang_code ${#lang_code[@]}, ali_dir ${#ali_dir[@]}, data_dir ${#data_dir[@]}" && \
  exit 1
num_langs=${#lang_code[@]}

# Parse objective function csl
if [ -z "$objective_csl" ]; then
  objective_csl=$(echo $(printf "%s:" $(for i in `seq 1 $num_langs`; do echo xent; done))|sed 's/.$//')
fi

# Parse lang weight csl
if [ -z "$lang_weight_csl" ]; then
  lang_weight_csl=$(echo $(printf "%s:" $(for i in `seq 1 $num_langs`; do echo 1; done))|sed 's/.$//')
fi

# Parse data type csl
if [ -z "$data_type_csl" ]; then
  data_type_csl=$(echo $(printf "%s:" $(for i in `seq 1 $num_langs`; do echo dt; done))|sed 's/.$//')
fi
data_type=($(echo $data_type_csl | tr ',:' ' '))
for ((i=0; i<$num_langs; i++)); do
  type=${data_type[$i]}
  if ! [ "$type" == "dt" -o "$type" == "pt" -o "$type" == "semisup" -o "$type" == "unsup"  ]; then
    echo "data type at position $((i + 1)) is \"$type\" is not supported"
    exit 1
  fi
done

# Parse label type csl
if [ -z "$label_type_csl" ]; then
  label_type_csl=$(echo $(printf "%s:" $(for i in `seq 1 $num_langs`; do echo s; done))|sed 's/.$//')
fi
label_type=($(echo $label_type_csl | tr ',:' ' '))
for ((i=0; i<$num_langs; i++)); do
  type=${label_type[$i]}
  if ! [ "$type" == "s" -o "$type" == "p" -o "$type" == "f" ]; then
    echo "label type at position $((i + 1)) is \"$type\" is not supported"
    exit 1
  fi
done

# Parse label confidence threshold csl
if [ -z "$threshold_csl" ]; then
  for ((i=0; i<$num_langs; i++)); do
    if [ "${data_type[$i]}" ==  "dt" ]; then
      threshold_csl="$threshold_csl:0.0"
    else
      threshold_csl="${threshold_csl}:${threshold_default}"
    fi
  done  
  threshold_csl=${threshold_csl#*:}
fi
threshold_list=($(echo $threshold_csl | tr ',:' ' '))
for ((i=0; i<$num_langs; i++)); do
  type=${data_type[$i]}
  threshold=${threshold_list[$i]}
  if [ "$type" ==  "dt" -a "$threshold" != "0.0" ]; then
    echo "data type at position $((i + 1)) is \"dt\" but threshold has a non-zero value $threshold"
    exit 1
  fi
done

# Parse duplicate copies csl
make_num_copies="false"
if [ -z "$dup_and_merge_csl" ]; then
  dup_and_merge_csl=$(echo $(printf "%s:" $(for i in `seq 1 $num_langs`; do echo "0>>0"; done))|sed 's/.$//')
fi
dup_and_merge=($(echo $dup_and_merge_csl | tr ',:' ' '))

# Parse the decoding task csl
test_block=($(echo $test_block_csl | tr ',:' ' '))
for ((i=0; i<${#test_block[@]}; i++)); do
    [ "${test_block[$i]}" -lt "0" ] && echo "test_block[$i]=${test_block[$i]} cannot be less than 0" && exit 1;
    [ "${test_block[$i]}" -gt "$num_langs" ] && echo "test_block[$i]=${test_block[$i]} cannot be greater than $num_langs" && exit 1;
done

# Check if all the parsed arrays have the same number of elements
if [ "$num_langs" -ne "${#data_type[@]}" -o "$num_langs" -ne "${#dup_and_merge[@]}" ]; then
  echo "Non-matching number of 'csl' items: data_type ${#data_type[@]}, dup_and_merge ${#dup_and_merge[@]}"
  exit 1;
fi

# Check if all the input directories exist,
echo ""
lat_dir=($(echo "$lat_dir_csl" | tr ',:' ' '))
for i in $(seq 0 $[num_langs-1]); do
  echo "lang = ${lang_code[$i]}, type = ${data_type[$i]}, alidir = ${ali_dir[$i]}, datadir = ${data_dir[$i]}, latdir = ${lat_dir[$i]}, dup_and_merge = ${dup_and_merge[$i]}"
  
  [ ! -d ${ali_dir[$i]} ] && echo  "Missing ${ali_dir[$i]}" && exit 1
  [ ! -d ${data_dir[$i]} ] && echo "Missing ${data_dir[$i]}" && exit 1
  
  # if data_type is pt or semisup, then lat_dir must exist. 
  # if data_type is dt or unsup, then lat_dir should be set to - (meaning empty)
  if [ "${data_type[$i]}" ==  "pt" -o "${data_type[$i]}" ==  "semisup" ]; then
    [ ! -d ${lat_dir[$i]} ] && echo  "Missing ${lat_dir[$i]}" && exit 1
  elif [ "${data_type[$i]}" ==  "dt" -o "${data_type[$i]}" ==  "unsup" ]; then
    [ ${lat_dir[$i]} != "-" ] && echo  "For data type = ${data_type[$i]}, latdir should be set to a hyphen to indicate empty directory" && exit 1  
  fi
done
echo ""

# Make the features,
data_tr90=$data/combined_tr90
data_cv10=$data/combined_cv10
set +e
semisup_present=$(echo ${data_type[@]}|grep -wc "semisup")
unsup_present=$(echo ${data_type[@]}|grep -wc "unsup")
echo "semisup = $semisup_present, unsup = $unsup_present"
set -e
if [ $stage -le 0 ]; then    
  # Make local copy of data-dirs (while adding language-code),  
  tr90=""
  cv10=""	  
  for i in $(seq 0 $[num_langs-1]); do
	code=${lang_code[$i]}
	dir=${data_dir[$i]}
	lab=${label_type[$i]}
	type=${data_type[$i]}
	tgt_dir=$data/${code}_${lab}_$(basename $dir)
	
	echo -e "\nLanguage = $code, Data type = $type, Label Type = $lab, Src feat = $dir, Tgt feat = $tgt_dir"
	
	# Check if feat tgt dir already exists, If so do not recreate it
	# Useful when multiple jobs are accessing the same feat dir
	invalid=1
	if [ -d $tgt_dir ]; then
	  echo "tgt dir = $tgt_dir : exists"
	  if [ "$type" ==  "unsup" ]; then
	    utils/validate_data_dir.sh --no-text ${tgt_dir}
	  else
	    utils/validate_data_dir.sh ${tgt_dir}
	  fi
	  invalid=$?
	fi
		
	# If invalid is still non-zero value, it means we need to create the feat dir
	if [ $invalid -ne 0 ]; then
	
	  echo "tgt dir = $tgt_dir : create new"
	  # if label type is phone, prefix the utt id with "p"
	  if [ "$lab" != "p" ]; then
	    utils/copy_data_dir.sh $dir $tgt_dir || exit 1
	  else
	    utils/copy_data_dir.sh --utt-prefix ${lab}- --spk-prefix ${lab}- $dir $tgt_dir || exit 1
      fi      
	  
	  # Create CV set (10% held-out) only for pt or dt data types; No unsup/semisup in CV
	  #if [ "$type" !=  "semisup" -a "$type" !=  "unsup" ]; then
	  #  utils/subset_data_dir_tr_cv.sh $tgt_dir ${tgt_dir}_tr90 ${tgt_dir}_cv10
	  #  tr90="$tr90 ${tgt_dir}_tr90"
	  #  cv10="$cv10 ${tgt_dir}_cv10"
	  #else
	  #  utils/copy_data_dir.sh $tgt_dir ${tgt_dir}_tr90
	  #  tr90="$tr90 ${tgt_dir}_tr90"
	  #  #tr90="$tr90 ${tgt_dir}"
	  #fi
	  	  
	  # Create CV set (10% held-out); Allow unsup/semisup in CV
	  utils/subset_data_dir_tr_cv.sh $tgt_dir ${tgt_dir}_tr90 ${tgt_dir}_cv10
	  tr90="$tr90 ${tgt_dir}_tr90"
	  cv10="$cv10 ${tgt_dir}_cv10"
	else
	  echo "$tgt_dir already exists and was validated. Skip recreating it"
	fi
	
  done
  
  ## Merge the datasets,
  if [ $invalid -ne 0 ]; then
    if [[ "${semisup_present}" -gt 0  || "${unsup_present}" -gt 0 ]]; then
	  ## If we don't specify --skip-fix "true", the combined scp will exclude utts which do not have text
	  utils/combine_data.sh --skip-fix "true" $data_tr90 $tr90
	  utils/combine_data.sh --skip-fix "true" $data_cv10 $cv10
    else
	  utils/combine_data.sh  $data_tr90 $tr90
	  utils/combine_data.sh  $data_cv10 $cv10
	  ## Validate
	  utils/validate_data_dir.sh $data_tr90
	  utils/validate_data_dir.sh $data_cv10
    fi
  
    echo $tr90 | tr ' ' '\n' > $data_tr90/datasets
    echo $cv10 | tr ' ' '\n' > $data_cv10/datasets
    
    echo -e "\nCombined \"$tr90\" and saved in $data_tr90"
    echo -e "\nCombined \"$cv10\" and saved in $data_cv10" 
  fi  
fi

# Extract the tied-state numbers from transition models,
for i in $(seq 0 $[num_langs-1]); do
  # If data is dt/pt/semisup, alignments exist. If data is unsup, 
  # alignments don't exist and hence we'll use feat dim as the ali dim.
  if [ "${data_type[$i]}" !=  "unsup" ]; then
    ali_dim[i]=$(hmm-info ${ali_dir[i]}/final.mdl | grep pdfs | awk '{ print $NF }')
  else
    feat_dim=$(feat-to-dim --print-args=false "ark:copy-feats scp:${data_dir[i]}/feats.scp ark:- |" - )
    ali_dim[i]=$((feat_dim*(delta_order+1)*(2*splice+1)))
    echo "${data_dir[i]}, raw dim = $feat_dim, network indim = ${ali_dim[i]}"
  fi
done
ali_dim_csl=$(echo ${ali_dim[@]} | tr ' ' ':')

# Total number of DNN outputs (sum of all per-language blocks),
output_dim=$(echo ${ali_dim[@]} | tr ' ' '\n' | awk '{ sum += $i; } END{ print sum; }')
echo "Total number of DNN outputs: $output_dim = $(echo ${ali_dim[@]} | sed 's: : + :g')"

# Objective function string (per-language weights are imported from '$lang_weight_csl'),
objective_function="multitask$(echo ${ali_dim[@]} | tr ' ' '\n' | \
  awk -v crit=$objective_csl -v w=$lang_weight_csl 'BEGIN{ split(w,w_arr,/[,:]/); split(crit,crit_arr,/[,:]/); } { printf(",%s,%d,%s", crit_arr[NR], $1, w_arr[NR]); }')"
echo "Multitask objective function: $objective_function"

# Process the $renew_nnet_opts string: add double quotes before and after the right value 
# e.g. --nnet-proto-opts -:--no-softmax:--output-activation-type <Tanh> --> --nnet-proto-opts "-:--no-softmax:--output-activation-type <Tanh>"
if [ ! -z "$renew_nnet_opts" ]; then
  #renew_nnet_opts=$(echo $renew_nnet_opts |sed -e 's:nnet-proto-opts:nnet-proto-opts :' -e 's/$//')
  #renew_nnet_opts="--nnet-proto-opts -:-:--output-activation-type<Tanh>"
  #renew_nnet_opts="--nnet-proto-opts \"-:-:--no-softmax\""
  echo "renew_nnet_opts = $renew_nnet_opts"
fi

# DNN training will be in $dir, the alignments are prepared beforehand,
#dir=exp/dnn4g-multilingual${num_langs}-$(echo $lang_code_csl | tr ',' '-')-${nnet_type} 
dir=$nnet_dir
[ ! -e $dir ] && mkdir -p $dir
echo -e "$0 $@\n\n\n 
Experiment Settings:\n
lang_code_csl = $lang_code_csl\n
objective_csl = $objective_csl\n
lang_weight_csl = $lang_weight_csl\n
ali_dir_csl  = $ali_dir_csl\n
data_dir_csl = $data_dir_csl\n
ali_dim_csl =  $ali_dim_csl\n
objective_function = $objective_function\n
lat_dir_csl = $lat_dir_csl\n
data_type_csl = $data_type_csl\n
dup_and_merge_csl = $dup_and_merge_csl\n
threshold_csl  = $threshold_csl\n
test_block_csl = $test_block_csl\n
feat dir = $data\n
dnn init = $dnn_init\n
remove last components = $remove_last_components\n
renew_nnet_type = $renew_nnet_type\n
renew_nnet_opts = $renew_nnet_opts\n
dnn out = $dir" > $dir/config

# Make the features and targets for MTL
if [ $stage -le 1 ]; then
  # Step 1: Prepare the feature scp, target posterior scp, and frame weight scp for each task in MTL,
  # Generate the following:
  # (a) Features: $data/combined_tr90/feats.scp, $data/combined_cv10/feats.scp
  # (b) Target posteriors: $dir/ali-post/post_task_<d>.scp, where d = 1, 2, 3  etc
  # (c) Frame weights: $dir/ali-post/frame_weights_task_<d>.scp, where d = 1, 2, 3  etc
  local/make_task_scps.sh --acwt $acwt --use-soft-counts $use_soft_counts --disable-upper-cap $disable_upper_cap \
  ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
  $lang_code_csl $data_type_csl $label_type_csl $data_dir_csl  \
  $ali_dir_csl   $lat_dir_csl   $threshold_csl \
  $dup_and_merge_csl $data $dir/ali-post  
  
  # Step 2: Combine the task specific target posteriors and frame weights into a single MTL task,
  # pasting the ali's, adding language-specific offsets to the posteriors,
  featlen="ark:feat-to-len 'scp:cat $data_tr90/feats.scp $data_cv10/feats.scp |' ark,t:- |" # get number of frames for every utterance,
  post_scp_list=$( echo $(seq 1 $num_langs)| tr ' ' '\n' | awk -v d=$dir '{ printf(" scp:%s/ali-post/post_task_%s.scp", d, $1); }')
  echo "post_scp_list =========== $post_scp_list"| tr ' ' '\n'; echo -e "\n"
  paste-post --allow-partial=true "$featlen" "${ali_dim_csl}" ${post_scp_list} \
    ark,scp:$dir/ali-post/post_combined.ark,$dir/ali-post/post_combined.scp || exit 1
  # pasting the frame weights
  frame_weights_scp_list=$( echo $(seq 1 $num_langs)| tr ' ' '\n' | awk -v d=$dir '{ printf(" %s/ali-post/frame_weights_task_%s.scp", d, $1); }')
  echo "frame_weights_scp_list =========== $frame_weights_scp_list" | tr ' ' '\n'; echo -e "\n"
  copy-vector "scp:cat ${frame_weights_scp_list}|" ark,t,scp:$dir/ali-post/frame_weights_combined.ark,$dir/ali-post/frame_weights_combined.scp || exit 1
fi

# Train the DNN
if [ $stage -le 2 ]; then
  case $nnet_type in
    bn)
    # Bottleneck network (40 dimensional bottleneck is good for fMLLR),
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --hid-layers 2 --hid-dim 1500 --bn-dim 40 \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --feat-type "traps" --splice 5 --traps-dct-basis 6 \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --randomizer-size ${randomizer_size} --minibatch-size ${minibatch_size} \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir
    ;;
    sbn)
    # Stacked Bottleneck Netowork, no fMLLR in between,
    bn1_dim=80
    bn2_dim=30
    # Train 1st part,
    dir_part1=${dir}_part1
    $cuda_cmd ${dir}_part1/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --hid-layers 2 --hid-dim 1500 --bn-dim $bn1_dim \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --feat-type "traps" --splice 5 --traps-dct-basis 6 \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --randomizer-size ${randomizer_size} --minibatch-size ${minibatch_size} \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir_part1
    # Compose feature_transform for 2nd part,
    nnet-initialize <(echo "<Splice> <InputDim> $bn1_dim <OutputDim> $((13*bn1_dim)) <BuildVector> -10 -5:5 10 </BuildVector>") \
      $dir_part1/splice_for_bottleneck.nnet 
    nnet-concat $dir_part1/final.feature_transform "nnet-copy --remove-last-layers=4 $dir_part1/final.nnet - |" \
      $dir_part1/splice_for_bottleneck.nnet $dir_part1/final.feature_transform.part1
    # Train 2nd part,
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --feature-transform $dir_part1/final.feature_transform.part1 \
        --hid-layers 2 --hid-dim 1500 --bn-dim $bn2_dim \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --randomizer-size ${randomizer_size} --minibatch-size ${minibatch_size} \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir
    ;;
    dnn_small)
    # 4 hidden layers, 1024 sigmoid neurons
    if [[ ! -z ${dnn_init} ]]; then
	  nnet_init=$dir/nnet.init	  
      if [ $renew_nnet_type == "parallel" ]; then
        # create a generic network for each task
	    local/nnet/renew_nnet_parallel.sh --remove-last-components $remove_last_components \
          ${renew_nnet_opts:+ $renew_nnet_opts} \
          ${parallel_nhl_opts:+ --parallel-nhl-opts "$parallel_nhl_opts"} ${parallel_nhn_opts:+ --parallel-nhn-opts "$parallel_nhn_opts"} \
          ${ali_dim_csl} ${dnn_init} ${nnet_init}
	  elif [ $renew_nnet_type == "blocksoftmax" ]; then
	    # create a softmax layer for each task
	    local/nnet/renew_nnet_blocksoftmax.sh --remove-last-components $remove_last_components ${renew_nnet_opts} ${ali_dim_csl} ${dnn_init} ${nnet_init}
	  else
	    # create a single softmax layer across all tasks
	    local/nnet/renew_nnet_softmax.sh --softmax-dim ${output_dim} --remove-last-components $remove_last_components ${renew_nnet_opts} ${ali_dir[0]}/final.mdl ${dnn_init} ${nnet_init}
	  fi
  
      $cuda_cmd $dir/log/train_nnet.log \
      local/nnet/train_pt.sh  ${nnet_init:+ --nnet-init "$nnet_init" --hid-layers 0} \
		--learn-rate 0.008 \
        ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-opts "--delta-order=$delta_order" --splice $splice --splice-step $splice_step \
        --labels-trainf "scp:$dir/ali-post/post_combined.scp" \
        --labels-crossvf "scp:$dir/ali-post/post_combined.scp" \
        --frame-weights  "scp:$dir/ali-post/frame_weights_combined.scp" \
        --num-tgt $output_dim \
        --copy-feats "false" \
        --randomizer-size ${randomizer_size} --minibatch-size ${minibatch_size} \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ${ali_dir[0]} ${ali_dir[0]} $dir			# ${ali_dir[0]} is used only to copy the HMM transition model
        
        # Experimental Code for stacked softmax; Uncomment to train stacked softmax
        if false; then
        local/nnet/make_activesoftmax_from_blocksoftmax.sh $dir/final.nnet "$(echo ${ali_dim_csl}|tr ':' ',')" $active_block $decode_dir/final.nnet
        local/nnet/train_stackedsoftmax.sh  ${nnet_init:+ --nnet-init "$dir/final.nnet" --hid-layers 0} \
		--learn-rate 0.008 \
        ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-opts "--delta-order=$delta_order" --splice $splice --splice-step $splice_step \
        --labels-trainf "scp:$dir/ali-post/post_block_2.scp" \
        --labels-crossvf "scp:$dir/ali-post/post_block_2.scp" \
        --frame-weights  "scp:$dir/ali-post/frame_weights_block_2.scp" \
        --num-tgt $output_dim \
        --copy-feats "false" \
        --randomizer-size ${randomizer_size} --minibatch-size ${minibatch_size} \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ${ali_dir[0]} ${ali_dir[0]} $dir
        fi
    else
      $cuda_cmd $dir/log/train_nnet.log \
      local/nnet/train_pt.sh  --hid-layers $hid_layers --hid-dim $hid_dim  \
		--learn-rate 0.008 \
        ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-opts "--delta-order=$delta_order" --splice $splice --splice-step $splice_step \
        --labels-trainf "scp:$dir/ali-post/post_combined.scp" \
        --labels-crossvf "scp:$dir/ali-post/post_combined.scp" \
        --frame-weights  "scp:$dir/ali-post/frame_weights_combined.scp" \
        --num-tgt $output_dim \
        --copy-feats "false" \
        --randomizer-size ${randomizer_size} --minibatch-size ${minibatch_size} \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ${ali_dir[0]} ${ali_dir[0]} $dir			# ${ali_dir[0]} is used only to copy the HMM transition model        
    fi    
    ;;
    dnn)
    # 6 hidden layers, 2048 simgoid neurons,
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --hid-layers 6 --hid-dim 2048 \
        ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-opts "--delta-order=$delta_order" --splice $splice --splice-step $splice_step \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
        --randomizer-size ${randomizer_size} --minibatch-size ${minibatch_size} \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir
    ;;
    *)
    echo "Unknown --nnet-type $nnet_type"; exit 1;
    ;;
  esac
fi

L=${lang_code[0]}     # the first lang is the test language
exp_dir=${ali_dir[0]} # this should be a gmm dir
# Decoding stage
if [ $stage -le 3 ]; then
  case $nnet_type in
    dnn_small)
      echo "Decoding $L"
      for active_block in ${test_block[@]}; do  #$(seq 1 $num_langs)
        
        graph_dir=$exp_dir/graph_text_G_$L
	    [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_text_G $exp_dir $graph_dir || exit 1; }
	    
        for type in "eval"; do # "dev" "eval"
          decode_dir=$dir/decode_block_${active_block}_${type}_text_G_$L
          if [[ $renew_nnet_type == "parallel" ]]; then
            # extract the active network from the parallel network
            local/nnet/make_activesoftmax_from_parallel.sh --remove-last-components $remove_last_components $dir/final.nnet $active_block $decode_dir/final.nnet
          elif [[ $renew_nnet_type == "blocksoftmax" ||  -z $renew_nnet_type ]]; then
            # extract the active softmax from block softmax
		    local/nnet/make_activesoftmax_from_blocksoftmax.sh $dir/final.nnet "$(echo ${ali_dim_csl}|tr ':' ',')" $active_block $decode_dir/final.nnet
		  else
		    echo "Decoding with $renew_nnet_type not supported" && exit 1
		  fi		  
		  # make other necessary dependencies available
          (cd $decode_dir; ln -s ../{final.mdl,final.feature_transform,norm_vars,cmvn_opts,delta_opts} . ;)
          # create "prior_counts"          
          steps/nnet/make_priors.sh --use-gpu ${use_gpu} $data_tr90 $decode_dir
          # finally, decode
          (steps/nnet/decode.sh --nj 4 ${parallel_opts} --cmd "$decode_cmd" --use-gpu ${use_gpu} --config conf/decode_dnn.config --acwt 0.2 --srcdir $decode_dir \
	        $graph_dir $(dirname ${data_dir[0]})/$type $decode_dir || exit 1;) &
	    done
	    	    
	  done
    ;;
    *)
      echo "Decoding not supported for --nnet-type $nnet_type"; exit 1;
    ;;
  esac
fi

exit 0
