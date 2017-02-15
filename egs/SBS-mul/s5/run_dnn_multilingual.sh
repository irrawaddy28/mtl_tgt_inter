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
splice=5         # temporal splicing
splice_step=1    # stepsize of the splicing (1 == no gap between frames)
test_block_csl=1

# Multi-task training options
lang_weight_csl="1.0:1.0:1.0"
lat_dir_csl="-:-:-"    # lattices obtained after decoding PT or unsup data
data_type_csl="dt:dt:dt"
dup_and_merge_csl="0>>1:0>>2:0>>3" # x>>y means: create x copies of data and use those to train the softmax layer in block y
renew_nnet_type="parallel"  # can be "parallel", "blocksoftmax", or an empty string (single softmax for all tasks)
renew_nnet_opts=            # options for the renew nnnet. Details about these options in local/nnet/renew_nnet_* scripts. For e.g., In case of parallel net, option could be "--parallel-nhl-opts 3:3 --parallel-nhn-opts 1024:1024 "; 

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
#$0 --dnn-init "exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax/decode_dev_text_G_SW/final.nnet" --lang-weight-csl "1.0:1.0:1.0" --threshold-csl "0.7:1.0:0.8" --lat-dir-csl "exp/tri3cpt_ali/SW/decode_train:-:exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax/decode_unsup_4k_SW"
# --data-type-csl "pt:dt:unsup" --dup-and-merge-csl "4>>1:0>>0:1>>1"
# "SW:AR_CA_HG_MD_UR:SW" "exp/tri3cpt_ali/SW:exp/tri3b_ali/SW:exp/tri3cpt_ali/SW"
# "data-fmllr-tri3c_map/SW/SW/train:data-fmllr-tri3b/SW/AR_CA_HG_MD_UR/train:data-fmllr-tri3c_map/SW/SW/unsup_4k_SW" 
# "data-fmllr-tri3c_map/SW/combined" "exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax2b"

# ./run_multilingual.sh --dnn-init "exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax/decode_dev_text_G_SW/final.nnet" --lang-weight-csl "1.0:1.0:0.0" --threshold-csl "0.7:1.0:0.8" --lat-dir-csl "exp/tri3cpt_ali/SW/decode_train:-:exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax/decode_unsup_4k_SW" --data-type-csl "pt:dt:unsup" --dup-and-merge-csl "4>>1:0>>0:1>>1" "SW:AR_CA_HG_MD_UR:SW" "exp/tri3cpt_ali/SW:exp/tri3b_ali/SW:exp/tri3cpt_ali/SW" "data-fmllr-tri3c/SW/SW/train:data-fmllr-tri3b/SW/AR_CA_HG_MD_UR/train:data-fmllr-tri3c/SW/SW/unsup_4k_SW" "data-fmllr-tri3c_map/SW/combined" "exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax2c"
if [ $# != 5 ]; then
   echo "Usage: $0 [options] <lang code csl> <ali dir csl> <data dir csl> <data output dir> <nnet output dir>" 
   echo "e.g.: $0 --dnn-init nnet.init --lang-weight-csl 1.0:1.0 SW:HG exp/tri3b_ali/SW:exp/tri3b_ali/HG data-fmllr-tri3b/SW/train:data-fmllr-tri3b/HG/train data-fmllr-tri3b/combined exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax "
   echo ""
fi

lang_code_csl=$1 # AR_CA_HG_MD_UR:SW:SW
ali_dir_csl=$2   # exp/tri3b_ali/SW:exp/tri3cpt_ali/SW:
data_dir_csl=$3  # data-fmllr-tri3c/SW/SW/train:data-fmllr-tri3c/SW/AR_CA_HG_MD_UR/train
data=$4          # this is where we'll save the combined data for multisoftmax training 
nnet_dir=$5      # o/p dir of multisoftmax nnet

# Convert 'csl' to bash array (accept separators ',' ':'),
lang_code=($(echo $lang_code_csl | tr ',:' ' ')) 
ali_dir=($(echo $ali_dir_csl | tr ',:' ' '))
data_dir=($(echo $data_dir_csl | tr ',:' ' '))

# Make sure we have same number of items in lists,
! [ ${#lang_code[@]} -eq ${#ali_dir[@]} -a ${#lang_code[@]} -eq ${#data_dir[@]} ] && \
  echo "Non-matching number of 'csl' items: lang_code ${#lang_code[@]}, ali_dir ${#ali_dir[@]}, data_dir ${#data_dir[@]}" && \
  exit 1
num_langs=${#lang_code[@]}

# Convert csls to bash arrays
if [ -z "$lang_weight_csl" ]; then 
  for ((i=0; i<$num_langs; i++)); do 
    lang_weight_csl="$lang_weight_csl:1.0"
  done
  lang_weight_csl=${lang_weight_csl#*:}  
fi

if [ -z "$data_type_csl" ]; then
  for ((i=0; i<$num_langs; i++)); do 
    data_type_csl="$data_type_csl:dt"    
  done
  data_type_csl=${data_type_csl#*:}
fi
data_type=($(echo $data_type_csl | tr ',:' ' '))
for ((i=0; i<$num_langs; i++)); do
  type=${data_type[$i]}
  if ! [ "$type" == "pt" -o "$type" == "unsup" -o "$type" == "dt" ]; then
    echo "data type at position $((i + 1)) is \"$type\" is not supported"
    exit 1
  fi
done

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

make_num_copies="false"
if [ -z "$dup_and_merge_csl" ]; then 
  for ((i=0; i<$num_langs; i++)); do
    dup_and_merge_csl="$dup_and_merge_csl:0>>0"
  done
  dup_and_merge_csl=${dup_and_merge_csl#*:}
fi
dup_and_merge=($(echo $dup_and_merge_csl | tr ',:' ' '))

if [ "$num_langs" -ne "${#data_type[@]}" -o "$num_langs" -ne "${#dup_and_merge[@]}" ]; then
  echo "Non-matching number of 'csl' items: data_type ${#data_type[@]}, dup_and_merge ${#dup_and_merge[@]}"
  exit 1;
fi

test_block=($(echo $test_block_csl | tr ',:' ' '))
for ((i=0; i<${#test_block[@]}; i++)); do
    [ "${test_block[$i]}" -lt "0" ] && echo "test_block[$i]=${test_block[$i]} cannot be less than 0" && exit 1;
    [ "${test_block[$i]}" -gt "$num_langs" ] && echo "test_block[$i]=${test_block[$i]} cannot be greater than $num_langs" && exit 1;
done


# Check if all the input directories exist,
lat_dir=($(echo "$lat_dir_csl" | tr ',:' ' '))
for i in $(seq 0 $[num_langs-1]); do
  echo "lang = ${lang_code[$i]}, type = ${data_type[$i]}, alidir = ${ali_dir[$i]}, datadir = ${data_dir[$i]}, latdir = ${lat_dir[$i]}, dup_and_merge = ${dup_and_merge[$i]}"
  
  [ ! -d ${ali_dir[$i]} ] && echo  "Missing ${ali_dir[$i]}" && exit 1
  [ ! -d ${data_dir[$i]} ] && echo "Missing ${data_dir[$i]}" && exit 1
  
  # if data_type is pt or unsup, then lat_dir must exist. 
  # if data_type is dt, then lat_dir should be set to - (meaning empty)
  if [ "${data_type[$i]}" ==  "pt" -o "${data_type[$i]}" ==  "unsup" ]; then
    [ ! -d ${lat_dir[$i]} ] && echo  "Missing ${lat_dir[$i]}" && exit 1
  elif   [ "${data_type[$i]}" ==  "dt" ]; then
    [ ${lat_dir[$i]} != "-" ] && echo  "latdir should be set to a hyphen to indicate empty directory" && exit 1
  fi
done


# Make the features,
data_tr90=$data/combined_tr90
data_cv10=$data/combined_cv10
if [ $stage -le 0 ]; then
  # Make local copy of data-dirs (while adding language-code),
  tr90=""
  cv10=""  
  rm -rf $data
  for i in $(seq 0 $[num_langs-1]); do
    code=${lang_code[$i]}
    dir=${data_dir[$i]}
    tgt_dir=$data/${code}_$(basename $dir)
    type=${data_type[$i]}
    
    utils/copy_data_dir.sh $dir $tgt_dir 
    
    ## extract features, get cmvn stats,
    #steps/make_fbank_pitch.sh --nj 30 --cmd "$train_cmd -tc 10" $tgt_dir{,/log,/data}
    #steps/compute_cmvn_stats.sh $tgt_dir{,/log,/data}
    
    # Split lists 90% train / 10% held-out only for supervised datasets like pt, dt.
    # For unsupervised dataset, we use 100% train / 0% held-out.
    if [ "${data_type[$i]}" !=  "unsup" ]; then
      utils/subset_data_dir_tr_cv.sh $tgt_dir ${tgt_dir}_tr90 ${tgt_dir}_cv10
      tr90="$tr90 ${tgt_dir}_tr90"
      cv10="$cv10 ${tgt_dir}_cv10"
    else
      utils/copy_data_dir.sh $tgt_dir ${tgt_dir}_tr90
      tr90="$tr90 ${tgt_dir}_tr90"
    fi
  done
  
  ## Merge the datasets,  
  set +e;
  unsup_present=$(echo ${data_type[@]}|grep -w "unsup")
  set -e;
  echo -e "\ntr90 list = \"$tr90\", unsup_present = \" ${unsup_present}\""
  if [ ! -z "${unsup_present}" ]; then
    ## If we don't specify skip-fix true, the combined scp will exclude utts which do not have text
    utils/combine_data.sh --skip-fix "true" $data_tr90 $tr90
    utils/combine_data.sh --skip-fix "true" $data_cv10 $cv10
  else    
    utils/combine_data.sh  $data_tr90 $tr90
    utils/combine_data.sh  $data_cv10 $cv10    
    ## Validate,
    utils/validate_data_dir.sh $data_tr90
    utils/validate_data_dir.sh $data_cv10
  fi
  echo "combined \"$tr90\" and saved in $data_tr90"
  echo "combined \"$cv10\" and saved in $data_cv10"
fi

# Extract the tied-state numbers from transition models,
for i in $(seq 0 $[num_langs-1]); do
  ali_dim[i]=$(hmm-info ${ali_dir[i]}/final.mdl | grep pdfs | awk '{ print $NF }')
done
ali_dim_csl=$(echo ${ali_dim[@]} | tr ' ' ':')

# Total number of DNN outputs (sum of all per-language blocks),
output_dim=$(echo ${ali_dim[@]} | tr ' ' '\n' | awk '{ sum += $i; } END{ print sum; }')
echo "Total number of DNN outputs: $output_dim = $(echo ${ali_dim[@]} | sed 's: : + :g')"

# Objective function string (per-language weights are imported from '$lang_weight_csl'),
objective_function="multitask$(echo ${ali_dim[@]} | tr ' ' '\n' | \
  awk -v w=$lang_weight_csl 'BEGIN{ split(w,w_arr,/[,:]/); } { printf(",xent,%d,%s", $1, w_arr[NR]); }')"
echo "Multitask objective function: $objective_function"

# DNN training will be in $dir, the alignments are prepared beforehand,
#dir=exp/dnn4g-multilingual${num_langs}-$(echo $lang_code_csl | tr ',' '-')-${nnet_type} 
dir=$nnet_dir
[ ! -e $dir ] && mkdir -p $dir
echo -e "$0 $@\n\n\n 
Experiment Settings:\n 
lang_code_csl = $lang_code_csl\n   
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
dnn out = $dir" > $dir/config 

# Prepare the merged targets,
if [ $stage -le 1 ]; then
  [ ! -e $dir/ali-post ] && mkdir -p $dir/ali-post
  tr90="" # save all training dirs, including num_copies, in this list
  # re-saving the ali in posterior format, indexed by 'scp',
  for i in $(seq 0 $[num_langs-1]); do
    code=${lang_code[$i]}
    tgt_dir=$data/${code}_$(basename ${data_dir[$i]})
    type=${data_type[$i]}
    threshold=${threshold_list[$i]}
    ali=${ali_dir[$i]}
    lat=${lat_dir[$i]}
    dam=($(echo ${dup_and_merge[$i]}|tr '>>' ' '))
    dup=${dam[0]}
    blockid="block_$((i+1))"
    
    echo "==================================="
    echo "Generating posteriors and frame weights for: lang = $code, type = $type, ali = $ali, lat = $lat, num copies = $dup"
    echo "==================================="
    
    postdir=$dir/ali-post/$code/$type/post_train_thresh${threshold:+_$threshold}
    [ -d $postdir ] || mkdir -p $postdir
    best_path_dir=$dir/ali-post/$code/$type/bestpath_ali
    if [ "$type" == "pt" -o "$type" == "unsup" ]; then
      # pt or unsupervised data
      decode_dir=$lat
	  
      local/posts_and_best_path_weights.sh --acwt $acwt --threshold $threshold \
	    --use-soft-counts $use_soft_counts --disable-upper-cap $disable_upper_cap \
        $ali $decode_dir $best_path_dir $postdir
    else
      # dt data      
      ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | \
        ali-to-post ark:- ark,scp:$postdir/post.ark,$postdir/post.scp
      
      ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | \
       awk '{printf $1" ["; for (i=2; i<=NF; i++) { printf " "1; }; print " ]";}' | \
         copy-vector ark,t:- ark,t,scp:$postdir/frame_weights.ark,$postdir/frame_weights.scp || exit 1;      
    fi
    
    (cd $dir/ali-post; rm -f post_${blockid}.scp frame_weights_${blockid}.scp;
        cp $code/$type/post_train_thresh${threshold:+_$threshold}/post.scp post_${blockid}.scp;
        cp $code/$type/post_train_thresh${threshold:+_$threshold}/frame_weights.scp frame_weights_${blockid}.scp)
        
    # If dup copies were requested by the user, make num_copies of posteriors, frame weights and data 
    num_copies=$dup
    if [ $num_copies -gt 0 ]; then
      make_num_copies="true"
      
      awk -v num_copies=$num_copies \
      '{for (i=0; i<num_copies; i++) { print i"-"$1" "$2 } }' \
      $postdir/post.scp > $postdir/post_${num_copies}x.scp
  
      awk -v num_copies=$num_copies \
      '{for (i=0; i<num_copies; i++) { print i"-"$1" "$2 } }' \
      $postdir/frame_weights.scp > $postdir/frame_weights_${num_copies}x.scp

      copied_data_dirs=
      for i in `seq 0 $[num_copies-1]`; do
        utils/copy_data_dir.sh --utt-prefix ${i}- --spk-prefix ${i}- ${tgt_dir}_tr90 \
        ${tgt_dir}_tr90_$i || exit 1
        copied_data_dirs="$copied_data_dirs ${tgt_dir}_tr90_$i"
      done
      utils/combine_data.sh ${tgt_dir}_tr90_${num_copies}x $copied_data_dirs || exit 1
      tr90="$tr90 ${tgt_dir}_tr90 ${tgt_dir}_tr90_${num_copies}x" # add the base dir + num_copies dir to the list
    else
      tr90="$tr90 ${tgt_dir}_tr90"
    fi
  done
  
  # If dup copies were requested by the user, merge num_copies of the data to the combined data dir  
  echo -e "\n\nall dirs list = $tr90"
  if $make_num_copies ; then
    ## Merge the datasets
    set +e;
    unsup_present=$(echo ${data_type[@]}|grep -w "unsup")
    set -e;
    if [ ! -z "${unsup_present}" ]; then
      ## If we don't specify skip-fix true, the combined scp will exclude utts which do not have text
      utils/combine_data.sh --skip-fix "true" $data_tr90 $tr90
    else
      utils/combine_data.sh  $data_tr90 $tr90
      ## Validate,
      utils/validate_data_dir.sh $data_tr90
    fi
  fi
  
  # If dup copies were requested by the user, merge num_copies of the posteriors and frame weights  
  if $make_num_copies ; then
    for i in $(seq 0 $[num_langs-1]); do      
      dam=($(echo ${dup_and_merge[$i]}|tr '>>' ' '))
      dup=${dam[0]}
      dstnblk=${dam[1]}
      threshold=${threshold_list[$i]}
      num_copies=$dup
      if [ $num_copies -gt 0 -a $dstnblk -gt 0 ]; then
        dstnblk=$((dstnblk - 1)); # this is the block where we want to send the targets to
        srccode=${lang_code[$i]}
        srctype=${data_type[$i]}            
        dstncode=${lang_code[$dstnblk]}
        dstntype=${data_type[$dstnblk]}
        blockid="block_$((dstnblk+1))"
        threshold=${threshold_list[$i]}
        
        echo -e "\nGenerating post.scp for $blockid: src lang = $srccode, src type = $srctype, dstn lang = $dstncode, dstn type = $dstntype, num_copies = $dup"                
        srcpostscp=$dir/ali-post/$srccode/$srctype/post_train_thresh${threshold:+_$threshold}/post_${num_copies}x.scp
        srcfwscp=$dir/ali-post/$srccode/$srctype/post_train_thresh${threshold:+_$threshold}/frame_weights_${num_copies}x.scp
        dstnpostscp=$dir/ali-post/post_${blockid}.scp
        dstnfwscp=$dir/ali-post/frame_weights_${blockid}.scp        
        echo "srcpostscp = $srcpostscp, dstnpostscp = $dstnpostscp"
        echo "srcfwscp = $srcfwscp, dstnfwscp = $dstnfwscp"
        sort -k1,1 $srcpostscp $dstnpostscp -o $dstnpostscp
        sort -k1,1 $srcfwscp $dstnfwscp   -o $dstnfwscp
      fi
    done
  fi
      
  # pasting the ali's, adding language-specific offsets to the posteriors,
  featlen="ark:feat-to-len 'scp:cat $data_tr90/feats.scp $data_cv10/feats.scp |' ark,t:- |" # get number of frames for every utterance,
  #post_scp_list=$(paste  <(echo ${lang_code[@]} | tr ' ' '\n') <(echo  ${data_type[@]} | tr ' ' '\n') | awk -v d=$dir '{ printf(" scp:%s/ali-post/post_%s_%s.scp", d, $1, $2); }')
  post_scp_list=$( echo $(seq 1 $num_langs)| tr ' ' '\n' | awk -v d=$dir '{ printf(" scp:%s/ali-post/post_block_%s.scp", d, $1); }')
  echo -e "\npost_scp_list = $post_scp_list"
  paste-post --allow-partial=true "$featlen" "${ali_dim_csl}" ${post_scp_list} \
    ark,t,scp:$dir/ali-post/post_combined.ark,$dir/ali-post/post_combined.scp
	
  # pasting the frame weights
  #frame_weights_scp_list=$(paste  <(echo ${lang_code[@]} | tr ' ' '\n') <(echo  ${data_type[@]} | tr ' ' '\n') | awk -v d=$dir '{ printf(" %s/ali-post/frame_weights_%s_%s.scp", d, $1, $2); }')
  frame_weights_scp_list=$( echo $(seq 1 $num_langs)| tr ' ' '\n' | awk -v d=$dir '{ printf(" %s/ali-post/frame_weights_block_%s.scp", d, $1); }')
  echo -e "\nframe_weights_scp_list = $frame_weights_scp_list"
  copy-vector "scp:cat ${frame_weights_scp_list}|" ark,t,scp:$dir/ali-post/frame_weights_combined.ark,$dir/ali-post/frame_weights_combined.scp || exit 1
fi

# Train the <BlockSoftmax> system, 1st stage of Stacked-Bottleneck-Network,
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
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ali-dummy ali-dummy $dir
    ;;
    dnn_small)
    # 4 hidden layers, 1024 sigmoid neurons
    if [[ ! -z ${dnn_init} ]]; then
	  nnet_init=$dir/nnet.init
	  
      if [ $renew_nnet_type == "parallel" ]; then
        # create a generic network for each task
	    local/nnet/renew_nnet_parallel.sh --remove-last-components $remove_last_components ${renew_nnet_opts} ${ali_dim_csl} ${dnn_init} ${nnet_init}
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
        --cmvn-opts "--norm-means=true --norm-vars=true" \
        --delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
        --labels-trainf "scp:$dir/ali-post/post_combined.scp" \
        --labels-crossvf "scp:$dir/ali-post/post_combined.scp" \
        --frame-weights  "scp:$dir/ali-post/frame_weights_combined.scp" \
        --num-tgt $output_dim \
        --copy-feats "false" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ${ali_dir[0]} ${ali_dir[0]} $dir			# ${ali_dir[0]} is used only to copy the HMM transition model
        
        # Experimental Code for stacked softmax; Uncomment to train stacked softmax
        if false; then
        local/nnet/make_activesoftmax_from_blocksoftmax.sh $dir/final.nnet "$(echo ${ali_dim_csl}|tr ':' ',')" $active_block $decode_dir/final.nnet
        local/nnet/train_stackedsoftmax.sh  ${nnet_init:+ --nnet-init "$dir/final.nnet" --hid-layers 0} \
		--learn-rate 0.008 \
        --cmvn-opts "--norm-means=true --norm-vars=true" \
        --delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
        --labels-trainf "scp:$dir/ali-post/post_block_2.scp" \
        --labels-crossvf "scp:$dir/ali-post/post_block_2.scp" \
        --frame-weights  "scp:$dir/ali-post/frame_weights_block_2.scp" \
        --num-tgt $output_dim \
        --copy-feats "false" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ${ali_dir[0]} ${ali_dir[0]} $dir
        fi
    else
      $cuda_cmd $dir/log/train_nnet.log \
      local/nnet/train_pt.sh  --hid-layers $hid_layers --hid-dim $hid_dim  \
		--learn-rate 0.008 \
        --cmvn-opts "--norm-means=true --norm-vars=true" \
        --delta-opts "--delta-order=2" --splice $splice --splice-step $splice_step \
        --labels-trainf "scp:$dir/ali-post/post_combined.scp" \
        --labels-crossvf "scp:$dir/ali-post/post_combined.scp" \
        --frame-weights  "scp:$dir/ali-post/frame_weights_combined.scp" \
        --num-tgt $output_dim \
        --copy-feats "false" \
        --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
        ${data_tr90} ${data_cv10} lang-dummy ${ali_dir[0]} ${ali_dir[0]} $dir			# ${ali_dir[0]} is used only to copy the HMM transition model        
    fi    
    ;;
    dnn)
    # 6 hidden layers, 2048 simgoid neurons,
    $cuda_cmd $dir/log/train_nnet.log \
      steps/nnet/train.sh --learn-rate 0.008 \
        --hid-layers 6 --hid-dim 2048 \
        --cmvn-opts "--norm-means=true --norm-vars=false" \
        --delta-opts "--delta-order=2" --splice 5 \
        --labels "scp:$dir/ali-post/combined.scp" --num-tgt $output_dim \
        --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
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
        for type in "dev" "eval"; do
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
          steps/nnet/make_priors.sh --use-gpu "yes" $data_tr90 $decode_dir
          # finally, decode
          (steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 --srcdir $decode_dir \
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
