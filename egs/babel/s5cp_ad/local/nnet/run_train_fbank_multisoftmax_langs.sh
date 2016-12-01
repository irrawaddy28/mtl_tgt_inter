#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains DNN with <MultiSoftmax> output on top of filter bank features.
# The network is trained on multiple languages simultaneously creating a separate softmax layer per language
# while sharing hidden layers across all languages.

# Usage: $0 --l=<lang code 1> --ali=<ali dir lang 1> --data=<data dir lang 1> ... \
#			--l=<lang code N> --ali=<ali dir lang N> --data=<data dir lang N>
# 
# Example for training a multilingual Babel nnet below:
# E.g. $0 --l=ASM --ali=exp/ASM/tri5_ali --data=data/ASM/train \
#		  --l=BNG --ali=exp/BNG/tri5_ali --data=data/BNG/train \
#		  --l=CNT --ali=exp/CNT/tri5_ali --data=data/CNT/train \
#		  --l=HAI --ali=exp/HAI/tri5_ali --data=data/HAI/train \
#		  --l=LAO --ali=exp/LAO/tri5_ali --data=data/LAO/train \
#		  --l=PSH --ali=exp/PSH/tri5_ali --data=data/PSH/train \
#		  --l=TAM --ali=exp/TAM/tri5_ali --data=data/TAM/train \
#		  --l=TGL --ali=exp/TGL/tri5_ali --data=data/TGL/train \
#		  --l=TUR --ali=exp/TUR/tri5_ali --data=data/TUR/train \
#		  --l=ZUL --ali=exp/ZUL/tri5_ali --data=data/ZUL/train
# 
# Note: Must specify at least 2 languages as i/p to script. Otherwise, it will fail.
#
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0
#. utils/parse_options.sh || exit 1;

set -u 
set -e
set -o pipefail

n=0
j=0
for i in "$@"
do
case $i in
    --l=*)
    lang[j]="${i#*=}"
    shift; n=$((++n)); 
    ;;    
    --ali=*)
    ali[j]="${i#*=}"
    shift; n=$((++n)); 
    ;;        
    --data=*)
    data[j]="${i#*=}"
    shift; n=$((++n)); 
    ;;    
    *)
    echo "Unknown argument: ${i#*=}, exiting"; exit 1 
    ;;    
esac
[[ $(( n%3 )) -eq 0 ]] && j=$((j + 1))
done

nlangs=$(( n/3 - 1))

# Check if all the user i/p directories exist
for i in  $(seq 0 $nlangs)
do
	echo "lang = ${lang[i]}, alidir = ${ali[i]}, datadir = ${data[i]}"
	[ ! -e ${ali[i]} ] && echo  "Missing  ${ali[i]}" && exit 1
	[ ! -e ${data[i]} ] && echo "Missing ${data[i]}" && exit 1
done

# Make the features
thisdata=data-fbank-multisoftmax/train
train_tr90_multilingual=$thisdata/train_tr90_multilingual
train_cv10_multilingual=$thisdata/train_cv10_multilingual
if [ $stage -le 0 ]; then
  rm -rf $thisdata 2>/dev/null;
  tr_90=""; cv_10="";
  for i in  $(seq 0 $nlangs)
  do
    # Store fbank features in language dep directories, so we can train on them easily,
	dir=$thisdata/${lang[i]}; mkdir -p $dir; 
	gmm_ali=${ali[i]}
	data_train=${data[i]}	
	echo "Language = ${lang[i]}: Generating features from datadir = ${data[i]} and saving in $dir"	
	
	utils/copy_data_dir.sh ${data_train} $dir || exit 1	
	steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
      $dir $dir/log $dir/data || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    
	#steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
    # --transform-dir ${gmm_ali} \
    # $dir ${data_train} ${gmm_ali} $dir/log $dir/data || exit 1    
    # steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;    
    
    
    # split the language dependent data : 90% train 10% cross-validation (held-out) 
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $dir ${dir}_tr90 ${dir}_cv10 || exit 1
    tr_90="${tr_90}   ${dir}_tr90 ";  cv_10="${cv_10}   ${dir}_cv10 ";
  done
  
  # Merge all language dependent 90%-training sets to one multilingual training set
  echo "Merging ${tr_90} to ${train_tr90_multilingual}"
  utils/combine_data.sh ${train_tr90_multilingual} ${tr_90} || exit 1
  utils/validate_data_dir.sh ${train_tr90_multilingual}  
  
  # Merge all language dependent 10%-cv sets to one multilingual cros-validation set
  echo "Merging ${cv_10} to ${train_cv10_multilingual}"
  utils/combine_data.sh ${train_cv10_multilingual} ${cv_10} || exit 1
  utils/validate_data_dir.sh ${train_cv10_multilingual}  
fi

# Make a colon separated list of the number of output nodes of each softmax layer
output_dim=
ali_dim_csl=
objective_function=multitask
for i in $(seq 0 $nlangs)
do
	ali_dim[i]=$(hmm-info ${ali[i]}/final.mdl | grep pdfs | awk '{ print $NF }')
	echo "Output dim of block softmax for ${lang[i]} = ${ali_dim[i]}"
	output_dim=$(( output_dim + ${ali_dim[i]} ))
	[[ -z ${ali_dim_csl} ]] && ali_dim_csl="${ali_dim[i]}" || ali_dim_csl="${ali_dim_csl}:${ali_dim[i]}" 
	objective_function="${objective_function},xent,${ali_dim[i]},1"
done
echo "Sum of all block output dims = $output_dim (${ali_dim_csl})"
echo "Multitask objective function: $objective_function"

# Prepare the merged targets
thisdir=exp/dnn4e-fbank_multisoftmax # this is our current expt dir
if [ $stage -le 1 ]; then  
  rm -rf $thisdir 2>/dev/null
  mkdir -p $thisdir/log
  post_scp_list=
  for i in $(seq 0 $nlangs)
  do
	dir=$thisdir/${lang[i]}; mkdir -p $dir
	gmm_ali=${ali[i]}		
	
  	copy-int-vector "ark:gunzip -c ${gmm_ali}/ali.*.gz |" ark,t:- | gzip -c >$dir/ali_${lang[i]}.gz
  	
  	# Store posteriors to disk, indexed by 'scp',	
  	ali-to-pdf ${gmm_ali}/final.mdl "ark:gunzip -c $dir/ali_${lang[i]}.gz |" ark,t:- | \
		ali-to-post ark,t:- ark,scp:$dir/post_${lang[i]}.ark,$dir/post_${lang[i]}.scp	
	post_scp_list="${post_scp_list}  scp:$dir/post_${lang[i]}.scp"
  done
  echo "post_scp_list = ${post_scp_list}"
  feats_scp_list=
  for i in $(seq 0 $nlangs)
  do
	feats_scp_list="${feats_scp_list}  ${data[i]}/feats.scp" 
  done
  featlen="ark:feat-to-len 'scp:cat ${feats_scp_list} |' ark,t:- |"   # print number of frames for every utterance in feats.scp     
  
  paste-post --allow-partial=true "$featlen" ${ali_dim_csl} ${post_scp_list} \
	ark,scp:$thisdir/pasted_post.ark,$thisdir/pasted_post.scp 2>$thisdir/log/paste_post.log  
fi

dir=$thisdir
# Train <MultiSoftmax> system,
if [ $stage -le 2 ]; then 
  [[ -d $dir ]] || { echo "$dir does not exist"; exit 1; }
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
	  --nnet-binary "false" \
	  --hid-layers 6 \
      --cmvn-opts "--norm-means=true --norm-vars=false" \
      --delta-opts "--delta-order=2" --splice 5 \
      --labels-trainf "scp:$dir/pasted_post.scp" --num-tgt $output_dim \
      --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
      --train-opts "--l2-penalty 0.0" \
      --train-tool "nnet-train-frmshuff --objective-function=$objective_function" \
      --learn-rate 0.008 \
      ${train_tr90_multilingual} ${train_cv10_multilingual} lang-dummy ali-dummy ali-dummy $dir || exit 1; 
fi

exit 0

