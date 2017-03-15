#!/bin/bash

# Copyright 2015-2018 University of Illinois (author: Amit Das),
# Apache 2.0
#
# This script takes a colon separated list (csl) of various task specific attributes (features, alignments etc) and creates
# task specific scps in preparation for for a multitask learning (MTL) based DNN

nj=10
acwt=0.2
use_soft_counts=true    # Use soft-posteriors as targets for PT data
disable_upper_cap=true

cmvn_opts=      # speaker specific cmvn for i/p features. For mean+var normalization, use "--norm-means=true --norm-vars=true"
delta_order=0   # Use 1 for delta, 2 for delta-delta
splice=5
splice_step=1
use_gpu=yes
# End of config.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;
[ -f cmd.sh ] && . ./cmd.sh;

. utils/parse_options.sh || exit 1;

usage="Usage: $0 <language csl> <data type csl> <feat dir csl> <ali csl> <lattice csl> <threshold csl> <copies and merge csl> <combined feat dir> <combined target posterior dir>"
# e.g.
#  local/merge_targets.sh --acwt 0.2 --use-soft-counts true --disable-upper-cap true --cmvn-opts --norm-means=true --norm-vars=true --delta-order 2 --splice 5 --splice-step 1 \
#  SW:MD:SW \
#  pt:dt:unsup \
#  data-fmllr-tri3c/SW/train:data-fmllr-tri3c/MD/train:data-fmllr-tri3c/SW/unsup \
#  exp/tri3cpt_ali/SW:exp/tri3c_ali/MD:exp/tri3cpt_ali/SW \
#  exp/tri3cpt_ali/SW/decode_train:-:- \
#  0.6:0.0:0.0 \
#  2>>1:1>>2:1>>3 \
#  data-fmllr-tri3c/combined_fw \
#  exp/dnn4_pretrain-dbn_dnn/mtl/ali-post

[[ $# -eq 10 ]] || { echo $usage; exit 1; }

lang_csl=$1    #
dtype_csl=$2   # data type: pt, dt, unsup
ltype_csl=$3   # label type: s (senone), p (monophone), f (feature)
ddir_csl=$4    #
ali_csl=$5     #  
lat_csl=$6     #
thresh_csl=$7  # 
dup_and_merge_csl=$8  #
feat_dir=$9    # dir to save features
dir=${10}      # dir to save target posteriors and frame weights


lang_list=($(echo $lang_csl | tr ':' ' '))
dtype_list=($(echo $dtype_csl | tr ':' ' '))
ltype_list=($(echo $ltype_csl | tr ':' ' '))
ddir_list=($(echo $ddir_csl | tr ':' ' '))
ali_list=($(echo $ali_csl | tr ':' ' '))
lat_list=($(echo $lat_csl | tr ':' ' '))
thresh_list=($(echo $thresh_csl | tr ':' ' '))
dup_and_merge_list=($(echo $dup_and_merge_csl | tr ':' ' '))

n_tasks=${#lang_list[@]}

for i in $(seq 0 $[n_tasks-1]); do
	echo -e "\nTask $((i+1))"
	echo "============="
	echo "Language   : ${lang_list[$i]}"
	echo "Data Type  : ${dtype_list[$i]}"
    echo "Label Type : ${ltype_list[$i]}"
	echo "Feat Dir   : ${ddir_list[$i]}"
	echo "Posterior Threshold  : ${thresh_list[$i]}"
	echo "HMM Dir    : ${ali_list[$i]}"
	echo "Lat Dir    : ${lat_list[$i]}"
	echo "Dup+Merge  : ${dup_and_merge_list[$i]}"	
done

feat_dir_tr90=$feat_dir/combined_tr90
feat_dir_cv10=$feat_dir/combined_cv10
make_num_copies=false
semisup_present=$(echo ${dtype_list[@]}|grep -wc "semisup")
unsup_present=$(echo ${dtype_list[@]}|grep -wc "unsup")

[ ! -e $dir ] && mkdir -p $dir

tr90="" # save list of task specific feature directories here; element i of the list is a feature dir of task i
tr90_nutts=""

# Make copies of task specific feature directories
for i in $(seq 0 $[n_tasks-1]); do

    taskid="task_$((i+1))"
    lang=${lang_list[$i]}
    dtype=${dtype_list[$i]}
    ltype=${ltype_list[$i]}
    feat_task_dir=${feat_dir}/${lang}_${ltype}_$(basename ${ddir_list[$i]})
    dam=($(echo ${dup_and_merge_list[$i]}|tr '>>' ' '))
    num_copies=${dam[0]}

    [ $dtype == "unsup" ] && tr90_unsup_scps="${feat_task_dir}_tr90/feats.scp" # include the base scp, save for future use

    # Make copies of features
    if [ $num_copies -gt 0 ]; then
      echo "==================================="
      echo "Making $num_copies copies of features for: lang = $lang, data type = $dtype, label type = $ltype"
      echo "==================================="
      make_num_copies="true"

      copied_data_dirs=
      for c in `seq 1 $[num_copies]`; do
        utils/copy_data_dir.sh --utt-prefix ${c}- --spk-prefix ${c}- ${feat_task_dir}_tr90 \
          ${feat_task_dir}_tr90_$c || exit 1
        copied_data_dirs="$copied_data_dirs ${feat_task_dir}_tr90_$c"
      done
      
      utils/combine_data.sh ${feat_task_dir}_tr90_${num_copies}x $copied_data_dirs || exit 1
      rm -rf $copied_data_dirs
      #[ $dtype == "unsup" ] && tr90_unsup_scps="${tr90_unsup_scps}  ${feat_task_dir}_tr90_${num_copies}x/feats.scp " # include the scp for copies, save for future use

      tr90="$tr90 ${feat_task_dir}_tr90 ${feat_task_dir}_tr90_${num_copies}x" # add the base feat dir + num_copies feat dir to the combined feat dir list
      tr90_nutts="$tr90_nutts `cat ${feat_task_dir}_tr90/feats.scp|wc -l` `cat ${feat_task_dir}_tr90_${num_copies}x/feats.scp|wc -l`"
    else
      tr90="$tr90 ${feat_task_dir}_tr90" # build the combined feat dir list
      tr90_nutts="$tr90_nutts `cat ${feat_task_dir}_tr90/feats.scp|wc -l`"
    fi
    #for n in "${tr90[@]}"; do tr90_nutts="$tr90_nutts `echo $(cat $tr90[$n]/feats.scp|wc -l)`"
done

# Merge copies of task specific features to a single combined feature dir
echo -e "\n\nTask specific tr90 directories: Number of utterances";
tr90_nutts_tot=""; for n in ${tr90_nutts[@]}; do tr90_nutts_tot=$((tr90_nutts_tot + n)); done
paste -d '\t' <(echo $tr90|sed 's/ /\n/g') <(echo $tr90_nutts|sed 's/ /\n/g');
echo -e "\nCombined tr90: Number of utterances";
echo -e "$feat_dir_tr90 \t $tr90_nutts_tot\n\n"
if $make_num_copies ; then
  ## Merge the datasets
  if [[ ${semisup_present} -gt 0 || ${unsup_present} -gt 0 ]]; then
    ## If we don't specify skip-fix true, the combined scp will exclude utts which do not have text
    utils/combine_data.sh --skip-fix "true" $feat_dir_tr90 $tr90
  else
    utils/combine_data.sh  $feat_dir_tr90 $tr90
    ## Validate,
    utils/validate_data_dir.sh $feat_dir_tr90
  fi
fi
[[ `cat $feat_dir_tr90/feats.scp|wc -l` -ne $tr90_nutts_tot ]] && echo "$feat_dir_tr90/feats.scp does not have $tr90_nutts_tot utterances" && exit 1

# Z-normalization of the unsup data: Compute the global CMVN (final.feature_transform) of the combined features using nnet; then apply cmvn to the unsup feats
if [ ${unsup_present} -gt 0 ]; then
  nnet_dir=$dir/local/feat_transform_nnet
  [ -f $nnet_dir/final.feature_transform ] || \
  $cuda_cmd $nnet_dir/log/feat_transform_nnet.log \
    local/nnet/train_pt.sh  --skip-nnet-train "true" \
      ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} \
      --delta-opts "--delta-order=$delta_order" --splice $splice --splice-step $splice_step \
      --labels-trainf "dummy" \
      --labels-crossvf "dummy" \
      --num-tgt 0 \
      --copy-feats "false" \
      ${feat_dir_tr90} ${feat_dir_cv10} lang-dummy ${ali_list[0]} ${ali_list[0]} $nnet_dir || exit 1

  # Save the scp for unsup feats in nnet_dir. This will be used during feed-forward of feats through the Z-norm transform
  utils/filter_scp.pl <(awk '{print $1}' $tr90_unsup_scps) $feat_dir_tr90/feats.scp > $nnet_dir/feats_unsup_raw.scp # filter the tr90 scp with the unsup feats scp
  #nnet-forward  $nnet_dir/final.feature_transform "$feats_unsup_tr" ark,scp:$nnet_dir/feats_unsup_znorm.ark,$nnet_dir/feats_unsup_znorm.scp
fi

# Make label(target) posteriors and frame weights of task specific feature directories
for i in $(seq 0 $[n_tasks-1]); do

	  taskid="task_$((i+1))"
    lang=${lang_list[$i]}
    dtype=${dtype_list[$i]}
    ltype=${ltype_list[$i]}
    feat_task_dir=${feat_dir}/${lang}_${ltype}_$(basename ${ddir_list[$i]})
    ali=${ali_list[$i]}
    lat=${lat_list[$i]}
    thresh=${thresh_list[$i]}
    dam=($(echo ${dup_and_merge_list[$i]}|tr '>>' ' '))
    num_copies=${dam[0]}
    
    postsubdir=local/${lang}_${dtype}_${ltype}/post_train_thresh${thresh:+_$thresh}
    postdir=$dir/$postsubdir
    [ -d $postdir ] || mkdir -p $postdir
    best_path_dir=`dirname $postdir`/bestpath_ali

    echo ""
    echo "==================================="
    echo "Generating posteriors and frame weights for: lang = $lang, data type = $dtype, label type = $ltype, ali = $ali, lat = $lat, num copies = $num_copies, posterior dir = $postdir"
    echo "==================================="

    
    if [ "$dtype" == "pt" -o "$dtype" == "semisup" ]; then
      # pt or semisupervised data
      decode_dir=$lat
	  
      local/posts_and_best_path_weights.sh --acwt $acwt --threshold $thresh \
	    --use-soft-counts $use_soft_counts --disable-upper-cap $disable_upper_cap \
        $ali $decode_dir $best_path_dir $postdir
    elif [ "$dtype" == "dt" ]; then
      # dt data
      # for monophones, append the label type to the utt id. E.g. amharic_140901_358053-1 --> p-amharic_140901_358053-1
      # for senones, do not append anything to the utt id.
      if [ "$ltype" != "p" ]; then
        ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | \
          ali-to-post ark:- ark,scp:$postdir/post.ark,$postdir/post.scp
      
        ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | \
         awk '{printf $1" ["; for (i=2; i<=NF; i++) { printf " "1; }; print " ]";}' | \
         copy-vector ark,t:- ark,t,scp:$postdir/frame_weights.ark,$postdir/frame_weights.scp || exit 1;
      else
         ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | awk -v label="$ltype" '{print   label"-"$0}' | \
          ali-to-post ark:- ark,scp:$postdir/post.ark,$postdir/post.scp
      
        ali-to-pdf $ali/final.mdl "ark:gunzip -c ${ali}/ali.*.gz |" ark,t:- | awk -v label="$ltype" '{print   label"-"$0}' | \
         awk '{printf $1" ["; for (i=2; i<=NF; i++) { printf " "1; }; print " ]";}' | \
         copy-vector ark,t:- ark,t,scp:$postdir/frame_weights.ark,$postdir/frame_weights.scp || exit 1;
      fi
    elif [ "$dtype" == "unsup" ]; then
      # unsup data
      # Do a feed-forward of the unsup data to generate Z-normalized features      
      #split_scps=""
      for n in $(seq $nj); do
        split_scps="$split_scps $nnet_dir/feats_unsup_raw.${n}.scp"
      done
      utils/split_scp.pl $nnet_dir/feats_unsup_raw.scp $split_scps
      for n in $(seq $nj); do
        feats_unsup_tr=$(cat $nnet_dir/feats_cmd) # read the feats cmd in $nnet_dir/feats_cmd
        feats_unsup_tr=$(echo $feats_unsup_tr| sed "s:train.scp:feats_unsup_raw.$n.scp:") # replace train.scp in feats cmd with unsupervised feats.scp
        nnet-forward --use-gpu=$use_gpu $nnet_dir/final.feature_transform "$feats_unsup_tr" ark:-  | \
          feat-to-post ark:- ark,scp:$postdir/post.$n.ark,$postdir/post.$n.scp || exit 1
      done
      cat $postdir/post.*.scp > $postdir/post.scp
      #$cmd JOB=1:$nj $nnet_dir/log/nnet-forward.JOB.log \
      #  feats_unsup_tr=$(echo $feats_unsup_tr| sed 's:train.scp:feats_unsup_raw.JOB.scp:') \
      #  nnet-forward --use-gpu=$use_gpu $nnet_dir/final.feature_transform "$feats_unsup_tr" ark:-  | \
      #    feat-to-post ark:- ark,scp:$postdir/post.ark,$postdir/post.scp || exit 1

      #feat-to-post "ark:$nnet_dir/feats_unsup_znorm.ark" ark,scp:$postdir/post.ark,$postdir/post.scp

      feat-to-len "scp:$tr90_unsup_scps" ark,t:- | \
        awk '{printf $1" ["; for (i=1; i<=$2; i++) { printf " "1; }; print " ]";}'| \
        copy-vector ark,t:- ark,t,scp:$postdir/frame_weights.ark,$postdir/frame_weights.scp || exit 1;
    else
      echo "Data type = $dtype not supported" && exit 1
    fi
    
    (cd $dir; rm -f post_${taskid}.scp frame_weights_${taskid}.scp;
      cp $postsubdir/post.scp post_${taskid}.scp;
      cp $postsubdir/frame_weights.scp frame_weights_${taskid}.scp)
    
    
    # Make num_copies of posteriors and frame weights
    if [ $num_copies -gt 0 ]; then
      make_num_copies="true"
      
      # Make copies of posteriors and frame weights
      awk -v num_copies=$num_copies \
      '{for (i=1; i<num_copies+1; i++) { print i"-"$1" "$2 } }' \
      $postdir/post.scp > $postdir/post_${num_copies}x.scp
  
      awk -v num_copies=$num_copies \
      '{for (i=1; i<num_copies+1; i++) { print i"-"$1" "$2 } }' \
       $postdir/frame_weights.scp > $postdir/frame_weights_${num_copies}x.scp
    fi
done

# Merge copies of task specific posteriors and weights to their specific destination tasks
if $make_num_copies ; then
  for i in $(seq 0 $[n_tasks-1]); do

	dam=($(echo ${dup_and_merge_list[$i]}|tr '>>' ' '))
	num_copies=${dam[0]}
	thresh=${thresh_list[$i]}

	dstn_task=${dam[1]}
	taskid="task_$dstn_task"
	dstn_task=$((dstn_task - 1)); # this is the task where we want the targets to be sent to

	  
	src_lang=${lang_list[$i]}; src_dtype=${dtype_list[$i]}; src_ltype=${ltype_list[$i]}
	dstn_lang=${lang_list[$dstn_task]}; dstn_dtype=${dtype_list[$dstn_task]}
	  
	if [ $num_copies -gt 0 -a ${dam[1]} -gt 0 ]; then
	  echo -e "\nTask $((i+1)): src lang = $src_lang, src data type = $src_dtype, src label type = $src_ltype, threshold = $thresh, dstn lang = $dstn_lang, dstn type = $dstn_dtype, num_copies = $num_copies"
	    
	  src_post_subdir=local/${src_lang}_${src_dtype}_${src_ltype}/post_train_thresh${thresh:+_$thresh}
	  src_post_scp=$dir/$src_post_subdir/post_${num_copies}x.scp
	  src_fwt_scp=$dir/$src_post_subdir/frame_weights_${num_copies}x.scp

	  dstn_post_scp=$dir/post_${taskid}.scp
	  dstn_fwt_scp=$dir/frame_weights_${taskid}.scp

	  echo "Append Posteriors: (src) $src_post_scp  >>  (dstn) $dstn_post_scp"
	  echo "Append Frame Weights: (src) $src_fwt_scp  >>  (dstn) $dstn_fwt_scp"
	  sort -k1,1 $src_post_scp $dstn_post_scp -o $dstn_post_scp
	  sort -k1,1 $src_fwt_scp $dstn_fwt_scp   -o $dstn_fwt_scp
	fi
  done
fi


exit 0
