#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains DNN with <MultiSoftmax> output on top of FMLLR features.
# The network is trained on multiple languages simultaneously creating a separate softmax layer per language
# while sharing hidden layers across all languages.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0
#. utils/parse_options.sh || exit 1;

set -u 
set -e
set -o pipefail
set -x

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
thisdata=data-fmllr-multisoftmax/train
train_tr90_multilingual=$thisdata/train_tr90_multilingual
train_cv10_multilingual=$thisdata/train_cv10_multilingual
if [ $stage -le 0 ]; then

  tr_90=""; cv_10="";
  for i in  $(seq 0 $nlangs)
  do
    # Store fMLLR features in language dep directories, so we can train on them easily,
	dir=$thisdata/${lang[i]}; mkdir -p $dir
	gmm_ali=${ali[i]}
	data_train=${data[i]}	
	echo "Language = ${lang[i]}: Generating features from datadir = ${data[i]} and saving in $dir"
	
	# To Do: Add language id as a prefix to the uttids (keys) in the data dir
	# tmpd=$(mktemp -d)
	# utils/copy_data_dir.sh --utt-prefix ${lang[i]} --spk-prefix ${lang[i]} ${data_train} $tmpd || exit 1
	# cp -R ${data_train}/split* $tmpd	
	# --keytag-opts "--utt-prefix ${lang[i]}  --spk-prefix ${lang[i]}"
	
	steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmm_ali} \
     $dir ${data_train} ${gmm_ali} $dir/log $dir/data || exit 1    #${data_train}    
    
    # utils/copy_data_dir.sh --utt-prefix ${lang[i]} --spk-prefix ${lang[i]} ${data_train} $dir || exit 1    
    
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


# Determine the lang dependent output nodes per softmax layer and the sum of all such output nodes
thisdir=exp/dnn4e-fmllr_multisoftmax # this is our current expt dir
output_dim=
ali_dim_csl=
for i in $(seq 0 $nlangs)
do
	ali_dim[i]=$(hmm-info ${ali[i]}/final.mdl | grep pdfs | awk '{ print $NF }')
	echo "Output dim of block softmax for ${lang[i]} = ${ali_dim[i]}"
	output_dim=$(( output_dim + ${ali_dim[i]} ))
	[[ -z ${ali_dim_csl} ]] && ali_dim_csl="${ali_dim[i]}" || ali_dim_csl="${ali_dim_csl}:${ali_dim[i]}" 
done
echo "Sum of all block output dims = $output_dim (${ali_dim_csl})"

# Prepare the merged targets
if [ $stage -le 1 ]; then    
  post_scp_list=
  for i in $(seq 0 $nlangs)
  do
	dir=$thisdir/${lang[i]};
		
	# To Do: Copy the language dep. alignments to the expt dir but prefix the uttids (keys) of the alignments using language id
	# The one line below adds the prefix to the uttids. We add prefix to ensure uttids from different languages are unique.
	# But for now, it is safe to assume that even w/o the prefix, the uttids are unique.
	# copy-int-vector "ark:gunzip -c ${ali[i]}/ali.*.gz |" ark,t:- | awk -v prefix=${lang[i]} '{ $1=prefix $1; print; }' | \
  	#	gzip -c >$dir/ali_${lang[i]}.gz
  	# The line below does not add prefix to uttids.
  	copy-int-vector "ark:gunzip -c ${ali[i]}/ali.*.gz |" ark,t:- | gzip -c >$dir/ali_${lang[i]}.gz
  	
  	# Store posteriors to disk, indexed by 'scp',	
  	ali-to-pdf ${ali[i]}/final.mdl "ark:gunzip -c $dir/ali_${lang[i]}.gz |" ark,t:- | \
		ali-to-post ark,t:- ark,scp:$dir/post_${lang[i]}.ark,$dir/post_${lang[i]}.scp	
	post_scp_list="${post_scp_list}  'scp:$dir/post_${lang[i]}.scp'"
  done
  
  feats_scp_list=
  for i in $(seq 0 $nlangs)
  do
	[[ -z ${feats_scp_list} ]] && feats_scp_list="${data[i]}/feats.scp" || feats_scp_list="${feats_scp_list}  ${data[i]}/feats.scp" 
  done
  featlen="ark:feat-to-len 'scp:cat ${feats_scp_list} |' ark,t:- |"   # print number of frames for every utterance in feats.scp     
  
  paste-post --allow-partial=true "$featlen" ${ali_dim_csl} ${post_scp_list} \
	ark,scp:$thisdir/pasted_post.ark,$thisdir/pasted_post.scp 2>$dir/log/paste_post.log  
fi

exit 1;
# Train <MultiSoftmax> system,
if [ $stage -le 2 ]; then
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_debug.sh \
	  --nnet-binary "false" \
	  --hid-layers 6 \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --labels-trainf "scp:$dir/pasted_post.scp" --num-tgt $output_dim \
      --proto-opts "--block-softmax-dims=${ali_dim_csl}" \
      --learn-rate 0.008 \
      ${train_tr90_multilingual} ${train_cv10_multilingual} lang-dummy ali-dummy ali-dummy $dir || exit 1;
  exit 0;   
  # Create files used in decdoing, missing due to --labels use,
  analyze-counts --binary=false "$ali1_pdf" $dir/ali_train_pdf.counts || exit 1
  copy-transition-model --binary=false $ali1_dir/final.mdl $dir/final.mdl || exit 1
  cp $ali1_dir/tree $dir/tree || exit 1
  exit 1;
  # Rebuild network, <MultiSoftmax> is removed, and neurons from 1st block are selected,
  nnet-concat "nnet-copy --remove-last-components=1 $dir/final.nnet - |" \
    "echo '<Copy> <InputDim> $output_dim <OutputDim> $ali1_dim <BuildVector> 1:$ali1_dim </BuildVector>' | nnet-initialize - - |" \
    "echo '<Softmax> <InputDim> $ali1_dim <OutputDim> $ali1_dim' | nnet-initialize - - |" \
    $dir/final.nnet.lang1 || exit 1
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --nnet $dir/final.nnet.lang1 \
    $gmm/graph $dev $dir/decode || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --nnet $dir/final.nnet.lang1 \
    $gmm/graph_ug $dev $dir/decode_ug || exit 1;
fi

exit 0

# TODO, 
# make nnet-copy support block selection, 
# - either by replacing <BlockSoftmax> by <Softmax> and shrinking <AffineTransform>,
# - or by appending <Copy> transform,
#
# Will it be compatible with other scripts/tools which assume <Softmax> at the end?
# Or is it better to do everything visually in master script as now?... 
# Hmmm, need to think about it...

# Train baseline system with <Softmax>,
if [ $stage -le 3 ]; then
  dir=exp/dnn4e-fbank_baseline
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --learn-rate 0.008 \
      ${train}_tr90 ${train}_cv10 data/lang ${gmm}_ali ${gmm}_ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph_ug $dev $dir/decode_ug || exit 1;
fi

echo Success
exit 0

#j=0;
#for i in "$@"
#do
#echo "i = $i argv = %%$@%%"
#case $i in
    #--l=*)
    #langcode[j]="${i#*=}"    
    #shift 
    #echo "j = $j, ${langcode[$j]}, after shift $@"
	#for k in "$@"
	#do
		#echo "Inside : argv = $@"
		#case $k in	
		#--ali=*)
			#ali[j]="${k#*=}"
			#shift
			#echo "j = $j, ${ali[$j]}, after shift $@, i = $i"
			#;; 
		#--data=*)
			#data[j]="${k#*=}"
			#shift
			#echo "j = $j, ${data[$j]}, after shift $@, i = $i" 
			#;; 
		#*)
		#echo "Unknown argument inside: ${i#*=}, exiting"; exit 1 ;;    
		#esac
	#done
	#;;    
    #*)
    #echo "Unknown argument outside: $i, exiting"; exit 1 ;;    
#esac
#j=$(($j + 1))
#done

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
