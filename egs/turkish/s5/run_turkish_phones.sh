#!/bin/bash
# E.g. > ./run_turkish_phones.sh --l2rho "0.0002" 9 500
. cmd.sh
set -e # exit on error
if [[ -f path.sh ]]; then . ./path.sh; fi

# default settings
# Acoustic model parameters (for full training set)
numLeavesTri1=2500 #2500
numGaussTri1=15000 #15000
numLeavesMLLT=2500 #2500
numGaussMLLT=15000 #15000
numLeavesSAT=2500 #2500
numGaussSAT=15000 #15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

feats_nj=10
train_nj=30
decode_nj=5

featdir=mfcc

# Directory where wav files are present
TURKROOT=${corpus_dir}/turkish/data/speech-text
# Enter either "WRD" (to measure WER) or "PHN" (to measure PER)
extn="PHN"

# L2 parameters
l2dir="../../timit/s5wb" 			# source language (l2) kaldi dir
l2rho=0 							# weight of l2 language
l2mapf="../../wsj/s5/utils/phonemap/timit2turkishmap.txt" # l2-to-l1 phone map file
l2mapcol=2                         # col in l2-to-l1 phone map file that contains the l1 phones
l2conf="conf/l2.conf"

. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <stage> <num utterances>"
  echo " e.g.: $0 --l2rho 0.02 1 all"
  echo "main options (for others, see top of script file)"  
  echo "  --l2dir                                         # source language (l2) kaldi dir"
  echo "  --l2rho <f|0>                                   # weight of l2 language"
  echo "  --l2mapf                                        # l2-to-l1 phone map file"
  echo "  --l2mapcol                                      # col in l2-to-l1 phone map file that contains the l1 phones"    
  echo "  --nj <n|1>                                      # Number of jobs (also see num-processes and num-threads)"    
  exit 1;
fi

# input args
stage=$1
# If you want to train on a subset of trn data, enter a number. Otherwise, enter "all" or "full" (which means train on full set)
num_trn_utt=$2

# [[ $num_trn_utt =~ ^[0-9]+$ ]] returns true if $num_trn_utt is a number
[[ $num_trn_utt =~ ^[0-9]+$ ]] && echo "Stage $stage: Train acoustic models on $num_trn_utt utterances" \
	|| { num_trn_utt=""; echo "Stage $stage: Train acoustic models on all utterances"; }

# Set up some configs based on input args
# 1) Number of senone labels and mixtures
# Calculate num leaves and num Gauss from the number of utterances using the rule: 
# nMD/100 = num utts x (avg durn in secs per utt), n = no. of frames per parameter, M = total #mixtures, D = params per mix.
# Each Turkish utt is about 4 secs long; skip rate at 100 frames/sec; D = 80 params per Gauss mix (mean = 39, diag cov = 39, wt = 1);
[[ ! -z $num_trn_utt ]] && {
export num_trn_utt;
numGaussTri1=`perl -e '$x=int($ENV{num_trn_utt}*4*100*2/80); print "$x";'`; # change 4*100*2/80 to 4*100*1/80 if you want to decrease num gauss 
numLeavesTri1=`echo "$numGaussTri1/5" | bc`

numLeavesMLLT=$numLeavesTri1 
numGaussMLLT=$numGaussTri1 

numLeavesSAT=$numLeavesTri1 
numGaussSAT=$numGaussTri1
}
echo -e "#Triphone States = $numLeavesTri1 \n#Triphone Mix = $numGaussTri1";

train=train${num_trn_utt}

post_fix=${num_trn_utt}_l2w${l2rho}

mono=mono${post_fix}
mono_ali=${mono}_ali

tri1=tri1${post_fix}
tri1_ali=${tri1}_ali

tri2a=tri2a${post_fix}
tri2a_ali=${tri2a}_ali

tri2b=tri2b${post_fix}
tri2b_ali=${tri2b}_ali

tri3b=tri3b${post_fix}
tri3b_ali=${tri3b}_ali

# 2) Build the conf/l2.conf file	
l2conf="conf/l2w${l2rho}.conf"
echo "$l2dir  $l2rho  $l2mapf  $l2mapcol" > $l2conf

# Training stages start from here
if [[ $stage -eq 1 ]]; then
# If $num_trn_utt is empty (train full set), then run the data prep and feat generation part.
local/turkish_data_prep.sh  $TURKROOT $extn

local/turkish_prepare_dict.sh $TURKROOT $extn

echo "Preparing lang models for type $extn ...";
if [ "$extn" = "PHN" ]; then  
utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
 data/local/dict 'sil' data/local/lang data/lang
else 
utils/prepare_lang.sh data/local/dict 'SIL' data/local/lang data/lang
fi

local/turkish_format_data.sh
fi

if [[ $stage -eq 2 ]]; then
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.   You can make a soft link if you want.
# Generate features for the full train dev test set even if you chose to train on subsets

for x in train dev test; do
  [ -s  data/$x/spk2utt ] && \
  { steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/$x exp/make_feat/$x $featdir
    #steps/make_plp.sh --nj 8 --cmd "run.pl" data/$x exp/make_feat/$x $featdir
    steps/compute_cmvn_stats.sh data/$x exp/make_feat/$x $featdir
  }
done

# Make a combined data dir where the data from all the test sets goes-- we do
# all our testing on this averaged set.  This is just less hassle.  We
# regenerate the CMVN stats as one of the speakers appears in two of the
# test sets; otherwise tools complain as the archive has 2 entries.

#utils/combine_data.sh data/test data/test_{mar87,oct87,feb89,oct89,feb91,sep92}
#steps/compute_cmvn_stats.sh data/test exp/make_feat/test $featdir
utils/subset_data_dir.sh data/train 1000 data/train.1k
fi

# Amit: Everything below this is same as rm/run.sh for word models and 
# timit/run.sh for phn models
if [[ $stage -eq 3 ]]; then

# create subset data dir if the training set is a reduced set
[[ -d $featdir && ! -d data/$train ]] && utils/subset_data_dir.sh data/train ${num_trn_utt} data/$train

echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================
steps/tl/train_mono.sh --nj "$train_nj" --cmd "$train_cmd" --langwts-config "$l2conf" data/train${num_trn_utt:-.1k} data/lang exp/$mono

#show-transitions data/lang/phones.txt exp/tri2a/final.mdl  exp/tri2a/final.occs | perl -e 'while(<>) { if (m/ sil /) { $l = <>; $l =~ m/pdf = (\d+)/|| die "bad line $l";  $tot += $1; }} print "Total silence count $tot\n";'

utils/mkgraph.sh --mono data/lang exp/$mono exp/$mono/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/$mono/graph data/dev exp/$mono/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
  exp/$mono/graph data/test exp/$mono/decode_test
fi

if [[ $stage -eq 4 ]]; then
echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
 data/$train data/lang exp/$mono exp/${mono_ali}

# cp the lang ali from mono/langali/* and save it in mono_ali/langali/*
# Ideally, we need to again do a convert-ali on lang ali using mono/final.mdl 
# since the lang ali in mono/langali/* is based on mono/38.mdl, not final.mdl
cp -r exp/${mono}/langali exp/${mono_ali} 2>/dev/null

# Train tri1, which is deltas + delta-deltas, on train data.
steps/tl/train_deltas.sh --cmd "$train_cmd" --langwts-config "$l2conf" \
 $numLeavesTri1 $numGaussTri1 data/$train data/lang exp/${mono_ali} exp/$tri1

utils/mkgraph.sh data/lang exp/$tri1 exp/$tri1/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/$tri1/graph data/dev exp/$tri1/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/$tri1/graph data/test exp/$tri1/decode_test
fi

if [[ $stage -eq 5 ]]; then
echo ============================================================================
echo "                 tri2b : LDA + MLLT Training & Decoding                    "
echo ============================================================================

steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
  data/$train data/lang exp/$tri1 exp/${tri1_ali}

steps/train_lda_mllt.sh --cmd "$train_cmd" \
 --splice-opts "--left-context=3 --right-context=3" \
 $numLeavesMLLT $numGaussMLLT data/$train data/lang exp/${tri1_ali} exp/$tri2b

utils/mkgraph.sh data/lang exp/$tri2b exp/$tri2b/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/$tri2b/graph data/dev exp/$tri2b/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/$tri2b/graph data/test exp/$tri2b/decode_test
fi

if [[ $stage -eq 6 ]]; then
echo ============================================================================
echo "              tri3b : LDA + MLLT + SAT Training & Decoding                 "
echo ============================================================================

# Align tri2 system with train data.
steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 --use-graphs true data/$train data/lang exp/$tri2b exp/${tri2b_ali}

# From tri2 system, train tri3 which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
 $numLeavesSAT $numGaussSAT data/$train data/lang exp/${tri2b_ali} exp/$tri3b

utils/mkgraph.sh data/lang exp/$tri3b exp/$tri3b/graph

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/$tri3b/graph data/dev exp/$tri3b/decode_dev

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/$tri3b/graph data/test exp/$tri3b/decode_test

steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 data/$train data/lang exp/$tri3b exp/${tri3b_ali}
fi

if [[ $stage -eq 7 ]]; then
echo ============================================================================
echo "  Not supported yet    tri1_mpe : (Delta + Delta-Delta) + MPE Training & Decoding     "
echo ============================================================================
# Align tri1 system with train data. 
# Use exp/${tri1_ali} to train on alignments from delta + delta-delta training
# Questions: "steps/align_si.sh --use-graphs true" means we need $tri1/fsts.JOB.gz to generate
# tri1 ali in a form thats reqd for mpe training. But currently $tri1 does not 
# generate such graphs ($tri1/fsts.JOB.gz). Can I generate tri ali w/o the depending
# on such graphs and proceed for mpe training? This is still sth I need to figure out.
# If we are able to train mpe on delta + delta-delta ali, only then can we compare
# evenly/fairly between ML-ML training (trained on delta + delta-delta feats) and train mpe.
# If we train mpe on lda + mllt (tri2b system), then we are inherently gaining more
# advantage by using lda + mllt.
steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 --use-graphs true data/$train data/lang exp/$tri1 exp/${tri1_ali} 
 
steps/make_denlats.sh --nj "$train_nj" --cmd "$train_cmd" \
  data/$train data/lang  exp/$tri1 exp/${tri1_denlats}

steps/train_mpe.sh data/$train data/lang exp/${tri1_ali} exp/${tri1_denlats} exp/${tri1_mpe}

steps/decode.sh --config conf/decode.config --iter 4 --nj "$decode_nj"  --cmd "$decode_cmd" \
   exp/$tri2b/graph data/test exp/${tri1_mpe}/decode_it4
   
steps/decode.sh --config conf/decode.config --iter 3 --nj "$decode_nj"  --cmd "$decode_cmd" \
   exp/$tri2b/graph data/test exp/${tri1_mpe}/decode_it3
fi

if [[ $stage -eq 8 ]]; then
echo ============================================================================
echo "                 tri2b_mpe : (LDA + MLLT) + MPE Training & Decoding     "
echo ============================================================================
# Align tri2 system with train data. 
# Use exp/${tri2b_ali} to train on alignments from LDA + MLLT training
steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 --use-graphs true data/$train data/lang exp/$tri2b exp/${tri2b_ali}
 
steps/make_denlats.sh --nj "$train_nj" --cmd "$train_cmd" \
  data/$train data/lang  exp/$tri2b exp/${tri2b_denlats}

steps/train_mpe.sh data/$train data/lang exp/${tri2b_ali} exp/${tri2b_denlats} exp/${tri2b_mpe}

steps/decode.sh --config conf/decode.config --iter 4 --nj "$decode_nj"  --cmd "$decode_cmd" \
   exp/$tri2b/graph data/test exp/${tri2b_mpe}/decode_it4
   
steps/decode.sh --config conf/decode.config --iter 3 --nj "$decode_nj"  --cmd "$decode_cmd" \
   exp/$tri2b/graph data/test exp/${tri2b_mpe}/decode_it3
fi

if [[ $stage -eq 9 ]]; then
# Karel's neural net recipe.     
[[ ! -z  ${num_trn_utt} ]] && num_trn_opt=$(echo "--num-trn-utt ${num_trn_utt}") || num_trn_opt="" 

# TL-HMM + DNN
# local/nnet/run_dnn.sh --precomp-dbn "../../multilingualdbn/s5/exp/dnn4_pretrain-dbn" \
#	--use-delta "true" --train-iters 20 ${num_trn_opt} exp/$tri2b

# TL-HMM + TL-DNN

# Convert tri1/langali using tri2b model and save in tri2b_ali/langali
# To do: Ideally langali/ali* should be converted during by the end of tri2b training 
alidir=exp/$tri1
dir=exp/${tri2b_ali}
mkdir -p $dir/langali
nj=$(cat $alidir/num_jobs 2>/dev/null)
$train_cmd JOB=1:$nj $dir/log/langali/convert.langali.JOB.log \
		convert-ali $alidir/final.mdl $dir/final.mdl $dir/tree \
		"ark,t:gunzip -c $alidir/langali/ali.lang0.JOB.gz|" "ark,t:|gzip -c >$dir/langali/ali.lang0.JOB.gz" || exit 1;
		
l2iters="2 4 6 8 10 12 14 16"
for i in $l2iters
do
steps/tl/nnet/run_dnn_sequential.sh --precomp-dbn "../../multilingualdbn/s5/exp/dnn4_pretrain-dbn" \
--use-delta "true" --train-iters 10 --l2-iters $i  --post-fix "l2iter$i"  ${num_trn_opt} $l2conf exp/$tri2b
done


minibatch_size=256
xent_wts="0 0.0001 0.0002 0.0004 0.0006 0.0008 0.001 0.002 0.004 0.006 0.008 0.01 0.02 0.04 0.06 0.08 0.1 0.2 0.4 0.6 0.8 1"
for w in $xent_wts 
do
steps/tl/nnet/run_dnn_joint.sh --precomp-dbn "../../multilingualdbn/s5/exp/dnn4_pretrain-dbn" \
--use-delta "true" --train-iters 20 --xent-wt $w --minibatch-size ${minibatch_size} --post-fix "xent${w}_l2100"  ${num_trn_opt} --num-trn-utt-l2 100 $l2conf exp/$tri2b
done
# Karel's CNN recipe.
# local/nnet/run_cnn.sh
fi
rm $l2conf
