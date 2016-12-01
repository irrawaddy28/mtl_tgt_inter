#!/bin/bash

# Top-level script to compute probabilistic phone error rate (PPER)
# Usage: . ./main.sh
#
# How is PPER computed?
# Step 1: The crowd source PTs are pruned to retain the most reliable transcripts
# Step 2: The probabilities on the arcs of the pruned PTs are stripped. 
#		  Such PTs are called unweighted PTs. If you don't prune the 
#		  probabilites on the arcs, the PTs are called weighted PTs.
# Step 3: The edit distance between the 1-best path in the ASR decoding 
#		  lattice and the unweighted (or weighted) pruned PT is computed
#
# The script computing PPER is: ./evaluate_zerotranscripts.sh (Author: Preethi Jyothi)
# 
# Note: Before running this script, make sure you have compiled
# convert_kaldilat2fst.cc by running the script ./convert.sh


## ================= Settings for Dinka =======================
dir="../../"
lang="DI"
# Multi-HMM
expdir="$dir/exp/tri3b/$lang/decode_eval_text_G_$lang"

# Multi-DNN
#expdir="$dir/exp/dnn4_pretrain-dbn_dnn/$lang/monosoftmax_dt/decode_eval_text_G_$lang"

# MAP-HMM
#expdir="$dir/exp/tri3c/$lang/decode_eval_text_G_$lang"

# DNN-2
#expdir="$dir/exp/dnn4_pretrain-dbn_dnn/$lang/multisoftmax_pt_fw0.6_cop0/decode_block_1_eval_text_G_$lang"
## ==============================================

. ../../path.sh

awk '{print $1}' $dir/data/$lang/eval/text > utt_ids.txt

# W/o "--w" option, we get the edit distance b/w 1-best path in ASR lattice
# and unweighted pruned PT.
./evaluate_zerotranscripts.sh "ark:gunzip -c $expdir/lat.*.gz|" \
  "$SBS_DATADIR/pt-lats/held-out-$lang" \
  utt_ids.txt  \
  $dir/data/$lang/lang_test_text_G/words.txt \
  lat_fsts
  
# With "--w" option, we get the edit distance b/w 1-best path in ASR lattice
# and weighted pruned PT.
./evaluate_zerotranscripts.sh --w "ark:gunzip -c $expdir/lat.*.gz|" \
 "$SBS_DATADIR/pt-lats/held-out-$lang" \
  utt_ids.txt  \
  $dir/data/$lang/lang_test_text_G/words.txt \
  lat_fsts


## ================= Results for Dinka =======================
## Unweighted scores
# Multi-HMM
# Avg. edit distance = 62.28301886792452830188 and avg. loglikelihood ratio = 0 over 53 utterances in the data set utt_ids.txt

# Multi-DNN
# Avg. edit distance = 64.09433962264150943396 and avg. loglikelihood ratio = 0 over 53 utterances in the data set utt_ids.txt

# MAP-HMM
# Avg. edit distance = 58.58490566037735849056 and avg. loglikelihood ratio = 0 over 53 utterances in the data set utt_ids.txt

# DNN-2
# Avg. edit distance = 60.64150943396226415094 and avg. loglikelihood ratio = 0 over 53 utterances in the data set utt_ids.txt


## Weighted scores
# Multi-HMM
# Avg. edit distance = -60.38910211320754716981 and avg. loglikelihood ratio = 0 over 53 utterances in the data set utt_ids.txt

# Multi-DNN
# Avg. edit distance = -59.45177996415094339622 and avg. loglikelihood ratio = 0 over 53 utterances in the data set utt_ids.txt

# MAP-HMM
# Avg. edit distance = -63.07980652264150943396 and avg. loglikelihood ratio = 0 over 53 utterances in the data set utt_ids.txt

# DNN-2
# Avg. edit distance = -61.76130885094339622641 and avg. loglikelihood ratio = 0 over 53 utterances in the data set utt_ids.txt
## ==============================================
