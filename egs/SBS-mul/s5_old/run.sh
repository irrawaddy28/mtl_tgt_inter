#!/bin/bash -e

# This script shows the steps needed to build a recognizer for certain matched languages (Arabic, Dutch, Mandarin, Hungarian, Swahili, Urdu) of the SBS corpus. 
# (Adapted from the egs/gp script run.sh)

echo "This shell script may run as-is on your system, but it is recommended 
that you run the commands one by one by copying and pasting into the shell."
#exit 1;

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

. path.sh || { echo "Cannot source path.sh"; exit 1; }

# Set the location of the SBS speech 
SBS_CORPUS=/export/ws15-pt-data/data/audio
SBS_TRANSCRIPTS=/export/ws15-pt-data/data/transcripts/matched
SBS_DATA_LISTS=/export/ws15-pt-data/data/lists
NUMLEAVES=2000  #1200
NUMGAUSSIANS=9000 #8000

# Set the language codes for SBS languages that we will be processing
export SBS_LANGUAGES="SW AR UR DT HG MD"
export TRAIN_LANG="AR UR DT HG MD"
export TEST_LANG="SW"

#### LANGUAGE SPECIFIC SCRIPTS HERE ####
local/sbs_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$SBS_CORPUS \
  --languages="$SBS_LANGUAGES"  --trans-dir=$SBS_TRANSCRIPTS --list-dir=$SBS_DATA_LISTS

local/sbs_uni_data_prep.sh "$TRAIN_LANG" "$TEST_LANG"

echo "dict prep"
local/sbs_dict_prep.sh $SBS_LANGUAGES  2>&1 | tee data/prepare_dict.log 

for L in $SBS_LANGUAGES; do
  echo "lang prep: $L"
  utils/prepare_lang.sh --position-dependent-phones false \
    data/$L/local/dict "<unk>" data/$L/local/lang_tmp data/$L/lang \
    >& data/$L/prepare_lang.log || exit 1;
done

for L in $SBS_LANGUAGES; do
  echo "LM prep: $L"
  local/sbs_format_phnlm.sh $L >& data/$L/format_lm.log || exit 1;
done

echo "universal lang"
utils/prepare_lang.sh --position-dependent-phones false \
  data/local/dict "<unk>" data/local/lang_tmp data/lang \
  >& data/prepare_lang.log || exit 1;

echo "universal LM"
local/sbs_format_uniphnlm.sh >& data/format_lm.log || exit 1;


echo "MFCC prep"
# Make MFCC features.
for L in $SBS_LANGUAGES; do
  mfccdir=mfcc/$L
  for x in train dev eval; do
    (      
		steps/make_mfcc.sh --nj 4 --cmd "$train_cmd" data/$L/$x exp/$L/make_mfcc/$x $mfccdir;
		utils/fix_data_dir.sh data/$L/$x  
		steps/compute_cmvn_stats.sh data/$L/$x exp/$L/make_mfcc/$x $mfccdir;
		utils/fix_data_dir.sh data/$L/$x      
    ) &
  done
done
wait;

mfccdir=mfcc
# for x in train eval; do
for x in train dev eval; do
  (    
	 steps/make_mfcc.sh --nj 4 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir;
	 utils/fix_data_dir.sh data/$x  
	 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir;
	 utils/fix_data_dir.sh data/$x  	 
   
  ) &
done
wait;

mkdir -p exp/mono;
steps/train_mono.sh --nj 8 --cmd "$train_cmd" \
  data/train data/lang exp/mono 2>&1 | tee exp/mono/train.log

graph_dir=exp/mono/graph
mkdir -p $graph_dir
utils/mkgraph.sh --mono data/lang_test exp/mono \
  $graph_dir 2>&1 | tee $graph_dir/mkgraph.log

for L in $SBS_LANGUAGES; do
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/mono/decode_eval_$L &
done
wait

# Training/decoding triphone models
mkdir -p exp/mono_ali
steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  data/train data/lang exp/mono exp/mono_ali \
  2>&1 | tee exp/mono_ali/align.log 

# Training triphone models with MFCC+deltas+double-deltas
mkdir -p exp/tri1
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
  data/train data/lang exp/mono_ali exp/tri1 2>&1 | tee exp/tri1/train.log || exit 1;

graph_dir=exp/tri1/graph
mkdir -p $graph_dir

utils/mkgraph.sh data/lang_test exp/tri1 $graph_dir 2>&1 \
  | tee $graph_dir/mkgraph.log 

for L in $SBS_LANGUAGES; do
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/tri1/decode_eval_$L &
done
wait


mkdir -p exp/tri1_ali
steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali 2>&1 \
  | tee exp/tri1_ali/align.log

#mkdir -p exp/tri2a
#steps/train_deltas.sh --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
  #data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

#graph_dir=exp/tri2a/graph
#mkdir -p $graph_dir
      
#utils/mkgraph.sh data/lang_test exp/tri2a $graph_dir \
  #>& $graph_dir/mkgraph.log 

#for L in $SBS_LANGUAGES; do
  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    #exp/tri2a/decode_eval_$L &
#done
#wait

mkdir -p exp/tri2b
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" $NUMLEAVES $NUMGAUSSIANS \
  data/train data/lang exp/tri1_ali exp/tri2b || exit 1;

# Train with LDA+MLLT transforms
graph_dir=exp/tri2b/graph
mkdir -p $graph_dir
      
utils/mkgraph.sh data/lang_test exp/tri2b $graph_dir 2>&1 \
  | tee $graph_dir/mkgraph.log 

for L in $SBS_LANGUAGES; do
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/tri2b/decode_dev_$L &
done

for L in $SBS_LANGUAGES; do
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/tri2b/decode_eval_$L &
done
wait

mkdir -p exp/tri2b_ali
steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
  data/train data/lang exp/tri2b exp/tri2b_ali 2>&1 \
  | tee exp/tri2b_ali/align.log 

steps/train_sat.sh --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
  data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;

graph_dir=exp/tri3b/graph
mkdir -p $graph_dir
utils/mkgraph.sh data/lang_test exp/tri3b $graph_dir 2>&1 \
  | tee $graph_dir/mkgraph.log

for L in $SBS_LANGUAGES; do
  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/tri3b/decode_dev_$L &
done

for L in $SBS_LANGUAGES; do
  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/tri3b/decode_eval_$L &
done
wait
# Getting PER numbers
# for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
