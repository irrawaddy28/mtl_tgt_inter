#!/bin/bash -e

# This script shows the steps needed to build a recognizer for certain matched languages (Arabic, Dutch, Mandarin, Hungarian, Swahili, Urdu) of the SBS corpus. 
# (Adapted from the egs/gp script run.sh)

echo "This shell script may run as-is on your system, but it is recommended 
that you run the commands one by one by copying and pasting into the shell."
#exit 1;

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

. path.sh || { echo "Cannot source path.sh"; exit 1; }

stage=1
graph_affix="" # if you want to decode the test data using LMs built from 
			   # text data taken from:
			   # SBS train set, set graph_affix to "" (supported)
			   # wiki, set graph_affix to "_text_G" (not supported yet)
			   
# Set the location of the SBS speech 
SBS_CORPUS=${SBS_DATADIR}/audio
SBS_TRANSCRIPTS=${SBS_DATADIR}/transcripts/matched
SBS_DATA_LISTS=${SBS_DATADIR}/lists
TEXT_PHONE_LM=${SBS_DATADIR}/text-phnlm
NUMLEAVES=1200
NUMGAUSSIANS=8000
# End of config

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "Usage: $0 [options] <train language code>" 
   echo "e.g.: $0 \'SW\" "
   echo ""
fi

# Set the language codes for SBS languages that we will be processing
export TRAIN_LANG=$1  #"AR CA HG MD UR" exclude DT, error in dt_to_ipa.py

export SBS_LANGUAGES="AR CA HG MD SW UR"

#### LANGUAGE SPECIFIC SCRIPTS HERE ####
if [ $stage -le 0 ]; then
## Data prep for monolingual data

# Data prep: monolingual in data/$L/{train,dev,eval,wav}
local/sbs_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$SBS_CORPUS \
  --languages="$SBS_LANGUAGES"  --trans-dir=$SBS_TRANSCRIPTS --list-dir=$SBS_DATA_LISTS
  
# Dictionaries: monolingual, in data/$L/local/dict ; multilingual in data/local/dict
echo "dict prep"
local/sbs_dict_prep.sh $SBS_LANGUAGES

# Lexicon prep: monolingual, in data/$L/lang/{L.fst,L_disambig.fst,phones.txt,words.txt}
for L in $SBS_LANGUAGES; do
  echo "lang prep: $L"
  utils/prepare_lang.sh --position-dependent-phones false \
    data/$L/local/dict "<unk>" data/$L/local/lang_tmp data/$L/lang
done

# LM training (based on training text): monolingual, in data/$L/lang_test/G.fst
for L in $SBS_LANGUAGES; do
  echo "LM prep: $L"
  local/sbs_format_phnlm.sh $L
done

# Lexicon + LM (based on wiki text): monolingual, in 
# data/$L/lang_test_text_G/{L.fst, L_disambig.gst,G.fst}
for L in $SBS_LANGUAGES; do
  echo "Prep text G for $L"
  local/sbs_format_text_G.sh --text-phone-lm $TEXT_PHONE_LM $L
done

echo "MFCC prep"
# Make MFCC features.
for L in $SBS_LANGUAGES; do
  mfccdir=mfcc/$L
  for x in train dev eval; do
    (
      steps/make_mfcc.sh --nj 4 --cmd "$train_cmd" data/$L/$x exp/make_mfcc/$L/$x $mfccdir
      steps/compute_cmvn_stats.sh data/$L/$x exp/make_mfcc/$L/$x $mfccdir
    ) &
  done
done
wait

fi

exp=exp/monolingual
if [ $stage -le 1 ]; then
for L in $TRAIN_LANG; do

  # Train monophone models
  mkdir -p $exp/mono/$L;
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" \
    data/$L/train data/$L/lang $exp/mono/$L
  
  graph_dir=$exp/mono/$L/graph${graph_affix}
  mkdir -p $graph_dir
  utils/mkgraph.sh --mono data/$L/lang_test${graph_affix} $exp/mono/$L $graph_dir
  
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    $exp/mono/$L/decode_dev${graph_affix} &    
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    $exp/mono/$L/decode_eval${graph_affix} &
      
  mkdir -p $exp/mono_ali/$L
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    data/$L/train data/$L/lang $exp/mono/$L $exp/mono_ali/$L
  
  # Train triphone models with MFCC+deltas+double-deltas
  mkdir -p $exp/tri1/$L
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
    data/$L/train data/$L/lang $exp/mono_ali/$L $exp/tri1/$L
  
  graph_dir=$exp/tri1/$L/graph${graph_affix}
  mkdir -p $graph_dir
  
  utils/mkgraph.sh data/$L/lang_test${graph_affix} $exp/tri1/$L $graph_dir

  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    $exp/tri1/$L/decode_dev${graph_affix} &    
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    $exp/tri1/$L/decode_eval${graph_affix} &  

  mkdir -p $exp/tri1_ali/$L
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    data/$L/train data/$L/lang $exp/tri1/$L $exp/tri1_ali/$L

  # Train with LDA+MLLT transforms
  mkdir -p $exp/tri2b/$L
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" $NUMLEAVES $NUMGAUSSIANS \
    data/$L/train data/$L/lang $exp/tri1_ali/$L $exp/tri2b/$L  
  
  graph_dir=$exp/tri2b/$L/graph${graph_affix}
  mkdir -p $graph_dir
        
  utils/mkgraph.sh data/$L/lang_test${graph_affix} $exp/tri2b/$L $graph_dir
  
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    $exp/tri2b/$L/decode_dev${graph_affix} &
  steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    $exp/tri2b/$L/decode_eval${graph_affix} &  

  mkdir -p $exp/tri2b_ali/$L
  
  steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
    data/$L/train data/$L/lang $exp/tri2b/$L $exp/tri2b_ali/$L
  
  # Train SAT models
  steps/train_sat.sh --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
    data/$L/train data/$L/lang $exp/tri2b_ali/$L $exp/tri3b/$L
  
  graph_dir=$exp/tri3b/$L/graph${graph_affix}
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test $exp/tri3b/$L $graph_dir
  
  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    $exp/tri3b/$L/decode_dev${graph_affix} &
  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    $exp/tri3b/$L/decode_eval${graph_affix} &  
done
wait

fi
# Getting PER numbers
# for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
