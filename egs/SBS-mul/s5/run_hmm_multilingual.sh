#!/bin/bash -e

# This script shows the steps needed to build multilingual recognizer for 
# certain matched languages (Arabic, Dutch, Mandarin, Hungarian, Swahili, Urdu) 
# of the SBS corpus. A language not part of the training languages is treated as
# the test language. 
# (Adapted from the egs/gp script run.sh)

echo "This shell script may run as-is on your system, but it is recommended
that you run the commands one by one by copying and pasting into the shell."

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

. path.sh || { echo "Cannot source path.sh"; exit 1; }


stage=0;
# Set the location of the SBS speech
# ${SBS_DATADIR} is defined in path.sh 
SBS_CORPUS=${SBS_DATADIR}/audio
SBS_TRANSCRIPTS=${SBS_DATADIR}/transcripts/matched
SBS_DATA_LISTS=${SBS_DATADIR}/lists
TEXT_PHONE_LM=${SBS_DATADIR}/text-phnlm
NUMLEAVES=1200
NUMGAUSSIANS=8000
# End of config.

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: $0 [options] <train languages code> <test lang code>" 
   echo "e.g.: $0 \"AR CA HG MD UR\" \'SW\" "
   echo ""
fi

# Set the language codes for SBS languages that we will be processing
export TRAIN_LANG=$1  #"AR CA HG MD UR" exclude DT, error in dt_to_ipa.py
export TEST_LANG=$2   #"SW"

export SBS_LANGUAGES="$TRAIN_LANG $TEST_LANG" # 
export UNILANG_CODE=$(echo $TRAIN_LANG |sed 's/ /_/g')

if [ $stage -le 0 ]; then
## Data prep for monolingual and multilingual data

# Data prep: monolingual in data/$L/{train,dev,eval,wav}
local/sbs_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$SBS_CORPUS \
  --languages="$SBS_LANGUAGES"  --trans-dir=$SBS_TRANSCRIPTS --list-dir=$SBS_DATA_LISTS

# Data prep: multilingual, in data/{train,dev,eval}
local/sbs_uni_data_prep.sh "$TRAIN_LANG" "$TEST_LANG" 

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

# Lexicon prep: multilingual, in data/lang/{L.fst,L_disambig.fst,phones.txt,words.txt}
echo "universal lang"
utils/prepare_lang.sh --position-dependent-phones false \
  data/local/dict "<unk>" data/local/lang_tmp data/lang
  
# LM training (based on training text): multilingual, in data/lang_test/G.fst 
echo "universal LM"
local/sbs_format_uniphnlm.sh

# Lexicon prep + LM training (based on wiki text): monolingual, in 
# data/$L/lang_test_text_G/{L.fst, L_disambig.gst,G.fst}
for L in $SBS_LANGUAGES; do
  echo "Prep text G for $L"
  local/sbs_format_text_G.sh --text-phone-lm $TEXT_PHONE_LM $L
done

# Now move all multilingual data to data/${UNILANG_CODE}
mkdir -p data/${UNILANG_CODE}
mv data/{train,dev,eval,local,lang,lang_test} data/${UNILANG_CODE}

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

mfccdir=mfcc/${UNILANG_CODE}
for x in train dev eval; do
  (
    steps/make_mfcc.sh --nj 4 --cmd "$train_cmd" data/${UNILANG_CODE}/$x exp/make_mfcc/${UNILANG_CODE}/$x $mfccdir
    steps/compute_cmvn_stats.sh data/${UNILANG_CODE}/$x exp/make_mfcc/${UNILANG_CODE}/$x $mfccdir
  ) &
done
wait

fi

if [ $stage -le 1 ]; then
## Multilingual training

# Train monophone models
mkdir -p exp/mono/${TEST_LANG};
steps/train_mono.sh --nj 8 --cmd "$train_cmd" \
  data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/mono/${TEST_LANG}

# Make HCLG graph: with monolingual LG (data/$L/lang_test/*, LM from wiki) 
graph_dir=exp/mono/${TEST_LANG}/graph_text_G
mkdir -p $graph_dir
utils/mkgraph.sh --mono data/${TEST_LANG}/lang_test_text_G exp/mono/${TEST_LANG} \
    $graph_dir >& $graph_dir/mkgraph.log

# Decode using monophone models     
steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/${TEST_LANG}/dev \
    exp/mono/${TEST_LANG}/decode_dev_text_G &
steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/${TEST_LANG}/eval \
    exp/mono/${TEST_LANG}/decode_eval_text_G &

## Make HCLG graph, with multilingual LG (data/lang_test/*, LM from oracle)
#graph_dir=exp/mono/graph
#mkdir -p $graph_dir
#utils/mkgraph.sh --mono data/lang_test exp/mono $graph_dir
    
#for L in $SBS_LANGUAGES; do
  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    #exp/mono/decode_dev_$L &
#done


# Align features using monophone models
mkdir -p exp/mono_ali/${TEST_LANG}
steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/mono/${TEST_LANG} exp/mono_ali/${TEST_LANG}

# Train triphone models with MFCC+deltas+double-deltas
mkdir -p exp/tri1
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
  data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/mono_ali/${TEST_LANG} exp/tri1/${TEST_LANG}
  
# Make HCLG graph: with monolingual LG (data/$L/lang_test/*, LM from wiki)
graph_dir=exp/tri1/${TEST_LANG}/graph_text_G
mkdir -p $graph_dir
utils/mkgraph.sh data/${TEST_LANG}/lang_test_text_G exp/tri1/${TEST_LANG} $graph_dir

# Decode using triphone models
steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/${TEST_LANG}/dev \
    exp/tri1/${TEST_LANG}/decode_dev_text_G &
steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/${TEST_LANG}/eval \
    exp/tri1/${TEST_LANG}/decode_eval_text_G &    

## Make HCLG graph, with multilingual LG (data/lang_test/*, LM from oracle)
#graph_dir=exp/tri1/graph
#mkdir -p $graph_dir

#utils/mkgraph.sh data/lang_test exp/tri1 $graph_dir

#for L in $SBS_LANGUAGES; do
  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    #exp/tri1/decode_dev_$L &
#done
#wait

# Align features using triphone models
mkdir -p exp/tri1_ali/${TEST_LANG}
steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/tri1/${TEST_LANG} exp/tri1_ali/${TEST_LANG}

# Train with LDA+MLLT transforms
mkdir -p exp/tri2b/${TEST_LANG}
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" $NUMLEAVES $NUMGAUSSIANS \
  data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/tri1_ali/${TEST_LANG} exp/tri2b/${TEST_LANG}

# Make HCLG graph: with monolingual LG (data/$L/lang_test/*, LM from wiki)
# Decode using LDA+MLLT models. It is expected that all languages which
# have training data have significantly better error rates than the
# test language which does not have any training data.
#for L in ${TEST_LANG} ${TRAIN_LANG}; do
for L in ${TEST_LANG}; do
graph_dir=exp/tri2b/${TEST_LANG}/graph_text_G_$L
mkdir -p $graph_dir
utils/mkgraph.sh data/${TEST_LANG}/lang_test_text_G exp/tri2b/${TEST_LANG} $graph_dir


steps/decode.sh --num-threads 6 --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/tri2b/${TEST_LANG}/decode_dev_text_G_$L &
steps/decode.sh --num-threads 6 --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/tri2b/${TEST_LANG}/decode_eval_text_G_$L &
wait
    
(cd exp/tri2b/${TEST_LANG}; ln -s  decode_dev_text_G_$L decode_dev_$L; ln -s decode_eval_text_G_$L decode_eval_$L)
done

## Make HCLG graph, with multilingual LG (data/lang_test/*, LM from oracle)
#graph_dir=exp/tri2b/graph
#mkdir -p $graph_dir

#utils/mkgraph.sh data/lang_test exp/tri2b $graph_dir

#for L in $SBS_LANGUAGES; do
  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    #exp/tri2b/decode_dev_$L &
#done
#wait

# Align features using LDA+MLLT models
mkdir -p exp/tri2b_ali/${TEST_LANG}
steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
  data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/tri2b/${TEST_LANG} exp/tri2b_ali/${TEST_LANG}

# Train SAT models
steps/train_sat.sh --cmd "$train_cmd" $NUMLEAVES $NUMGAUSSIANS \
  data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/tri2b_ali/${TEST_LANG} exp/tri3b/${TEST_LANG}

# Make HCLG graph, with monolingual LG (data/$L/lang_test/*, LM from wiki)
# Decode using SAT models. It is expected that the all languages which
# have training data have significantly better error rates than the
# test language which does not have any training data.
#for L in ${TEST_LANG} ${TRAIN_LANG}; do
for L in ${TEST_LANG}; do
graph_dir=exp/tri3b/${TEST_LANG}/graph_text_G_$L
mkdir -p $graph_dir
utils/mkgraph.sh data/${TEST_LANG}/lang_test_text_G exp/tri3b/${TEST_LANG} $graph_dir

steps/decode_fmllr.sh --num-threads 6 --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/tri3b/${TEST_LANG}/decode_dev_text_G_$L &
steps/decode_fmllr.sh --num-threads 6 --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/tri3b/${TEST_LANG}/decode_eval_text_G_$L &
wait    
(cd exp/tri3b/${TEST_LANG}; ln -s  decode_dev_text_G_$L decode_dev_$L; ln -s decode_eval_text_G_$L decode_eval_$L)
done

## Make HCLG graph, with multilingual LG (data/lang_test/*, LM from oracle)
#graph_dir=exp/tri3b/graph
#mkdir -p $graph_dir
#utils/mkgraph.sh data/${TEST_LANG}/lang_test exp/tri3b $graph_dir

#for L in $SBS_LANGUAGES; do
  #steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    #exp/tri3b/decode_dev_$L &
#done
#wait

# Align features using SAT models
mkdir -p exp/tri3b_ali/${TEST_LANG}
steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
	data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/tri3b/${TEST_LANG} exp/tri3b_ali/${TEST_LANG} || exit 1;

echo "Done: `date`"


# Getting PER numbers
for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done | grep dev
for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done | grep eval
#fi
fi
