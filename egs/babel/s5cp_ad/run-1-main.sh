#!/bin/bash
# E.g.: > ./run-1-main.sh --tri5-only "true" conf/lang/101-cantonese-limitedLP.official.conf
# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.
# Base period (BP) languages: Pashto, Turkish, Tagalog, Cantonese, Vietnamese
# Optional Period One (OP1) languages: Haitian, Lao, Zulu, Assamese, Bengali, Tamil

tri5_only=false
sgmm5_only=false
data_only=false
sil="sil"          # how do you want to label the silence phones?
sys_type="phone"   # can be "phone" (eval on PER) or "word" (eval on WER)
use_mfcc="true"
remove_tags="true" # remove tags _[0-9], ", % from phones?

# bengali: "conf/lang/103-bengali-limitedLP.official.conf"
# assamese: "conf/lang/102-assamese-limitedLP.official.conf"
# cantonese: "conf/lang/101-cantonese-limitedLP.official.conf"
# pashto: "conf/lang/104-pashto-limitedLP.official.conf"
# tagalog: "conf/lang/106-tagalog-limitedLP.official.conf"
# turkish: "conf/lang/105-turkish-limitedLP.official.conf"
# vietnamese: "conf/lang/107-vietnamese-limitedLP.official.conf"
# haitian: "conf/lang/201-haitian-limitedLP.official.conf"
# lao: "conf/lang/203-lao-limitedLP.official.conf"
# zulu: "conf/lang/206-zulu-limitedLP.official.conf"
# tamil: "conf/lang/204-tamil-limitedLP.official.conf"
langconf=$3

[[ -f $langconf ]] && cp $langconf ./lang.conf

[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

[[ -f path.sh ]] && . ./path.sh
sed -i.bak "s:/export/:${corpus_dir}/:g" lang.conf

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

[[ $sys_type == "phone" ]] && \
{ convert_word_to_phone="true"; oovSymbol="<oov>"; } || \
{ convert_word_to_phone="false"
  # here we retain the $oovSymbol defined in lang.conf file 
} 

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
#set -u           #Fail on an undefined variable

#Preparing dev2h and train directories
if [ ! -f data/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh "$train_data_dir" "$train_data_list" ./data/raw_train_data
    train_data_dir=`readlink -f ./data/raw_train_data`
    touch data/raw_train_data/.done
fi
nj_max=`cat $train_data_list | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
train_data_dir=`readlink -f ./data/raw_train_data`
echo "train_data_dir = $train_data_dir"

if [ ! -d data/raw_dev2h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV2H set"
  echo ---------------------------------------------------------------------  
  local/make_corpus_subset.sh "$dev2h_data_dir" "$dev2h_data_list" ./data/raw_dev2h_data || exit 1
fi

if [ ! -d data/raw_dev10h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV10H set"
  echo ---------------------------------------------------------------------  
  local/make_corpus_subset.sh "$dev10h_data_dir" "$dev10h_data_list" ./data/raw_dev10h_data || exit 1
fi

nj_max=`cat $dev2h_data_list | wc -l`
if [[ "$nj_max" -lt "$decode_nj" ]] ; then
  echo "The maximum reasonable number of jobs is $nj_max -- you have $decode_nj! (The training and decoding process has file-granularity)"
  exit 1
  decode_nj=$nj_max
fi

mkdir -p data/local
if [[ ! -f data/local/lexicon.txt || data/local/lexicon.txt -ot "$lexicon_file" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing lexicon in data/local on" `date`
  echo ---------------------------------------------------------------------  
  local/make_lexicon_subset.sh $train_data_dir/transcription $lexicon_file data/local/filtered_lexicon.txt
  [[ $remove_tags == "true" ]] && sed -E -i 's/_[0-9]|"|%//g' data/local/filtered_lexicon.txt
  phoneme_mapping=$(cat conf/sampa2ipa.txt|sed '/^;/d'|awk '{print $1, " = ", $2, ";"}' |tr '\n' ' ')
  phoneme_mapping=$(echo $phoneme_mapping; echo $phoneme_mapping_overrides)
  local/prepare_lexicon.pl  --phonemap "$phoneme_mapping" --sil "$sil" \
    $lexiconFlags data/local/filtered_lexicon.txt data/local
fi

if [[ ! -f data/train/wav.scp || data/train/wav.scp -ot "$train_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing acoustic training lists in data/train on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/train
  # What are fragments? Frags include
  # a) Mispronunciations. e.g. "representive" (mispron. word in audio) -> *representative* (word transcribed with the right spelling in text)
  # b) Stumbling speech. e.g.  "to- tomorrow" (speaker stumbles midway in the word tomorrow) -> to- tomorrow (word transcribed up to the cut off point and hyphenated)
  # c) Truncated words at the start or end of a recording. e.g. "tisfactory" (truncated word in audio) -> ~satisfactory (word transcribed w/o truncation but marked with a ~ to denote truncation)
  local/prepare_acoustic_training_data.pl \
    --vocab data/local/lexicon.txt --convert-word-to-phone  $convert_word_to_phone --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log  
fi

if [[ ! -f data/dev2h/wav.scp || data/dev2h/wav.scp -ot ./data/raw_dev2h_data/audio ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing dev2h data lists in data/dev2h on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/dev2h
  local/prepare_acoustic_training_data.pl \
    --vocab data/local/lexicon.txt  --convert-word-to-phone $convert_word_to_phone  --fragmentMarkers \-\*\~ \
    `pwd`/data/raw_dev2h_data data/dev2h > data/dev2h/skipped_utts.log || exit 1
fi

if [[ $convert_word_to_phone == "true" ]]; then
	cp data/local/lexicon.txt data/local/lexicon_words.txt    
	perl utils/extract_phones_from_lexicon.pl data/local/lexicon_words.txt > data/local/lexicon.txt
	#sed -i "s/\<oov\>/$oovSymbol/" data/local/lexicon.txt
fi

if [[ ! -f data/dev2h/glm || data/dev2h/glm -ot "$glmFile" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing dev2h stm files in data/dev2h on" `date`
  echo ---------------------------------------------------------------------
  if [ -z $dev2h_stm_file ]; then 
    echo "WARNING: You should define the variable stm_file pointing to the IndusDB stm"
    echo "WARNING: Doing that, it will give you scoring close to the NIST scoring.    "
    local/prepare_stm.pl --fragmentMarkers \-\*\~ data/dev2h || exit 1
  else
    local/augment_original_stm.pl $dev2h_stm_file data/dev2h || exit 1
  fi
  [ ! -z $glmFile ] && cp $glmFile data/dev2h/glm

fi

mkdir -p data/lang
rm -rf data/lang/*
if [[ ! -f data/lang/L.fst || data/lang/L.fst -ot data/local/lexicon.txt ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating L.fst etc in data/lang on" `date`
  echo ---------------------------------------------------------------------
  utils/prepare_lang.sh \
    --share-silence-phones true --position_dependent_phones false \
    data/local $oovSymbol data/local/tmp.lang data/lang
fi

# Create word_boundary.{txt, int} files under data/lang/phones. This reqd for phones if we are doing STM (segment time marked) based WER evaluation using SCLITE
if [[ $convert_word_to_phone == "true" ]]; then
dir=data/lang
cat $dir/phones/{silence,nonsilence}.txt | \
    awk '/<oov>/{print $1, "nonword"; next;} /<sss>/{print $1, "nonword"; next; }
         /<vns>/{print $1, "nonword"; next;} /sil/{print $1, "nonword"; next; }
         {print $1, "singleton";} ' > $dir/phones/word_boundary.txt
                  
utils/sym2int.pl -f 1 $dir/phones.txt <$dir/phones/word_boundary.txt \
    > $dir/phones/word_boundary.int || exit 1;         
fi

# We will simply override the default G.fst by the G.fst generated using SRILM
if [[ ! -f data/srilm/lm.gz || data/srilm/lm.gz -ot data/train/text ]]; then
  echo ---------------------------------------------------------------------
  echo "Training SRILM language models on" `date`
  echo ---------------------------------------------------------------------
  local/train_lms_srilm.sh --sys-type $sys_type --dev-text data/dev2h/text \
    --train-text data/train/text data data/srilm 
fi

if [[ ! -f data/lang/G.fst || data/lang/G.fst -ot data/srilm/lm.gz ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating G.fst on " `date`
  echo ---------------------------------------------------------------------
  local/arpa2G.sh data/srilm/lm.gz data/lang data/lang
fi
decode_nj=$dev2h_nj


echo ---------------------------------------------------------------------
echo "Starting plp feature extraction for data/train in plp on" `date`
echo ---------------------------------------------------------------------
#if [ ! -f data/train/.plp.done ]; then
#if $use_pitch; then
   #steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_plp_pitch/train plp
#else
    #steps/make_plp.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_plp/train plp
#fi
  #utils/fix_data_dir.sh data/train
  #steps/compute_cmvn_stats.sh data/train exp/make_plp/train plp
  #utils/fix_data_dir.sh data/train
  #touch data/train/.plp.done
#fi

if [[ ! -f data/train/.mfcc.done ]]; then
  (
	steps/make_mfcc.sh --nj $train_nj --cmd "$train_cmd" data/train exp/make_mfcc/train mfcc
	utils/fix_data_dir.sh data/train
	steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc
	utils/fix_data_dir.sh data/train
	touch data/train/.mfcc.done
  ) &
fi    
wait;

mkdir -p exp

if [ ! -f data/train_sub3/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting monophone training data in data/train_sub[123] on" `date`
  echo ---------------------------------------------------------------------
  numutt=`cat data/train/feats.scp | wc -l`;
  utils/subset_data_dir.sh data/train  5000 data/train_sub1
  if [ $numutt -gt 10000 ] ; then
    utils/subset_data_dir.sh data/train 10000 data/train_sub2
  else
    (cd data; ln -s train train_sub2 )
  fi
  if [ $numutt -gt 20000 ] ; then
    utils/subset_data_dir.sh data/train 20000 data/train_sub3
  else
    (cd data; ln -s train train_sub3 )
  fi

  touch data/train_sub3/.done
fi

if $data_only; then
  echo "--data-only is true" && exit 0
fi

if [ ! -f exp/mono/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) monophone training in exp/mono on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mono.sh \
    --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
    data/train_sub1 data/lang exp/mono
  touch exp/mono/.done
fi

if [ ! -f exp/tri1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) triphone training in exp/tri1 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
    data/train_sub2 data/lang exp/mono exp/mono_ali_sub2
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
    data/train_sub2 data/lang exp/mono_ali_sub2 exp/tri1
  touch exp/tri1/.done
fi


echo ---------------------------------------------------------------------
echo "Starting (medium) triphone training in exp/tri2 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri2/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
    data/train_sub3 data/lang exp/tri1 exp/tri1_ali_sub3
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
    data/train_sub3 data/lang exp/tri1_ali_sub3 exp/tri2
  touch exp/tri2/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (full) triphone training in exp/tri3 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri3/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesTri3 $numGaussTri3 data/train data/lang exp/tri2_ali exp/tri3
  touch exp/tri3/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri4/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri3 exp/tri3_ali
  steps/train_lda_mllt.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri3_ali exp/tri4
  touch exp/tri4/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (SAT) triphone training in exp/tri5 on" `date`
echo ---------------------------------------------------------------------

if [ ! -f exp/tri5/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri4 exp/tri4_ali
  steps/train_sat.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train data/lang exp/tri4_ali exp/tri5
  touch exp/tri5/.done
fi


################################################################################
# Ready to start SGMM training
################################################################################

if [ ! -f exp/tri5_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/tri5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri5 exp/tri5_ali
  touch exp/tri5_ali/.done
fi

if $tri5_only ; then
  echo "Exiting after stage TRI5, as requested. "
  echo "Everything went fine. Done"
  exit 0;
fi

if [ ! -f exp/ubm5/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/ubm5 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_ubm.sh \
    --cmd "$train_cmd" $numGaussUBM \
    data/train data/lang exp/tri5_ali exp/ubm5
  touch exp/ubm5/.done
fi

if [ ! -f exp/sgmm5/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_sgmm2.sh \
    --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
    data/train data/lang exp/tri5_ali exp/ubm5/final.ubm exp/sgmm5
  #steps/train_sgmm2_group.sh \
  #  --cmd "$train_cmd" "${sgmm_group_extra_opts[@]-}" $numLeavesSGMM $numGaussSGMM \
  #  data/train data/lang exp/tri5_ali exp/ubm5/final.ubm exp/sgmm5
  touch exp/sgmm5/.done
fi

if $sgmm5_only ; then
  echo "Exiting after stage SGMM5, as requested. "
  echo "Everything went fine. Done"
  exit 0;
fi
################################################################################
# Ready to start discriminative SGMM training
################################################################################

if [ ! -f exp/sgmm5_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
    --nj $train_nj --cmd "$train_cmd" --transform-dir exp/tri5_ali \
    --use-graphs true --use-gselect true \
    data/train data/lang exp/sgmm5 exp/sgmm5_ali
  touch exp/sgmm5_ali/.done
fi


if [ ! -f exp/sgmm5_denlats/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_denlats on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
    --nj $train_nj --sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
    --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" --transform-dir exp/tri5_ali \
    data/train data/lang exp/sgmm5_ali exp/sgmm5_denlats
  touch exp/sgmm5_denlats/.done
fi

if [ ! -f exp/sgmm5_mmi_b0.1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_mmi_b0.1 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mmi_sgmm2.sh \
    --cmd "$train_cmd" "${sgmm_mmi_extra_opts[@]}" \
    --drop-frames true --transform-dir exp/tri5_ali --boost 0.1 \
    data/train data/lang exp/sgmm5_ali exp/sgmm5_denlats \
    exp/sgmm5_mmi_b0.1
  touch exp/sgmm5_mmi_b0.1/.done
fi

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
