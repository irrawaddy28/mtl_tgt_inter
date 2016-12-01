#!/bin/bash -u

# After training an SAT gmm-hmm system with multilingual languages 
# (among Arabic, Dutch, Mandarin, Hungarian, Swahili, Urdu) of the SBS corpus,
# proceed to adapt gmm-hmm with probabilistic transcription of target language.

echo `date` && echo $0 

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

. ./path.sh || { echo "Cannot source path.sh"; exit 1; }

stage=-1
# Set the location of the SBS speech 
SBS_CORPUS=/export/ws15-pt-data/data/audio
SBS_TRANSCRIPTS=/export/ws15-pt-data/data/transcripts/matched
SBS_DATA_LISTS=/export/ws15-pt-data/data/lists
# Set the language codes for SBS languages that we will be processing
#export SBS_LANGUAGES="AR DT MD HG SW UR"
#export TRAIN_LANG="SW AR UR DT HG"
export TEST_LANG="SW"

feats_nj=4
train_nj=8
decode_nj=4

#---------------------------------------------------------------------------
# Post-process the raw probabilistic lattice/sausages, i.e., pt, and the main
# change is to map <eps>:<eps> to <#0>:<eps>.
# Probably need to make corresponding change to this stage to process 
# different raw pt lattices. Here is an eg.

for L in $TEST_LANG; do
# add the directory of raw pt
if [[ $L == "SW" ]]; then
	dir_raw_pt=/export/ws15-pt-data2/data/pt-stable-7-unigram/held-out-SW #/export/ws15-pt-data/data/phonelattices/monophones/trainedp2let/HG_MD_UR_DT_AR_SWdecode  # train on HG,MD,UR,DT,AR and adapt on SW
	prune_wt=1 # higher values means less pruning
else
	dir_raw_pt=
fi

dir_fsts=data_${L}_pt
if [ $stage -le -1 ]; then
  mkdir -p $dir_fsts
  disambig_sym=`grep "#0" data/lang/words.txt | awk '{print $2}'`
  #for f in $dir/*saus.fst; do
  for f in $dir_raw_pt/*lat.fst; do
    fstprint $f |  awk -v sym=$disambig_sym '{if (NF > 3 && $3 == 0) {$3 = sym}; print}' | \
      fstcompile | fstprune --weight=$prune_wt > $dir_fsts/${f##/*/}
  done
  echo ------------------------------------------
fi
done
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
## generate alignment for training data if needed
#if [ $stage -le 0 ]; then
  #mkdir -p exp/tri3b_ali
	#steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
		#data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;
  #echo ------------------------------------------
#fi

#---------------------------------------------------------------------------
# Adapting SAT+LDA+MLLT triphone systems
for L in $TEST_LANG; do
  if [ $stage -le 1 ]; then   
    # align pt of target language
    #
    # $dir_fsts needs to be properly assigned and contains the processed pt
    #dir_fsts=   # comment this line if it has been assigned
    if [ -z $dir_fsts ]; then echo "empty $dir_fsts" && exit 1;fi

    mkdir -p exp/tri3b_ali_${L}_pt
    local/align_fmllr_pt.sh --nj "$train_nj" --cmd "$big_memory_cmd" data/$L/train \
      data/lang exp/tri3b exp/tri3b_ali_${L}_pt $dir_fsts || exit 1;
    echo ------------------------------------------
  fi

  exp_dir=exp/tri3b_map_${L}_pt
  if [ $stage -le 2 ]; then
    local/train_sat_map_pt.sh --cmd "$train_cmd" \
      data/$L/train data/lang exp/tri3b_ali_${L}_pt $exp_dir || exit 1;
    echo ------------------------------------------
  fi
  
  if [ $stage -le 3 ]; then
	exp_dir=exp/tri3b_map_${L}_pt
	[[ -d $exp_dir ]] || continue; 
	
    graph_dir=$exp_dir/$L/graph_oracle_LG    
    [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_oracle_LG $exp_dir $graph_dir || exit 1; }    
    
	steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
		$exp_dir/decode_dev_oracle_LG_${L}  &

	steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
		$exp_dir/decode_eval_oracle_LG_${L} &    
  fi
done
wait
echo "Done"


#---------------------------------------------------------------------------
# Getting PER numbers
# for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done


