#!/bin/bash -u

# After training an SAT gmm-hmm system with multilingual languages 
# (among Arabic, Dutch, Mandarin, Hungarian, Swahili, Urdu) of the SBS corpus,
# proceed to adapt gmm-hmm with probabilistic transcription of target language.

# Begin configuration section.
stage=0
train_nj=8
# End configuration section.

echo `date` && echo $0 

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

. ./path.sh || { echo "Cannot source path.sh"; exit 1; }
. parse_options.sh || exit 1;

#---------------------------------------------------------------------------
# Set the location of the SBS speech 
SBS_CORPUS=${SBS_DATADIR}/audio
SBS_TRANSCRIPTS=${SBS_DATADIR}/transcripts/matched
SBS_DATA_LISTS=${SBS_DATADIR}/lists

# Set the language codes for SBS languages that we will be processing
export TRAIN_LANG=$1
export TEST_LANG=$2
export dir_raw_pt=$3 # Raw (not scored by language model) and unpruned probabilistic (crowdsourced) lattice ("P") per utterance
export UNILANG_CODE=$(echo $TRAIN_LANG |sed 's/ /_/g')


#---------------------------------------------------------------------------
# generate alignment for training data if needed, e.g., to train a dnn
if [ $stage -le -1 ]; then
  mkdir -p exp/tri3b_ali
	steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
		data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang exp/tri3b/${TEST_LANG} exp/tri3b_ali/${TEST_LANG} || exit 1;
  echo ------------------------------------------
fi
#---------------------------------------------------------------------------
# Post-process the raw probabilistic lattice/sausages, i.e., pt, 
# by composing G and pt.
# On G, add self-loop <#2>:<#2>; on pt, deletion arc is <#2>:<eps>.
# Maybe need to make corresponding change to this stage to process 
# different raw pt lattices. Here is an eg.

if [ -z $dir_raw_pt ]; then echo "empty dir_raw_pt" && exit 1;fi

dir_fsts=exp/data_pt-2/${TEST_LANG} # Dir to store G.Pp (G composed with pruned P) lattice for every training utterance of ${TEST_LANG}
dir_lang=data/${TEST_LANG}/lang_test_text_G
if [ $stage -le 0 ]; then
  # first, generate G_new.fst
  # assume $TEST_LANG is the only one held-out language
  mkdir -p $dir_fsts
  cp data/${TEST_LANG}/local/lm/lm_phone.arpa.gz $dir_fsts/lm_phone.arpa.gz || exit 1;

  gunzip -c $dir_fsts/lm_phone.arpa.gz | egrep -v '<s> <s>|</s> <s>|</s> </s>' | \
    arpa2fst - | fstprint | utils/eps2disambig.pl | utils/s2eps.pl \
    > $dir_fsts/lm_phone.txt || exit 1;

  s_bkoff=`cat $dir_fsts/lm_phone.txt | grep "#0" | awk '{print $2}' | uniq` || exit 1;
  # should be only one backoff state; if not, exit and diagnose
  num=`echo $s_bkoff | wc -w` || exit 1;
  if [ $num -gt 1 ]; then echo "expect only one backoff state" && exit 1; fi
  echo "backoff state $s_bkoff"

  # Check $dir_lang/words.txt does not already have two disambig symbols. 
  disambig_sym_orig=`tail -n1 data/${UNILANG_CODE}/lang_test/words.txt | awk '{print $2}'` || exit 1;
  disambig_sym=`tail -n1 $dir_lang/words.txt | awk '{print $2}'` || exit 1;
  [[ $disambig_sym_orig -ne $disambig_sym ]] && { echo "disambig symbol original ($disambig_sym_orig) does not match disambig_sym ($disambig_sym)"; exit 1; }
  # Since $dir_lang/words.txt does not already have two disambig symbols, add them now.
  disambig_sym1=$((disambig_sym+1))
  disambig_sym2=$((disambig_sym1+1))
  echo "new disambig_syms on G_new.fst " $disambig_sym1 $disambig_sym2

  # cat $dir_fsts/lm_phone.txt | awk -v disambig=$disambig_sym -v s_bkoff=$s_bkoff 'BEGIN{pre=0;}
  # {if ($1 != pre && pre != s_bkoff && pre != 0) {print pre"\t"pre"\t"disambig"\t"disambig;}
  # print $0; pre=$1;} END{print pre"\t"pre"\t"disambig"\t"disambig}' \
  # > $dir_fsts/G_new_fst || exit 1;
  cat $dir_fsts/lm_phone.txt | awk -v disambig1=$disambig_sym1 -v disambig2=$disambig_sym2 -v s_bkoff=$s_bkoff 'BEGIN{pre=0;}
  {if ($1 != pre && pre != s_bkoff && pre != 0) {print pre"\t"pre"\t"disambig1"\t"disambig1;print pre"\t"pre"\t"disambig2"\t"disambig2;}
  print $0; pre=$1;} END{print pre"\t"pre"\t"disambig1"\t"disambig1;print pre"\t"pre"\t"disambig2"\t"disambig2;}' \
  > $dir_fsts/G_new_fst || exit 1;

  #echo "$disambig_sym $disambig_sym" | cat - $dir_lang/words.txt \
  #   > $dir_fsts/words.txt
  ( cat $dir_lang/words.txt; echo "$disambig_sym1 $disambig_sym1"; echo "$disambig_sym2 $disambig_sym2"; ) \
    > $dir_fsts/words.txt
  cp $dir_fsts/words.txt $dir_lang/  

  cat $dir_fsts/G_new_fst | fstcompile --isymbols=$dir_fsts/words.txt \
    --osymbols=$dir_fsts/words.txt --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $dir_fsts/G_new.fst || exit 1;
  echo ------------------------------------------

  # second, compose G_new.fst with pruned pt lattices, and prune
  for f in $dir_raw_pt/*.TPLM.fst; do
  # fstprune --weight=2.0 $f | fstprint | awk -v disambig=$disambig_sym \
  # '{if (NF > 3 && $3 == 0) $3 = disambig; print}' | fstcompile | fstarcsort  > $dir_fsts/tmp.fst
    echo "Composing G_new.fst with $f"
    if [[ "${TEST_LANG}" == "CA" || "${TEST_LANG}" == "AM" || "${TEST_LANG}" == "DI" ]]; then
       fstarcsort $f > $dir_fsts/tmp.fst
       fstcompose $dir_fsts/G_new.fst $dir_fsts/tmp.fst | fstarcsort | fstprune --weight=1.0 > $dir_fsts/${f##/*/}
    else
       fstprune --weight=2.0 $f | fstarcsort  > $dir_fsts/tmp.fst
       fstcompose $dir_fsts/G_new.fst $dir_fsts/tmp.fst | fstarcsort | fstprune --weight=1.0 > $dir_fsts/${f##/*/}
    fi
    rm $dir_fsts/tmp.fst 2>/dev/null
    #break
  done
  echo ------------------------------------------
  # third, generate disambig_new.int and L_disambig_new.fst
  #  disambig_sym_=`tail -n1 $dir_lang/phones/disambig.int`
  #  disambig_sym_=$((disambig_sym_+1))
  #  echo "new disambig_sym on L_disambig_new.fst " $disambig_sym_ || exit 0;
  # echo $disambig_sym_ | cat $dir_lang/phones/disambig.int - \
  #  > $dir_lang/phones/disambig_new.int || exit 0;
  disambig_sym_=`tail -n1 $dir_lang/phones/disambig.int`
  disambig_sym1_=$((disambig_sym_+1))
  disambig_sym2_=$((disambig_sym1_+1))
  echo "new disambig_syms on L_disambig_new.fst " $disambig_sym1_ $disambig_sym2_  || exit 0;
  ( cat $dir_lang/phones/disambig.int; echo $disambig_sym1_; echo $disambig_sym2_; ) \
    > $dir_lang/phones/disambig_new.int;

  ##
  sym_l=`cat $dir_lang/phones.txt | grep "#0" | awk '{print $2}'`
  sym_g=`cat $dir_lang/words.txt | grep "#0" | awk '{print $2}'`
  echo "#0 in L and G " $sym_l $sym_g
  #line=`fstprint $dir_lang/L_disambig.fst | grep -P "${sym_l}\t${sym_g}" | \
  #  awk -v s_l=$disambig_sym_ -v s_g=$disambig_sym '{$3=s_l; $4=s_g; print}'`
  line=`fstprint $dir_lang/L_disambig.fst | grep -P "${sym_l}\t${sym_g}" | \
    awk -v s_l=$disambig_sym1_ -v s_g=$disambig_sym1 -v e_l=$disambig_sym2_ -v e_g=$disambig_sym2 '{$3=s_l; $4=s_g; print; $3=e_l; $4=e_g; print;}'`  
  fstprint $dir_lang/L_disambig.fst > $dir_fsts/L_disambig.txt
  echo "$line" | cat $dir_fsts/L_disambig.txt - | fstcompile > $dir_lang/L_disambig_new.fst || exit 1;

fi
#echo `date` && exit 0
#---------------------------------------------------------------------------
feats_nj=4
train_nj=8
decode_nj=4

# Adapting SAT+LDA+MLLT triphone systems to the $TEST_LANG
  L=$TEST_LANG
  alidir_g_pt=exp/tri3bpt_ali/${L}
  if [ $stage -le 1 ]; then
    # Align fmLLR features of the test language using the tri3b model and pt lattice

    # $dir_fsts needs to be properly assigned and contains the processed pt
    if [ -z $dir_fsts ]; then echo "empty $dir_fsts" && exit 1;fi

    echo "Aligning PTs"

    mkdir -p $alidir_g_pt
    # Using the SAT model (tri3b), decode the training data of the test language 
    # using the HCLGP graph. Here, H = tri3b HMM, P is the crowdsourced confusion net specific
    # to the test language. The decoder generates a lattice (instead of an alignment)
    # This lattice will be used as the training "transcripts". The FMLLR xforms are limited to 
    # training speakers of $TEST_LANG.
    local/align_fmllr_g_pt.sh --nj "$train_nj" --cmd "$train_cmd" data/$L/train \
      $dir_lang exp/tri3b/$L $alidir_g_pt $dir_fsts || exit 1;
    echo ------------------------------------------
  fi

  num_iters=12
  #exp_dir=exp/tri3b_map_${L}_g_pt_text_G_it${num_iters}
  exp_dir=exp/tri3c/${L}
  if [ $stage -le 2 ]; then  
    echo "Adapting to PTs"
    
    # Now adapt the SAT model using the decoded lattice (training "transcripts")
    if [[ "${TEST_LANG}" == "AM" ]]; then
	  local/train_sat_map_truncated_pt.sh --cmd "$train_cmd" --num-iters ${num_iters} \
		data/$L/train $dir_lang $alidir_g_pt $exp_dir || exit 1;
	else
	  local/train_sat_map_pt.sh --cmd "$train_cmd" --num-iters ${num_iters} \
		data/$L/train $dir_lang $alidir_g_pt $exp_dir || exit 1;
	fi
    echo ------------------------------------------
  fi

  if [ $stage -le 3 ]; then
    # Decode dev and test data of all languages  
    for L in ${TEST_LANG} ${TRAIN_LANG}; do
    #for L in ${TEST_LANG}; do
      echo "Decoding $L"
      
      graph_dir=$exp_dir/graph_text_G_$L
      mkdir -p $graph_dir
      utils/mkgraph.sh data/${TEST_LANG}/lang_test_text_G $exp_dir $graph_dir
      
      (steps/decode_fmllr.sh --num-threads 6 --nj "$decode_nj" --cmd "$decode_cmd" $graph_dir \
        data/$L/dev ${exp_dir}/decode_dev_text_G_$L || exit 1;) &
      (steps/decode_fmllr.sh --num-threads 6 --nj "$decode_nj" --cmd "$decode_cmd" $graph_dir \
        data/$L/eval ${exp_dir}/decode_eval_text_G_$L || exit 1;) &
      wait
            
      (cd $exp_dir; ln -s  decode_dev_text_G_$L decode_dev_$L; ln -s decode_eval_text_G_$L decode_eval_$L)
    done
  fi
  
  if [ $stage -le 4 ]; then
    # Align fMLLR features of the training languages based on PT adapted model
    echo "Aligning features of ${UNILANG_CODE} with DTs using PT adapted model $exp_dir/final.mdl"    
    steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
	  data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang $exp_dir exp/tri3c_ali/${TEST_LANG} || exit 1;
  fi
  
  alidir_g_pt=exp/tri3cpt_ali/${TEST_LANG}
  if [ $stage -le 5 ]; then
    # Align fMLLR features of the test language based on PT adapted model
    echo "Aligning features of ${TEST_LANG} with PTs using PT adapted model $exp_dir/final.mdl"
    if [ -z $dir_fsts ]; then echo "empty $dir_fsts" && exit 1;fi
        
    # Using the PT adapted SAT model (tri3c), decode the training data of the test language 
    # using the HCLGP graph. Here, H = tri3c, P is the crowdsourced confusion net specific
    # to the test language. The decoder generates a lattice (instead of an alignment)
    # This lattice will be used as the training "transcripts". New FMLLR xforms are
    # re-estimated using the alignments generated from the PT adapted SAT model. 
    # The FMLLR xforms are limited to training speakers of $TEST_LANG.
    # Note: These FMLLR xforms are different (& better) from PT unadapted SAT
    # models (i.e., tri3bpt_ali).
    local/align_fmllr_g_pt.sh --nj "$train_nj" --cmd "$train_cmd" data/${TEST_LANG}/train \
      $dir_lang exp/tri3c/${TEST_LANG} $alidir_g_pt $dir_fsts || exit 1;
    echo ------------------------------------------
  fi
    
  echo "Done: `date`"


#---------------------------------------------------------------------------
# Getting PER numbers
# for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
