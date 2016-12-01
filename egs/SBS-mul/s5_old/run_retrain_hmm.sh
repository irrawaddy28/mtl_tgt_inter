#!/bin/bash -e

# This script shows how to add a new language to a multilingual model that has not seen the language.
# In this case, we retrain a model using SW where the model was initially trained on AR+UR+DT+HG+MD.

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

# Set the language codes for SBS languages that we will be processing
export SBS_LANGUAGES="SW AR UR DT HG MD"
export TEST_LANG="SW" # this is the unseen language we want to add to our current multilingual system

# locn of "src model" where "src model" = some mono or multilingual model which has been trained on language ${TEST_LANG}
olddir=/export/ws15-pt-data2/amitdas/kaldi-trunk/egs/SBS-mul

feats_nj=4
train_nj=8
decode_nj=4

# Prepare the universal data directories under data/ALL 
utils/combine_data.sh data/ALL/train data/train data/${TEST_LANG}/train
cp -r $olddir/data/lang      data/ALL
cp -r $olddir/data/lang_test data/ALL

# Convert alignments from the "src model" to match the trans-ids of our current multilingual system
oldalidir=$olddir/exp/tri3b_ali
dir=exp/tri4b_ali; mkdir -p $dir; rm -rf $dir/*
cp -r exp/tri3b_ali/* $dir
rm -rf $dir/ali.*.gz $dir/fsts.*.gz $dir/trans* $dir/log $dir/q 
nj=1
[[ -f $dir/num_jobs ]] && nj=$(cat $dir/num_jobs)

# convert-ali old.mdl new.mdl new.tree ark:old.ali ark:new.ali
$train_cmd JOB=1 $dir/log/convert.JOB.log \
    convert-ali $oldalidir/final.alimdl $dir/final.alimdl $dir/tree "ark:gunzip -c $oldalidir/ali.*.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
# Sanity Check: convert-ali /export/ws15-pt-data2/amitdas/kaldi-trunk/egs/SBS-mul/exp/tri3b_ali/final.alimdl exp/tri4b_ali/final.alimdl exp/tri4b_ali/tree "ark:gunzip -c /export/ws15-pt-data2/amitdas/kaldi-trunk/egs/SBS-mul/exp/tri3b_ali/ali.*.gz|" ark,t:-|grep "swahili"|head -n 1|ali-to-phones  exp/tri4b_ali/final.alimdl ark,t:- ark,t:-|perl utils/int2sym.pl -f 2- data/ALL/lang/phones.txt -

# replicate the ali.1.gz file $nj times (workaround for a fix)
$train_cmd JOB=2:$nj $dir/log/copy_ali.JOB.log \
	cp $dir/ali.1.gz $dir/ali.JOB.gz || exit 1;

# Now we have alignments of the unseen language where the trans-ids of the ali
# match the trans-ids of the model. Retrain the system so that it is now trained on the unseen language
steps/train_sat.sh --cmd "$train_cmd" --train-tree "false" 0 0 \
  data/${TEST_LANG}/train data/ALL/lang exp/tri4b_ali exp/tri5b || exit 1;

# Generate alignments from the retrained system 
steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
  data/${TEST_LANG}/train data/ALL/lang exp/tri5b exp/tri5b_ali

# Build graphs and decode
for L in ${SBS_LANGUAGES}; do	
	exp_dir=exp/tri5b
	[[ -d $exp_dir ]] || continue; 
		
    graph_dir=$exp_dir/$L/graph_oracle_LG    
    [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_oracle_LG $exp_dir $graph_dir || exit 1; }
    
	steps/decode_fmllr.sh --nj ${decode_nj} --cmd "$decode_cmd" $graph_dir data/$L/dev $exp_dir/decode_dev_oracle_LG_${L} || exit 1;	
	steps/decode_fmllr.sh --nj ${decode_nj} --cmd "$decode_cmd" $graph_dir data/$L/eval $exp_dir/decode_eval_oracle_LG_${L} || exit 1;
done
wait

echo "Done retraining "
# Getting PER numbers
# for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
