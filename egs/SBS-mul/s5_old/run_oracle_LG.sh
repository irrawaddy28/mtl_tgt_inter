#!/bin/bash

#
# This is supposed to be run after run.sh and run_dnn.sh
#

set -e

stage=6
nnet_dir=
data_fmllr=data-fmllr-tri3b

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh || exit 1;


SBS_LANGUAGES="SW MD AR DT HG UR"

if [ $stage -le 0 ]; then
for L in $SBS_LANGUAGES; do
  echo "Prep oracle G for $L"
  local/sbs_format_oracle_LG.sh $L >& data/$L/format_oracle_LG.log
done
fi

## Decode with oracle G: mono
#if [ $stage -le 1 ]; then
#for L in $SBS_LANGUAGES; do
  #graph_dir=exp/mono/$L/graph_oracle_LG
  #mkdir -p $graph_dir
  #utils/mkgraph.sh --mono data/$L/lang_test_oracle_LG exp/mono \
    #$graph_dir >& $graph_dir/mkgraph.log

  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    #exp/mono/decode_eval_oracle_LG_$L &
#done
#wait
#fi

## Decode with oracle G: tri1 
#if [ $stage -le 2 ]; then
#for L in $SBS_LANGUAGES; do
  #graph_dir=exp/tri1/$L/graph_oracle_LG
  #mkdir -p $graph_dir
  #utils/mkgraph.sh data/$L/lang_test_oracle_LG exp/tri1 \
    #$graph_dir >& $graph_dir/mkgraph.log

  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    #exp/tri1/decode_eval_oracle_LG_$L &
#done
#wait
#fi

### Decode with oracle G: tri2a
##if [ $stage -le 3 ]; then
##for L in $SBS_LANGUAGES; do
  ##graph_dir=exp/tri2a/$L/graph_oracle_LG
  ##mkdir -p $graph_dir
  ##utils/mkgraph.sh data/$L/lang_test_oracle_LG exp/tri2a \
    ##$graph_dir >& $graph_dir/mkgraph.log

  ##steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    ##exp/tri2a/decode_eval_oracle_LG_$L &
##done
##wait
##fi

## Decode with oracle G: tri2b
#if [ $stage -le 3 ]; then
#for L in $SBS_LANGUAGES; do
  #graph_dir=exp/tri2b/$L/graph_oracle_LG
  #mkdir -p $graph_dir
  #utils/mkgraph.sh data/$L/lang_test_oracle_LG exp/tri2b \
    #$graph_dir >& $graph_dir/mkgraph.log

  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    #exp/tri2b/decode_eval_oracle_LG_$L &
#done
#wait
#fi

# Decode with oracle G: tri3b
if [ $stage -le 4 ]; then
for L in $SBS_LANGUAGES; do
  exp_dir=exp/tri3b
  [[ -d $exp_dir ]] || continue; 
  
  graph_dir=$exp_dir/$L/graph_oracle_LG
  mkdir -p $graph_dir
  
  utils/mkgraph.sh data/$L/lang_test_oracle_LG $exp_dir \
    $graph_dir >& $graph_dir/mkgraph.log

  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    $exp_dir/decode_dev_oracle_LG_$L &
    
  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    $exp_dir/decode_eval_oracle_LG_$L &
done
wait
fi

# Decode with oracle G: tri3b pt map
if [ $stage -le 5 ]; then
for L in $SBS_LANGUAGES; do	
	exp_dir=exp/tri3b_map_${L}_pt
	[[ -d $exp_dir ]] || continue; 
	
    graph_dir=$exp_dir/$L/graph_oracle_LG    
    [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_oracle_LG $exp_dir $graph_dir || exit 1; }    
    
steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    $exp_dir/decode_dev_oracle_LG_${L} || exit 1;

steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    $exp_dir/decode_eval_oracle_LG_${L} || exit 1;
done
wait
fi

# Nnet decode with oracle G: nnet + tri3b
if [ $stage -le 6 ]; then
[[ -d ${nnet_dir} ]] && [[ -d ${data_fmllr} ]] || { echo "${nnet_dir} and/or ${data_fmllr} do not exist"; exit 1; }
for L in $SBS_LANGUAGES; do
  exp_dir=exp/tri3b
  [[ -d $exp_dir ]] || continue; 
    
  graph_dir=$exp_dir/$L/graph_oracle_LG
  [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_oracle_LG $exp_dir $graph_dir || exit 1; }

steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $graph_dir $data_fmllr/$L/dev ${nnet_dir}/decode_dev_oracle_LG_$L &

steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $graph_dir $data_fmllr/$L/eval ${nnet_dir}/decode_eval_oracle_LG_$L &
done
wait
fi


# Nnet decode with oracle G: nnet + tri3b pt map
if [ $stage -le 7 ]; then
[[ -d ${nnet_dir} ]] && [[ -d ${data_fmllr} ]] || { echo "${nnet_dir} and/or ${data_fmllr} do not exist"; exit 1; }
for L in $SBS_LANGUAGES; do
  exp_dir=exp/tri3b_map_${L}_pt
  [[ -d $exp_dir ]] || continue; 
    
  graph_dir=$exp_dir/$L/graph_oracle_LG
  [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_oracle_LG $exp_dir $graph_dir || exit 1; }

steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $graph_dir $data_fmllr/dev ${nnet_dir}/decode_dev_oracle_LG_$L &

steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $graph_dir $data_fmllr/eval ${nnet_dir}/decode_eval_oracle_LG_$L &
done
wait
fi
