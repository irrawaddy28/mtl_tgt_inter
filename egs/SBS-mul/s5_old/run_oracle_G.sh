#!/bin/bash

#
# This is supposed to be run after run.sh and run_dnn.sh
#

set -e

stage=0
nnet_dir=
data_fmllr=data-fmllr-tri3b

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh || exit 1;

SBS_LANGUAGES="SW MD AR DT HG UR"

if [ $stage -le 0 ]; then
for L in $SBS_LANGUAGES; do
  echo "Prep oracle G for $L"
  local/sbs_format_oracle_G.sh $L >& data/$L/format_oracle_G.log
done
fi

## Decode with oracle G: mono
#if [ $stage -le 1 ]; then
#for L in $SBS_LANGUAGES; do
  #graph_dir=exp/mono/$L/graph_oracle_G
  #mkdir -p $graph_dir
  #utils/mkgraph.sh --mono data/$L/lang_test_oracle_G exp/mono \
    #$graph_dir >& $graph_dir/mkgraph.log

  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    #exp/mono/decode_eval_oracle_G_$L &
#done
#wait
#fi

## Decode with oracle G: tri1
#if [ $stage -le 2 ]; then
#for L in $SBS_LANGUAGES; do
  #graph_dir=exp/tri1/$L/graph_oracle_G
  #mkdir -p $graph_dir
  #utils/mkgraph.sh data/$L/lang_test_oracle_G exp/tri1 \
    #$graph_dir >& $graph_dir/mkgraph.log

  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    #exp/tri1/decode_eval_oracle_G_$L &
#done
#wait
#fi

### Decode with oracle G
##if [ $stage -le 3 ]; then
##for L in $SBS_LANGUAGES; do
  ##graph_dir=exp/tri2a/$L/graph_oracle_G
  ##mkdir -p $graph_dir
  ##utils/mkgraph.sh data/$L/lang_test_oracle_G exp/tri2a \
    ##$graph_dir >& $graph_dir/mkgraph.log

  ##steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    ##exp/tri2a/decode_eval_oracle_G_$L &
##done
##wait
#fi

## Decode with oracle G: tri2b
#if [ $stage -le 3 ]; then
#for L in $SBS_LANGUAGES; do
  #graph_dir=exp/tri2b/$L/graph_oracle_G
  #mkdir -p $graph_dir
  #utils/mkgraph.sh data/$L/lang_test_oracle_G exp/tri2b \
    #$graph_dir >& $graph_dir/mkgraph.log

  #steps/decode.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    #exp/tri2b/decode_eval_oracle_G_$L &
#done
#wait
#fi

# Decode with oracle G: tri3b
if [ $stage -le 4 ]; then
for L in $SBS_LANGUAGES; do
  graph_dir=exp/tri3b/$L/graph_oracle_G
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test_oracle_G exp/tri3b \
    $graph_dir >& $graph_dir/mkgraph.log

  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/dev \
    exp/tri3b/decode_dev_oracle_G_$L &
      
  steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $graph_dir data/$L/eval \
    exp/tri3b/decode_eval_oracle_G_$L &
done
wait
fi

# Nnet decode with oracle G: nnet
if [ $stage -le 5 ]; then

[[ -d ${nnet_dir} ]] && [[ -d ${data_fmllr} ]] || { echo "${nnet_dir} and/or ${data_fmllr} do not exist"; exit 1; }

for L in $SBS_LANGUAGES; do
  graph_dir=exp/tri3b/$L/graph_oracle_G
  mkdir -p $graph_dir
  utils/mkgraph.sh data/$L/lang_test_oracle_G exp/tri3b \
    $graph_dir >& $graph_dir/mkgraph.log

steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $graph_dir $data_fmllr/$L/dev ${nnet_dir}/decode_dev_oracle_G_$L &

steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $graph_dir $data_fmllr/$L/eval ${nnet_dir}/decode_eval_oracle_G_$L &
done
wait
fi
