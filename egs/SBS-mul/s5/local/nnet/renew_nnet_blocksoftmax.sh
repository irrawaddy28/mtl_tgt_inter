#!/bin/bash

# Copyright 2012-2017 University of Illinois (author: Amit Das), 
# Apache 2.0

remove_last_components=2
# End of config.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

usage="Usage: $0 <block softmax dims> <input_network> <output_network>"
##e.g. renew_nnet_blocksoftmax.sh "900:1000" nnet.old nnet.new

[[ $# -eq 3 ]] || { echo $usage; exit 1; }

block_softmax_dims=$1    # csl list of block softmax dimensions
oldnn=$2  # old dnn
newnn=$3  # new dnn

[[ -e $oldnn ]] || { echo "$oldnn does not exist"; exit 1; }

echo "Replace softmax layer of network by block softmax";

last_hid_dim=`nnet-copy --binary=false --remove-last-components=${remove_last_components} ${oldnn} - |nnet-info - | grep "output-dim" | head -n 1 |awk '{print $2}'`

out_dim_array=($(echo ${block_softmax_dims}| tr ':' ' '))
out_dim=0
for d in "${out_dim_array[@]}"
do  
  out_dim=$((out_dim=out_dim+d))  
done

proto=`dirname $newnn`/blocksoftmax.proto
python utils/nnet/make_nnet_proto.py --block-softmax-dims=${block_softmax_dims} ${last_hid_dim} ${out_dim} 0 1024 >  $proto # last number in the parameter list is don't care as long as it is positive

nnet-copy --remove-last-components=${remove_last_components} ${oldnn} - | nnet-concat  -  "nnet-initialize $proto - |" $newnn || exit 1

exit 0
