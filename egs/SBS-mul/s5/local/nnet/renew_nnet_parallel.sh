#!/bin/bash

# Copyright 2012-2017 University of Illinois (author: Amit Das), 
# Apache 2.0

remove_last_components=2
parallel_softmax_opts=
parallel_nhl_opts=
parallel_nhn_opts=
parallel_nhl=0  # no. of hidden layers in the tasks, Default 0 hidden layers for all tasks
parallel_nhn=1024  # no. of hidden neurons in the tasks, Default 1024 neurons for all tasks
# End of config.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. utils/parse_options.sh || exit 1;

usage="Usage: $0 <csl parallel odims> <input network> <output_network>"
# E.g.(a) To create a parallel network consisting of a 3 tasks where task1 and task2 use softmax layers
# and task3 has a single affine xform (no softmax), use
# renew_nnet_parallel.sh --parallel-softmax-opts "::--no-softmax"  500:500:1000 nnet.old nnet.new
# (b) To create 3 hidden layers for each task with hidden neurons 1024, 1024, 2048 resp., use
# renew_nnet_parallel.sh --parallel-softmax-opts "::--no-softmax" --parallel-nhl-opts "3:3:3" --parallel-nhn-opts "1024:1024:2048" 500:500:1000 nnet.old nnet.new

[[ $# -eq 3 ]] || { echo $usage; exit 1; }

#parallel_protos=$1   # csl list of proto names of the parallel network
parallel_odims=$1     # csl list of o/p dims of the parallel tasks
oldnn=$2  # old dnn
newnn=$3  # new dnn

[[ -e $oldnn ]] || { echo "$oldnn does not exist"; exit 1; }

echo "Replace softmax layer of the i/p network by parallel network";

# Get the dimension of the final hidden layer located just before the parallel network
last_hid_dim=`nnet-copy --binary=false --remove-last-components=${remove_last_components} ${oldnn} - |nnet-info - | grep "output-dim" | head -n 1 |awk '{print $2}'`
echo "$oldnn: Final hidden layer has $last_hid_dim neurons"

# Get the names of the proto files and output dimension of each subnet in the parallel network
#proto_array=($(echo ${parallel_protos}| tr ':' ' '))
out_dim_array=($(echo ${parallel_odims}| tr ':' ' '))
out_dim=0
for d in "${out_dim_array[@]}"; do
  out_dim=$((out_dim=out_dim+d))   # o/p dim of parallel n/w = sum of o/p dims of individual subnets
done

dir=$(dirname $newnn)
[[ ! -d $dir ]] && mkdir -p $dir

# Create the proto files for each task
py_opt=
ntasks=${#out_dim_array[@]}
OIFS=$IFS; IFS=":"
softmax_opts_array=($parallel_softmax_opts)
nhl_opts_array=($parallel_nhl_opts)
nhn_opts_array=($parallel_nhn_opts)
IFS=$OIFS
for t in `seq 0 $(($ntasks-1))`; do
  proto=$dir/task$((t+1)).proto
  softmax_opt=$(echo ${softmax_opts_array[t]}| tr ':' ' ')
  nhl_opt=$(echo ${nhl_opts_array[t]}| tr ':' ' ')
  [ ! -z $nhl_opt ] && parallel_nhl=$nhl_opt
  nhn_opt=$(echo ${nhn_opts_array[t]}| tr ':' ' ')
  [ ! -z $nhn_opt ] && parallel_nhn=$nhn_opt
  python utils/nnet/make_nnet_proto.py ${softmax_opt} ${last_hid_dim} ${out_dim_array[t]} ${parallel_nhl} ${parallel_nhn} > ${proto}
  py_opt="${py_opt}${proto},${out_dim_array[t]}:"
done

# Create the parallel n/w from the individual proto files
py_opt=$(echo ${py_opt}|sed s'/.$//') # remove the colon at the end of string
parallel_proto=$dir/parallel.proto
python utils/nnet/make_nnet_proto.py --no-softmax --parallel-net ${py_opt}  $last_hid_dim $out_dim 0 1 > $parallel_proto # last number in the parameter list is don't care as long as it is positive

# Finally, initialize the parallel n/w and add it on top of the final hidden layer of the i/p nnet
nnet-copy --remove-last-components=${remove_last_components} ${oldnn} - | nnet-concat  -  "nnet-initialize $parallel_proto - |" $newnn || exit 1

exit 0
