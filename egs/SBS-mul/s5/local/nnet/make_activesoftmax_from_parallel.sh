#!/bin/bash

# Copyright 2012-2017 University of Illinois (author: Amit Das), 
# Apache 2.0

remove_last_components=2
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 3 ]; then
   echo "Usage: $0 <nnet-in> <task id> <nnet-out>"
   echo ""
   echo "This script takes an input nnet with parallel networks and a task id "
   echo "and extracts a nnet with the desired task while discarding other tasks in the parallel networks"
   echo ""
   exit 1;
fi

nnetin=$1       # in nnet
active_block=$2 # the id of the desired task
nnetout=$3      # out nnet 

[[ ! -e $nnetin ]] && echo "nnet does not exist" && exit 1

dir=$(dirname $nnetin)

decode_dir=$(dirname $nnetout)
[[ ! -d $decode_dir ]] && mkdir -p $decode_dir

nnet-concat "nnet-copy --remove-last-components=$remove_last_components $dir/final.nnet - |" \
            "nnet-copy --from-parallel-component=$active_block $dir/final.nnet - |" \
            $decode_dir/final.nnet
