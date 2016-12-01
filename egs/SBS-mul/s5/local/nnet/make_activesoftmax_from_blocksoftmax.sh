#!/bin/bash

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely), Daniel Povey
# Apache 2.0


# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 4 ]; then
   echo "Usage: $0 <nnet-in> <csl block dims> <active block number> <nnet-out>"   
   echo ""
   echo "This script takes an input blocksoftmax nnet and a block number "
   echo "for the active block and extracts a nnet corresponding to the "
   echo "active block"
   echo ""   
   exit 1;
fi

nnetin=$1
blocksoftmax_dims=$2    # 'csl' with block-softmax dimensions: dim1,dim2,dim3,...
blocksoftmax_active=$3  # block number of active block
nnetout=$4

[[ ! -e $nnetin ]] && echo "block softmax nnet does not exist" && exit 1

dir=$(dirname $nnetout)
[[ ! -d $dir ]] && mkdir -p $dir

# select a block from blocksoftmax,
# getting dims,
dim_total=$(awk -F',' '{ for(i=1;i<=NF;i++) { sum += $i }; print sum; }' <(echo $blocksoftmax_dims))
dim_block=$(awk -F',' -v active=$blocksoftmax_active '{ print $active; }' <(echo $blocksoftmax_dims))
offset=$(awk -F',' -v active=$blocksoftmax_active '{ sum=0; for(i=1;i<active;i++) { sum += $i }; print sum; }' <(echo $blocksoftmax_dims))

# create components which select a block,
nnet-initialize <(echo "<Copy> <InputDim> $dim_total <OutputDim> $dim_block <BuildVector> $((1+offset)):$((offset+dim_block)) </BuildVector>"; 
                    echo "<Softmax> <InputDim> $dim_block <OutputDim> $dim_block") $dir/copy_and_softmax.nnet 
# nnet is assembled on-the fly, <BlockSoftmax> is removed, while <Copy> + <Softmax> is added,
nnet-concat "nnet-copy --remove-last-components=1 $nnetin - |" $dir/copy_and_softmax.nnet  $nnetout
