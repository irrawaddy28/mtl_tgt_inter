#!/bin/bash
# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

#
# Extract the 1-best alignment from lattice.
# This is useful for unsupervised training of Neural networks.
#

#Begin configuration
cmd=
acwt=0.1
#End configuration

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "usage: $0 <dir-with-lats>"
   echo "e.g.:  $0 exp/decode"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>          # config containing options"
   echo "  --acwt "
   exit 1;
fi

dir=$1
model=$(dirname $dir)/final.mdl #Assuming one dir up

[[ ! -f $dir/lat.1.gz ]] && echo "Missing lattices $trans_tri" && exit 1

[ -z "$cmd" ] && echo "--cmd not set" && exit 1
nj=$(cat $dir/num_jobs)

# Generate ctms with word confidences
echo "Generating best path alignments from lat in $dir"
$cmd JOB=1:$nj $dir/log/ali_1best.JOB.log \
  lattice-best-path --acoustic-scale=$acwt \
  "ark:gunzip -c $dir/lat.JOB.gz |" ark:/dev/null ark,t:$dir/ali_1best.JOB.ark || exit 1

# Merge...
echo "Merging into $dir/ali_1best.ark.gz"
for ((n=1; n<=nj; n++)); do
  cat $dir/ali_1best.$n.ark || exit 1
done | gzip -c > $dir/ali_1best.ark.gz
# clean
rm $dir/ali_1best.*.ark

# Make the lattice-dir compatible as 'ali-dir' for 'steps/nnet/train.sh' training,
(cd $dir; ln -s ali_1best.ark.gz ali.0.gz) # symlink to ark-file,
[ -e $dir/../final.mdl ] && copy-transition-model $dir/../final.mdl $dir/final.mdl # copy the transition model,
[ -e $dir/../tree ] && cp $dir/../tree $dir/tree # copy the tied-state tree,

# Conversion on alignment, used for analysis,
ali-to-pdf $model "ark:gunzip -c $dir/ali_1best.ark.gz |" ark,t:$dir/ali_1best.ark.pdf || exit 1
ali-to-phones --per-frame=true $model "ark:gunzip -c $dir/ali_1best.ark.gz |" ark,t:$dir/ali_1best.ark.phones || exit 1
   
echo "Success..."


