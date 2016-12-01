#!/bin/bash
# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

#
# Make per-frame confidence measure based on lattices.
# Extracts the pdf-posteriors under the 1-best path.
#

#Begin configuration
cmd=run.pl
acwt=0.002
lmwt=0.02
#End configuration

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "usage: $0 <dir-with-lats> <out-dir>"
   echo "e.g.:  $0 exp/tri5a/decode_test"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>          # config containing options"
   echo "  --cmd"
   echo "  --acwt"
   echo "  --lmwt"
   exit 1;
fi

latdir=$1
dir=$2

model=$(dirname $latdir)/final.mdl
[ ! -f $model ] && echo "Missing $model" && exit 1
[ ! -f $latdir/lat.1.gz ] && echo "Missing $latdir/lat.1.gz" && exit 1

nj=$(cat $latdir/num_jobs)

# Get the posterior from the pdf alignments from the lattice
pdfpost="ark:lattice-to-post --acoustic-scale=$acwt --lm-scale=$lmwt \"ark:gunzip -c $latdir/lat.JOB.gz |\" ark:- | post-to-pdf-post $model ark:- ark:- |"

# Select get the per-frame confidence vectors
$cmd JOB=1:$nj $dir/log/frame-confidence.JOB.log \
  lattice-best-path --acoustic-scale=$acwt --lm-scale=$lmwt \
    "ark:gunzip -c $latdir/lat.JOB.gz |" ark:/dev/null ark:- \| \
  ali-to-pdf $model ark:- ark:- \| \
  get-post-on-ali "$pdfpost" ark:- ark,t:$dir/confidence_frame.JOB.ark || exit 1

# Merge...
echo "Merging into $dir/confidence_frame.ark"
for ((n=1; n<=nj; n++)); do
  cat $dir/confidence_frame.$n.ark || exit 1
done > $dir/confidence_frame.ark
# clean
rm $dir/confidence_frame.*.ark
   
echo "Success..."


