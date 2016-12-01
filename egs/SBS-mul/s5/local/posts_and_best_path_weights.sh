#!/bin/bash


# Frame weighting options
threshold=0.7   # If provided, use frame thresholding -- keep only frames whose
                # best path posterior is above this value.  
use_soft_counts=true    # Use soft-posteriors as targets for PT data
disable_upper_cap=true
acwt=0.2
parallel_opts="--num-threads 6"
# End of config.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 4 ]; then
   echo "Usage: $0 [options] <gmmdir> <train lat dir> <best path weights o/p dir> <posterior o/p dir>"   
   echo ""
   echo " "
   echo " "
   echo " "
   echo ""   
   exit 1;
fi

gmmdir=$1         # i/p dir where final.mdl is present
decode_dir=$2     # i/p dir where the training lattices (lat.*.gz) are present. Usually in $gmmdir/decode_* 
best_path_dir=$3  # o/p dir to save best path weights
postdir=$4        # o/p dir to save the posteriors

required="$gmmdir/final.mdl $decode_dir/lat.1.gz"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

# Get frame weights as the posteriors of the best path in the lattice
 local/best_path_weights.sh --acwt $acwt  $decode_dir $best_path_dir || exit 1   
	 
# Get frame posteriors from lattice. Multiple posteriors per frame is possible.
 nj=$(cat $best_path_dir/num_jobs)
 if ! $use_soft_counts; then
   # Get 1-hot posteriors from best path of the lattice of target language
   $train_cmd JOB=1:$nj $postdir/get_hard_post.JOB.log \
     ali-to-pdf $gmmdir/final.mdl "ark:gunzip -c $best_path_dir/ali.JOB.gz |" ark:- \| \
     ali-to-post ark:- ark,scp:$postdir/post.JOB.ark,$postdir/post.JOB.scp || exit 1
 else 
   # Get soft posteriors from the lattice of target language
   $train_cmd JOB=1:$nj $postdir/get_soft_post.JOB.log \
     lattice-to-post --acoustic-scale=$acwt "ark:gunzip -c $decode_dir/lat.JOB.gz |" ark:- \| \
     post-to-pdf-post $gmmdir/final.mdl ark:- \
     ark,t,scp:$postdir/post.JOB.ark,$postdir/post.JOB.scp || exit 1
 fi
 
 for n in `seq $nj`; do
   cat $postdir/post.$n.scp
 done > $postdir/post.scp
  
# Apply binary threshold on the frame weights. For frame weight above threshold,
# set the frame weight to either a) 1.0 (by setting --upper-cap=1.0) or 
# b) the posterior value itself (by setting --disable-upper-cap=true). 
# Frame weight below threshold is set to 0.
 copy_command=copy-vector
 if [ ! -z "$threshold" ]; then
   if $disable_upper_cap ; then		
	 copy_command="thresh-vector --threshold=$threshold --lower-cap=0.0  --disable-upper-cap=true"
   else
	 copy_command="thresh-vector --threshold=$threshold --lower-cap=0.0 --upper-cap=1.0"
   fi
 fi
  
 $train_cmd JOB=1:$nj $postdir/copy_frame_weights.JOB.log \
   $copy_command "ark:gunzip -c $best_path_dir/weights.JOB.gz |" \
   ark,t,scp:$postdir/frame_weights.JOB.ark,$postdir/frame_weights.JOB.scp || exit 1
  
 for n in `seq $nj`; do
   cat $postdir/frame_weights.$n.scp
 done > $postdir/frame_weights.scp
 
echo "Saved scp posteriors in $postdir/post.scp"
echo "Saved scp best path weights in $postdir/frame_weights.scp"

exit 0;
