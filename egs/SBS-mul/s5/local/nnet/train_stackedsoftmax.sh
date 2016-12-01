#!/bin/bash

# Copyright 2012/2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Begin configuration.
config=            # config, which is also sent to all other scripts

# NETWORK INITIALIZATION
nnet_init=          # select initialized MLP (override initialization)
nnet_proto=         # select network prototype (initialize it)
proto_opts=        # non-default options for 'make_nnet_proto.py'
feature_transform= # provide feature transform (=splice,rescaling,...) (don't build new one)
pytel_transform=   # use external transform defined in python (BUT specific)
network_type=dnn   # (dnn,cnn1d,cnn2d,lstm) select type of neural network
cnn_proto_opts=     # extra options for 'make_cnn_proto.py'
#
hid_layers=4       # nr. of hidden layers (prior to sotfmax or bottleneck)
hid_dim=1024       # select hidden dimension
bn_dim=            # set a value to get a bottleneck network
dbn=               # select DBN to prepend to the MLP initialization
prepend_cnn=false
#
init_opts=         # options, passed to the initialization script
nnet_binary=true

# FEATURE PROCESSING
copy_feats=true # resave the train/cv features into /tmp (disabled by default)
copy_feats_tmproot= # tmproot for copy-feats (optional)
# feature config (applies always)
cmvn_opts=
delta_opts=
# feature_transform:
splice=5         # temporal splicing
splice_step=1    # stepsize of the splicing (1 == no gap between frames)
feat_type=plain
# feature config (applies to feat_type traps)
traps_dct_basis=11 # nr. od DCT basis (applies to `traps` feat_type, splice10 )
# feature config (applies to feat_type transf) (ie. LDA+MLLT, no fMLLR)
transf=
splice_after_transf=5
# feature config (applies to feat_type lda)
lda_dim=300        # LDA dimension (applies to `lda` feat_type)

# LABELS
labels_trainf=      # use these labels to train (override deafault pdf alignments, has to be in 'Posterior' format, see ali-to-post) 
labels_crossvf=

# TRAINING SCHEDULER
learn_rate=0.008   # initial learning rate
train_opts=        # options, passed to the training script
train_tool=        # optionally change the training tool
frame_weights=     # per-frame weights for gradient weighting
train_iters=20
minibatch_size=256

# OTHER
seed=777    # seed value used for training data shuffling and initialization
skip_cuda_check=false
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


. parse_options.sh || exit 1;


if [ $# != 8 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali_train exp/mono_ali_cv exp/mono_nnet"
   echo ""
   echo " Training data : <data-train>,<ali-train> (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev>,<ali-dev> (for learn-rate/model selection based on cross-entopy)"
   echo " note.: <ali-train>,<ali-dev> can point to same directory, or 2 separate directories."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo ""
   echo "  --apply-cmvn <bool>      # apply CMN"
   echo "  --norm-vars <bool>       # add CVN if CMN already active"
   echo "  --splice <N>             # concatenate input features"
   echo "  --splice-step <N>        # stepsize of the splicing"
   echo "  --feat-type <type>       # select type of input features"
   echo ""
   echo "  --nnet-proto <file>       # use this NN prototype"
   echo "  --nnet-init  <file>       # use this to initialize NN" 
   echo "  --nnet-binary <bool>     # write nnet model in bin or txt "
   echo "  --feature-transform <file> # re-use this input feature transform"
   echo "  --hid-layers <N>         # number of hidden layers"
   echo "  --hid-dim <N>            # width of hidden layers"
   echo "  --bn-dim <N>             # make bottle-neck network with bn-with N"
   echo ""
   echo "  --labels-trainf  <file>	# targets for training"
   echo "  --labels-crossvf <file>  # targets for cross-validation"  
   echo ""
   echo "  --learn-rate <float>     # initial leaning-rate"
   echo "  --train-iters <N>        # number of nnet training iterations"
   echo "  --copy-feats <bool>      # copy input features to /tmp (it's faster)"
   echo "  --frame-weights <ark file>  # per-frame weights for gradient weighting"
   echo "  --minibatch-size <N>     # num of frames reqd. to perform parameter update in minibatch SGD"
   echo ""
   exit 1;
fi

data=$1
data_cv=$2
lang=$3
alidir=$4
alidir_cv=$5
num_tgt=$6
nnet_bottom=$7
dir=$8

echo -e "data = $data
		 data_cv = $data_cv
		 lang = $lang
		 alidir = $alidir
		 alidir_cv = $alidir_cv
		 num_tgt = $num_tgt
		 nnet_bottom = $nnet_bottom
		 dir = $dir
		 labels_trainf = $labels_trainf"

# Using alidir for supervision (default)
if [ -z "$labels_trainf" ]; then 
  silphonelist=`cat $lang/phones/silence.csl` || exit 1;
  for f in $alidir/final.mdl; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
fi

for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $alidir \n"
printf "\t CV-set    : $data_cv $alidir_cv \n"
echo

mkdir -p $dir/{log,nnet}

# skip when already trained
[ -e $dir/final.nnet ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))\n\n" && exit 0

# check if CUDA compiled in and GPU is available,
if ! $skip_cuda_check; then cuda-gpu-available || exit 1; fi


###### PREPARE ALIGNMENTS ######
echo
echo "# PREPARING ALIGNMENTS"
if [ ! -z "$labels_trainf" ]; then
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels_trainf"
  labels_cv="$labels_crossvf"
else
  echo "Using PDF targets from dirs '$alidir' '$alidir_cv'"
  # define pdf-alignment rspecifiers
  labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
  labels_cv="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir_cv/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
fi

# copy the old transition model, will be needed by decoder
copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl || exit 1

# copy the tree
cp $alidir/tree $dir/tree || exit 1

echo "labels_tr = $labels_tr"
echo "labels_cv = $labels_cv"

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# shuffle the list
echo "Preparing train/cv lists :"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

# re-save the train/cv features to /tmp, reduces LAN traffic, avoids disk-seeks due to shuffled features
if [ "$copy_feats" == "true" ]; then
  tmpdir=$(mktemp -d $copy_feats_tmproot); mv $dir/train.scp{,_non_local}; mv $dir/cv.scp{,_non_local}
  copy-feats scp:$dir/train.scp_non_local ark,scp:$tmpdir/train.ark,$dir/train.scp || exit 1
  copy-feats scp:$dir/cv.scp_non_local ark,scp:$tmpdir/cv.ark,$dir/cv.scp || exit 1
  trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT
fi

#create a 10k utt subset for global cmvn estimates
head -n 10000 $dir/train.scp > $dir/train.scp.10k


###### PREPARE FEATURE PIPELINE ######
# optionally import feature setup from pre-training,
if [ ! -z $feature_transform ]; then
  D=$(dirname $feature_transform)
  [ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
  [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  [ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
  [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
  echo "Imported config : cmvn_opts='$cmvn_opts' delta_opts='$delta_opts'"
fi

# read the features,
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"
#feats_tr="nnet-forward --use-gpu=yes  --feature-transform=$feature_transform \"ark:copy-feats scp:$dir/train.scp ark:- |\" $nnet_init ark:- "
#feats_cv="nnet-forward --use-gpu=yes  --feature-transform=$feature_transform \"ark:copy-feats scp:$dir/cv.scp ark:- |\" $nnet_init ark:- "

# optionally add per-speaker CMVN,
if [ ! -z "$cmvn_opts" ]; then
  echo "Will use CMVN statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
  [ ! -r $data_cv/cmvn.scp ] && echo "Missing $data_cv/cmvn.scp" && exit 1;
  feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn $cmvn_opts --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp ark:- ark:- |"
else
  echo "apply-cmvn is not used"
fi
# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  feats_cv="$feats_cv add-deltas $delta_opts ark:- ark:- |"
  echo "add-deltas with $delta_opts"
fi
# keep track of the config,
[ ! -z "$cmvn_opts" ] && echo "$cmvn_opts" >$dir/cmvn_opts 
[ ! -z "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts
#

# optionally append python feature transform,
if [ ! -z "$pytel_transform" ]; then
  cp $pytel_transform $dir/pytel_transform.py
  { echo; echo "### Comes from here: '$pytel_transform' ###"; } >> $dir/pytel_transform.py
  pytel_transform=$dir/pytel_transform.py
  feats_tr="$feats_tr /bin/env python $pytel_transform |"
  feats_cv="$feats_cv /bin/env python $pytel_transform |"
fi

# get feature dim
echo -e "\n\nGetting feature dim before feature transformation: "
feat_dim=$(feat-to-dim --print-args=false "$feats_tr" -)
echo "Feature dim is : $feat_dim"


###### INITIALIZE THE NNET ######
# Strip all softmax layers except the first one from $nnet_bottom
final_feature_transform=$dir/final.feature_transform
echo -e "\n\nCreate the feature transform for the top nnet: $final_feature_transform"
local/nnet/make_activesoftmax_from_blocksoftmax.sh  $nnet_bottom  $num_tgt,$num_tgt 1  $final_feature_transform

# Create the proto for top nnet and initialize it
if [ -z $nnet_proto ]; then
nnet_proto=$dir/nnet.proto
utils/nnet/make_nnet_proto.py $num_tgt $num_tgt $hid_layers $num_tgt > $nnet_proto
fi

if [ -z $nnet_init ]; then
nnet_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
echo -e "\n\nInitializing $nnet_proto -> $nnet_init ... "
nnet-initialize --seed=$seed $nnet_proto $nnet_init  2>$log || { cat $log; exit 1; }
fi

echo -e "\n\nGetting feature dim after feature transformation: "
feats_tr="$feats_tr nnet-forward --use-gpu=yes $feature_transform ark:- ark:- |  nnet-forward --use-gpu=yes $final_feature_transform ark:- ark:- |"
feats_cv="$feats_cv nnet-forward --use-gpu=yes $feature_transform ark:- ark:- |  nnet-forward --use-gpu=yes $final_feature_transform ark:- ark:- |"
feat_dim=$(feat-to-dim --print-args=false "$feats_tr" -)
echo "Feature dim is : $feat_dim"
echo "learn rate: $learn_rate"
###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
steps/nnet/train_scheduler.sh \
  --learn-rate $learn_rate \
  --randomizer-seed $seed \
  --max-iters $train_iters \
  --minibatch-size ${minibatch_size} \
  ${train_opts} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${frame_weights:+ --frame-weights "$frame_weights"} \
  ${config:+ --config $config} \
  $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir || exit 1

if $prepend_cnn; then
  echo "Preparing feature transform with CNN layers for RBM pre-training."
  nnet-concat $dir/final.feature_transform "nnet-copy --remove-last-layers=$(((hid_layers+1)*2)) $dir/final.nnet - |" \
    $dir/final.feature_transform_cnn 2>$dir/log/concat_transf_cnn.log || exit 1
fi

exit 0
# Compute priors by doing a forward pass of the data through nnet
echo "Computing priors by doing a forward pass of the data through nnet ..."
steps/nnet/make_priors.sh --use-gpu yes $data $dir

echo "$0 successfuly finished.. $dir"

sleep 3
exit 0
