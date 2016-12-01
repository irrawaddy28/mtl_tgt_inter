#!/bin/bash
. ./path.sh
. ./cmd.sh

dir=exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax_pt_fw0.6_cop0
data_tr90=data-fmllr-tri3c/SW/combined_fw0.6_cop0/AM_AR_CA_DI_HG_MD_train_tr90
data_cv10=data-fmllr-tri3c/SW/combined_fw0.6_cop0/AM_AR_CA_DI_HG_MD_train_cv10
graph_dir=exp/tri3cpt_ali/SW/graph_text_G_SW
	
for nhlayers in 0 1 2 3 4 5 6; do
	dirss=$dir/stackedsoftmax_l$nhlayers
	nnet_init=$dirss/nnet.init
	nnet_proto=$dirss/nnet.proto
	final_feature_transform=$dir/final.feature_transform
	seed=777
	
	local/nnet/train_stackedsoftmax.sh --feature-transform $dir/final.feature_transform --hid-layers $nhlayers --learn-rate 0.008 --train-iters 40 --cmvn-opts "--norm-means=true --norm-vars=true" --delta-opts --delta-order=2 --splice 5 --splice-step 1 --labels-trainf scp:$dir/ali-post/post_block_2.scp --labels-crossvf scp:$dir/ali-post/post_block_2.scp --frame-weights scp:$dir/ali-post/frame_weights_block_2.scp --copy-feats false  $data_tr90 $data_cv10 lang-dummy exp/tri3cpt_ali/SW exp/tri3cpt_ali/SW  950 $dir/final.nnet $dirss	
	
	for type in "dev" "eval"; do
	  decode_dir=$dir/decode_$(basename $dirss)_block_1_${type}_text_G_SW
	  mkdir -p $decode_dir
	  
	  # make other necessary dependencies available
	  (cd $decode_dir; ln -s ../{final.mdl,final.feature_transform,norm_vars,cmvn_opts,delta_opts} . ;)
	  	  
	  # create "prior_counts" first and then create the concatenated nnet
	  #cp $dirss/final.feature_transform $decode_dir/final.nnet
	  #steps/nnet/make_priors.sh --use-gpu "yes" $data_tr90 $decode_dir
	  #nnet-concat $dirss/final.feature_transform $dirss/final.nnet  $decode_dir/final.nnet

	  # create the concatenated nnet first and then create "prior_counts"
          nnet-concat $dirss/final.feature_transform $dirss/final.nnet  $decode_dir/final.nnet
	  steps/nnet/make_priors.sh --use-gpu "yes" $data_tr90 $decode_dir	  
	  
	  # finally, decode
	  data_dir=data-fmllr-tri3c/SW/SW/$type
	  (steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 --srcdir $decode_dir \
	     $graph_dir $(dirname ${data_dir[0]})/$type $decode_dir || exit 1;) &
done

done
