#!/bin/bash -e

# Copyright 2015-2016  University of Illinois (Author: Amit Das)
# Apache 2.0
#

# Top most level script to train and test GMM-HMM and DNN for speech 
# recognition using probablistic transcripts (PT) generated 
# from crowdsource workers
#
# Try:
# 1) looping with increasing DT pseudo labels ... run for each language.
#  

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
[ -f cmd.sh ] && . ./cmd.sh;

## Options
stage=0

# PT-Dir
dir_raw_pt=$SBS_DATADIR/pt-lats

# DNN configurations
cmvn_opts=" --norm-means=true --norm-vars=true "
delta_order=0   # Use 1 for delta, 2 for delta-delta
splice=5
splice_step=1

# Bottleneck configurations
use_bn=false
skip_selftrain=false
hid_layers_bn_stage1=6    # number of hidden layers in BN-DNN
hid_dim_bn_stage1=1024    # dim of hidden layers of BN-DNN
bn_layer=5         # the index of the layer where BN is placed. Using 1-based indexing where W_{i/p-hidden} = W_1
bn_dim=40          # dim of BN layer
remove_last_components=4 # number of components to remove from the top of BN-DNN to expose the BN layer
hid_layers_bn_stage2=4  # number of hidden layers of AM-DNN trained using BN+FMLLR feats
hid_dim_bn_stage2=1024  # dim of hidden layers of AM-DNN trained using BN+FMLLR feats
splice_bn_stage2=2      # splice size of AM-DNN trained using BN+FMLLR feats
splice_step_bn_stage2=2 # splice step of AM-DNN trained using BN+FMLLR feats
# end config

. utils/parse_options.sh || exit 1;


# Usage: ./run_1.sh "AM AR CA DI HG MD" "SW"
TRAIN_LANG=$1  # Example: "AM AR CA HG MD SW"
TEST_LANG=$2   # Example: "DI"
UNILANG_CODE=$(echo $TRAIN_LANG |sed 's/ /_/g')

dir_raw_pt=$dir_raw_pt/${TEST_LANG}
dbn_dir=exp/dnn4_pretrain-dbn/${TEST_LANG}
dnn_dir=exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}
hmm_dir=exp/tri3c/${TEST_LANG}
ali_dt_dir=exp/tri3c_ali/${TEST_LANG}
ali_pt_dir=exp/tri3cpt_ali/${TEST_LANG}
lats_pt_dir=${ali_pt_dir}/decode_train
data_fmllr_dir=data-fmllr-tri3c/${TEST_LANG}
data_bn_dir=data-bn/${TEST_LANG}
data_fmllr_bn_dir=data-fmllr-tri3c-bn/${TEST_LANG}

# =========================================
## Train a multilingual GMM-HMM system (exp/tri3b) using multilingual DT (deterministic transcripts) of training languages
## Test on a test language unseen during training
if [[ $stage -le 0 ]]; then
  ./run_hmm_multilingual.sh "${TRAIN_LANG}" "${TEST_LANG}"
fi
# =========================================

# =========================================
if [[ $stage -le 1 ]]; then
## Do a MAP adaptation of the multilingual GMM-HMM to PT (probabilistic transcripts) of test language. MAP adapted GMM-HMM is saved in exp/tri3c.
  ./run-pt-text-G-map-2.sh  "${TRAIN_LANG}" "${TEST_LANG}" ${dir_raw_pt}
fi
# =========================================

# =========================================
## Train and test a monolingual GMM-HMM (exp/monolingual/tri3b)
## using monolingual DT (deterministic transcripts)
if [[ $stage -le 2 ]]; then
  ./run_hmm_monolingual.sh --stage 1 "${TEST_LANG}"
fi
# =========================================

# =========================================
## Train and test a monolingual DNN system (exp/monolingual/dnn4_pretrain-dbn_dnn)
## using monolingual DT (deterministic transcripts)
if [[ $stage -le 3 ]]; then
  ./run_dnn_monolingual.sh ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
    "${TEST_LANG}" \
    exp/monolingual/tri3b/${TEST_LANG}  \
    exp/monolingual/tri3b_ali/${TEST_LANG} \
    exp/monolingual/data-fmllr-tri3b/${TEST_LANG} \
    exp/monolingual/dnn4_pretrain-dbn/${TEST_LANG}/indbn \
    exp/monolingual/dnn4_pretrain-dbn_dnn/${TEST_LANG}/monosoftmax_dt
fi
# =========================================

# =========================================
if [[ $stage -le 4 ]]; then
## Train a multilingual DNN system using multilingual DT of training languages. Use tri3c_ali as targets.
## This nnet, trained using DT, is used to provide a good initialization of the shared hidden layers (SHLs) using DBN pre-training. If we start training
## with both PT + DT, the SHLs may be unreliable.
  if $use_bn; then
    # train a BN-DNN    
    # w/ random init
    ./run_dnn_adapt_to_multi_dt.sh --stage 2 --train-dbn false --hid-layers $((hid_layers_bn_stage1 - 2)) --hid-dim $hid_dim_bn_stage1 \
         --bn-dim $bn_dim \
        "${TRAIN_LANG}" "${TEST_LANG}" \
         ${hmm_dir} ${ali_dt_dir} ${data_fmllr_dir} \
        dummy ${dnn_dir}/monosoftmax_dt_bn_stage1 || exit 1;
    # w/ pre-training
    #./run_dnn_adapt_to_multi_dt.sh --stage 1 --train-dbn true --hid-layers $hid_layers_bn_stage1 --hid-dim $hid_dim_bn_stage1 \
        #--bn-layer $bn_layer --bn-dim $bn_dim \
        #"${TRAIN_LANG}" "${TEST_LANG}" \
        #${hmm_dir} ${ali_dt_dir} ${data_fmllr_dir} \
        #${dbn_dir}/monosoftmax_dt_bn_stage1 ${dnn_dir}/monosoftmax_dt_bn_stage1 || exit 1;
        
    # get BN feats and append with fMLLR feats
    ./run_bnf.sh --use-gpu yes --remove-last-components $remove_last_components \
        "${TRAIN_LANG}" "${TEST_LANG}" ${data_fmllr_dir} \
        ${dnn_dir}/monosoftmax_dt_bn_stage1 ${data_bn_dir}/monosoftmax_dt_bn_stage1 ${data_fmllr_bn_dir}/monosoftmax_dt_bn_stage1 || exit 1;
        
    # initialize an AM-DNN with fMLLR+BN feats obtained from multilingual data. Then train the AM-DNN.
    # w/ random init
    ./run_dnn_adapt_to_multi_dt.sh --stage 3 --train-dbn false --hid-layers $((hid_layers_bn_stage2 - 2)) --hid-dim $hid_dim_bn_stage2 \
        --bn-dim $bn_dim \
        --splice $splice_bn_stage2 --splice-step $splice_step_bn_stage2 \
        --skip-decode false \
        "${TRAIN_LANG}" "${TEST_LANG}" \
       ${hmm_dir} ${ali_dt_dir} ${data_fmllr_bn_dir}/monosoftmax_dt_bn_stage1 \
        dummy ${dnn_dir}/monosoftmax_dt_bn_stage2 || exit 1;            
    # w/ pre-training
    #./run_dnn_adapt_to_multi_dt.sh --stage 2 --train-dbn true --hid-layers $hid_layers_bn_stage2 --hid-dim $hid_dim_bn_stage2 \
        #"${TRAIN_LANG}" "${TEST_LANG}" \
        #${hmm_dir} ${ali_dt_dir} ${data_fmllr_bn_dir}/monosoftmax_dt_bn_stage1 \
        #${dbn_dir}/indbn/monosoftmax_dt_bn_stage2 ${dnn_dir}/monosoftmax_dt_bn_stage2 || exit 1;
  else
    ./run_dnn_adapt_to_multi_dt.sh --stage 1 --train-dbn true \
      ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
      "${TRAIN_LANG}" "${TEST_LANG}" \
      ${hmm_dir} ${ali_dt_dir} ${data_fmllr_dir} \
      ${dbn_dir}/monosoftmax_dt ${dnn_dir}/monosoftmax_dt || exit 1;
  fi
fi
# =========================================

# =========================================
## Demonstrate the efficacy of a monosoftmax DNN trained with PTs where PTs are generated by an ASR system rather than crowdsource workers.
if [[ $stage -le 20 && $skip_selftrain == "false" ]]; then
  unsup_dir_tag="train"
  acwt=0.2
  feat_unsup_dir=data-fmllr-tri3b/${TEST_LANG}/${unsup_dir_tag}
  decoding_mdl_dir=${dnn_dir}/monosoftmax_dt # dnn mdl directory used to decode the unsup data
  lats_unsup_dir=${decoding_mdl_dir}/decode_${unsup_dir_tag}_text_G_${TEST_LANG} # dir where lattices generated by decoding unsup data will be saved
  
  # Now decode the training data using a reasonably well trained DNN model. The fMLLR transforms for train set are saved in tri3b_ali/decode_train
  # and the decoding lattices in the same DNN directory which is used for decoding the training data.
  # Note: We could also use GMM-HMM model tri3b/final.mdl for decoding but as of now the unsup lats scipt supports decoding using a nnet model.
  local/get_unsup_lats.sh --stage -2 --feats-nj 10 --unsup-dir-name ${unsup_dir_tag} \
    ${TEST_LANG} \
    exp/tri3b_ali/${TEST_LANG} \
    data-fmllr-tri3b/${TEST_LANG}/${TEST_LANG}/${unsup_dir_tag} \
    exp/tri3b/${TEST_LANG}/graph_text_G_${TEST_LANG}  \
    ${decoding_mdl_dir}  \
    ${lats_unsup_dir} || exit 1;
   
  # Copy the fMLLR transforms for dev and eval sets from tri3b to tri3b_ali
  rm -rf exp/tri3b_ali/${TEST_LANG}/decode_dev_${TEST_LANG} exp/tri3b_ali/${TEST_LANG}/decode_eval_${TEST_LANG}
  cp -Lr exp/tri3b/${TEST_LANG}/decode_dev_${TEST_LANG} exp/tri3b_ali/${TEST_LANG}/decode_dev_${TEST_LANG}
  cp -Lr exp/tri3b/${TEST_LANG}/decode_eval_${TEST_LANG} exp/tri3b_ali/${TEST_LANG}/decode_eval_${TEST_LANG}
  
  # Now fine tune the DNN using the decoded unsup lattice. Use different levels of frame weighting derived from best path lattice.
  # Use the fMLLR transforms from tri3b_ali/decode_* and training lattices from the DNN directory ${lats_unsup_dir}
  thresh=0.5
  ./run_dnn_adapt_to_mono_pt_frame_wt.sh --stage 1 --replace-softmax "true"  \
       --transform-dir-train "exp/tri3b_ali/${TEST_LANG}/decode_${unsup_dir_tag}_${TEST_LANG}" --threshold ${thresh} \
       ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
      "${TEST_LANG}" \
      exp/tri3b_ali/${TEST_LANG}  \
      ${lats_unsup_dir} \
      "${dnn_dir}/monosoftmax_dt/final.nnet" \
      data-fmllr-tri3b/${TEST_LANG} \
      ${dnn_dir}/monosoftmax_asrpt_fw${thresh} || exit 1;
  for thresh in 0.6 0.7 0.8 0.9 ; do
    (./run_dnn_adapt_to_mono_pt_frame_wt.sh --stage 2 --replace-softmax "true" \
       --transform-dir-train "exp/tri3b_ali/${TEST_LANG}/decode_${unsup_dir_tag}_${TEST_LANG}"  --threshold ${thresh} \
       ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
      "${TEST_LANG}" \
      exp/tri3b_ali/${TEST_LANG}  \
      ${lats_unsup_dir} \
      "${dnn_dir}/monosoftmax_dt/final.nnet" \
      data-fmllr-tri3b/${TEST_LANG} \
      ${dnn_dir}/monosoftmax_asrpt_fw${thresh} || exit 1;) &
  done
  wait
fi
# =========================================

# =========================================
## Demonstrate the efficacy of a monosoftmax DNN trained with PTs where PTs are generated by crowdsource workers.
if [[ $stage -le 30 ]]; then
# Now, on top of the hidden layers of the multilingual DT system, create a new soft-max layer. This becomes a new DNN.
# Fine tune all layers of this new DNN using PT of the test language.
    if $use_bn; then
        # adapt the BN-DNN using PTs
        ./run_dnn_adapt_to_mono_pt.sh --stage 1 --train-dbn false --precomp-dnn "${dnn_dir}/monosoftmax_dt_bn_stage1/final.nnet" --replace-softmax "true" \
            "${TRAIN_LANG}" "${TEST_LANG}" \
            ${hmm_dir} ${ali_pt_dir} \
            ${data_fmllr_dir}  ${dnn_dir}/monosoftmax_pt_bn_stage1  ${dnn_dir}/monosoftmax_pt_bn_stage1 || exit 1;
        
        # get BN feats and append with fMLLR feats
        ./run_bnf.sh --use-gpu yes --remove-last-components $remove_last_components \
            "${TRAIN_LANG}" "${TEST_LANG}" ${data_fmllr_dir} \
            ${dnn_dir}/monosoftmax_pt_bn_stage1 ${data_bn_dir}/monosoftmax_pt_bn_stage1 ${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1 || exit 1;
        
        # initialize an AM-DNN with fMLLR+BN feats obtained from multilingual data
        ./run_dnn_adapt_to_multi_dt.sh --stage 3 --train-dbn false --hid-layers $((hid_layers_bn_stage2 - 2)) --hid-dim $hid_dim_bn_stage2 \
            --bn-dim $bn_dim \
            ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-opts "--delta-order=$delta_order" --splice $splice_bn_stage2 --splice-step $splice_step_bn_stage2 \
            --skip-decode true \
            "${TRAIN_LANG}" "${TEST_LANG}" \
            ${hmm_dir} ${ali_dt_dir} \
            ${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1 \
            ${dnn_dir}/monosoftmax_pt_bn_init_stage2 ${dnn_dir}/monosoftmax_pt_bn_init_stage2 || exit 1;
                    
        # train the AM-DNN
        (./run_dnn_adapt_to_mono_pt.sh --stage 3 --train-dbn false --precomp-dnn "${dnn_dir}/monosoftmax_pt_bn_init_stage2/final.nnet" \
            --replace-softmax true \
            ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-opts "--delta-order=$delta_order" --splice $splice_bn_stage2 --splice-step $splice_step_bn_stage2 \
            "${TRAIN_LANG}" "${TEST_LANG}" \
            ${hmm_dir} ${ali_pt_dir} \
            ${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1 \
            ${dnn_dir}/monosoftmax_pt_bn_stage2 ${dnn_dir}/monosoftmax_pt_bn_stage2 || exit 1;) &
            
        # Train the AM-DNN w/ pretraining
        #(./run_dnn_adapt_to_mono_pt.sh --stage 2 --train-dbn true \
            #--hid-layers $hid_layers_bn_stage2 --hid-dim $hid_dim_bn_stage2 \
            #--splice $splice --splice-step $splice_step \
            #"${TRAIN_LANG}" "${TEST_LANG}" \
            #${hmm_dir} ${ali_pt_dir} \
            #${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1 ${dbn_dir}/monosoftmax_pt_bn_stage2_$tag ${dnn_dir}/monosoftmax_pt_bn_stage2_$tag || exit 1; ) &         
        
        ## train a new AM-DNN with fMLLR+BN feats
        ## randomly init dnn
        #for h_layers_bn_stage2 in 4 6; do
          #for h_dim_bn_stage2 in 1024 2048; do
            #for splice_step in 1 2 5; do
              #for splice in 2 5; do
                #tag=l_${h_layers_bn_stage2}_d_${h_dim_bn_stage2}_sp_${splice}_st_${splice_step}
                
                ## Initialize the AM-DNN with multilingual data
                #if [[ ! -f ${dnn_dir}/monosoftmax_pt_init_bn_stage2_$tag/final.nnet ]]; then
                  #./run_dnn_adapt_to_multi_dt.sh --stage 3 --train-dbn false --hid-layers $((h_layers_bn_stage2 - 2)) --hid-dim $h_dim_bn_stage2 \
                    #--bn-dim $bn_dim \
                    #--splice $splice --splice-step $splice_step \
                    #--skip-decode true \
                    #"${TRAIN_LANG}" "${TEST_LANG}" \
                    #${hmm_dir} ${ali_dt_dir} \
                    #${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1 \
                    #${dnn_dir}/monosoftmax_pt_init_bn_stage2_$tag ${dnn_dir}/monosoftmax_pt_init_bn_stage2_$tag || exit 1;
                #fi
        
                ## Train the AM-DNN w/ multilingual data initialization
                #(./run_dnn_adapt_to_mono_pt.sh --stage 3 --train-dbn false --precomp-dnn "${dnn_dir}/monosoftmax_pt_init_bn_stage2_$tag/final.nnet" \
                    #--replace-softmax true \
                    #--splice $splice --splice-step $splice_step \
                    #"${TRAIN_LANG}" "${TEST_LANG}" \
                    #${hmm_dir} ${ali_pt_dir} \
                    #${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1 \
                    #${dnn_dir}/monosoftmax_pt_bn_stage2_$tag ${dnn_dir}/monosoftmax_pt_bn_stage2_$tag || exit 1;) &
                ## Train the AM-DNN w/ pretraining
                ##(./run_dnn_adapt_to_mono_pt.sh --stage 2 --train-dbn true \
                    ##--hid-layers $hid_layers_bn_stage2 --hid-dim $hid_dim_bn_stage2 \
                    ##--splice $splice --splice-step $splice_step \
                    ##"${TRAIN_LANG}" "${TEST_LANG}" \
                    ##${hmm_dir} ${ali_pt_dir} \
                    ##${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1 ${dbn_dir}/monosoftmax_pt_bn_stage2_$tag ${dnn_dir}/monosoftmax_pt_bn_stage2_$tag || exit 1; ) &
              #done
              #wait
            #done
          #done
        #done       
        
    else
        ./run_dnn_adapt_to_mono_pt.sh --stage 1  --train-dbn false  --precomp-dnn "${dnn_dir}/monosoftmax_dt/final.nnet" --replace-softmax "true" \
          ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
          "${TRAIN_LANG}" \
          "${TEST_LANG}" \
          ${hmm_dir} \
          ${ali_pt_dir} \
          ${data_fmllr_dir} \
          ${dnn_dir}/monosoftmax_pt ${dnn_dir}/monosoftmax_pt || exit 1;
    fi
fi

if [[ $stage -le 31 ]]; then
# Now try the same thing using different levels of frame weighting derived from best path PT lattice. Do we get good improvements using frame weighting?
for thresh in 0.5 0.6 0.7 0.8 0.9 ; do
   if $use_bn; then
        # adapt the BN-DNN using PTs
        ./run_dnn_adapt_to_mono_pt_frame_wt.sh --stage 2  --replace-softmax "true" \
            --splice 5 --splice-step 1 --threshold ${thresh} \
            "${TEST_LANG}" ${ali_pt_dir} ${lats_pt_dir} \
            ${dnn_dir}/monosoftmax_dt_bn_stage1/final.nnet \
            ${data_fmllr_dir}  ${dnn_dir}/monosoftmax_pt_bn_stage1_fw${thresh} || exit 1;
        
        # get BN feats and append with fMLLR feats
        ./run_bnf.sh --use-gpu yes --remove-last-components $remove_last_components \
            "${TRAIN_LANG}" "${TEST_LANG}" ${data_fmllr_dir} \
            ${dnn_dir}/monosoftmax_pt_bn_stage1_fw${thresh} ${data_bn_dir}/monosoftmax_pt_bn_stage1_fw${thresh} ${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1_fw${thresh} || exit 1;            
        
        # initialize an AM-DNN with fMLLR+BN feats obtained from multilingual data
        ./run_dnn_adapt_to_multi_dt.sh --stage 3 --train-dbn false --hid-layers $((hid_layers_bn_stage2 - 2)) --hid-dim $hid_dim_bn_stage2 \
            --bn-dim $bn_dim \
            --splice $splice_bn_stage2 --splice-step $splice_step_bn_stage2 \
            --skip-decode true \
            "${TRAIN_LANG}" "${TEST_LANG}" \
            ${hmm_dir} ${ali_dt_dir} \
            ${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1_fw${thresh} \
            ${dnn_dir}/monosoftmax_pt_bn_init_stage2_fw${thresh} ${dnn_dir}/monosoftmax_pt_bn_init_stage2_fw${thresh} || exit 1;
        
        # train the AM-DNN
        (./run_dnn_adapt_to_mono_pt_frame_wt.sh --stage 2  --replace-softmax "true" \
            --splice $splice_bn_stage2 --splice-step $splice_step_bn_stage2 --threshold ${thresh} \
            "${TEST_LANG}" ${ali_pt_dir} ${lats_pt_dir} \
            ${dnn_dir}/monosoftmax_pt_bn_init_stage2_fw${thresh}/final.nnet \
            ${data_fmllr_bn_dir}/monosoftmax_pt_bn_stage1_fw${thresh} ${dnn_dir}/monosoftmax_pt_bn_stage2_fw${thresh} || exit 1;) &
        
    else
        (./run_dnn_adapt_to_mono_pt_frame_wt.sh --stage 2 --replace-softmax "true" \
            ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
            --threshold ${thresh} \
            "${TEST_LANG}" \
            ${ali_pt_dir} \
            ${lats_pt_dir} \
            "${dnn_dir}/monosoftmax_dt/final.nnet" \
            ${data_fmllr_dir} \
            ${dnn_dir}/monosoftmax_pt_fw${thresh} || exit 1;) &
    fi
done
wait
fi
# =========================================

# =========================================
## Demonstrate the efficacy of a multisoftmax DNN trained with both PT and DT where softmax blocks are arranged as block 1:block2 = PT:DT. PT from crowdsource workers.
if [[ $stage -le 40 ]]; then
    
for thresh in 0.5 0.6 0.7; do # full range: 0.5 0.6 0.7 0.8 0.9
  for num_copies_dt in 0 2 4; do # full range: 0 2 4 6
    for num_copies in 3 4 5; do # full range: 0 1 2 3 4
      if $use_bn; then
        data_fmllr_cop_dir=${data_fmllr_dir}/combined_bn_stage1_fw${thresh}_cop${num_copies}
        bn_dnn_dir=${dnn_dir}/multisoftmax_pt_bn_stage1_fw${thresh}_cop${num_copies}
        bn_feats_dir=${data_bn_dir}/multisoftmax_pt_bn_stage1_fw${thresh}_cop${num_copies}
        bn_fmllr_feats_dir=${data_fmllr_bn_dir}/multisoftmax_pt_bn_stage1_fw${thresh}_cop${num_copies}    
      
        # adapt the BN-DNN using PTs
        if [[ ! -f "${bn_dnn_dir}/final.nnet" ]]; then
        ./run_dnn_multilingual.sh --dnn-init "${dnn_dir}/monosoftmax_dt_bn_stage1/final.nnet" --data-type-csl "pt:dt"  --lang-weight-csl "1.0:1.0"  \
          --threshold-csl "${thresh}:0.0" --lat-dir-csl "${lats_pt_dir}:-" --dup-and-merge-csl "${num_copies}>>1:0>>0" \
          --splice 5 --splice-step 1 \
          "${TEST_LANG}:${UNILANG_CODE}" "${ali_pt_dir}:${ali_dt_dir}" \
          "${data_fmllr_dir}/${TEST_LANG}/train:${data_fmllr_dir}/${UNILANG_CODE}/train" ${data_fmllr_cop_dir} \
          ${bn_dnn_dir} || exit 1;

        # get BN feats and append with fMLLR feats
        ./run_bnf.sh --use-gpu yes --remove-last-components ${remove_last_components} \
          "${TRAIN_LANG}" "${TEST_LANG}" ${data_fmllr_dir} \
          ${bn_dnn_dir} ${bn_feats_dir} ${bn_fmllr_feats_dir} || exit 1;
        else
          echo "Features already exist in: ${bn_fmllr_feats_dir} . Skip"
        fi
       
        for splice_step_bn_stage2 in 1 2 5; do
          for splice_bn_stage2 in 2 5; do        
            tag="sp_${splice_bn_stage2}_st_${splice_step_bn_stage2}"          
            bnam_fmllr_feats_cop_dir=${data_fmllr_bn_dir}/combined_bn_stage2_fw${thresh}_cop${num_copies}
            bnam_init_dnn_dir=${dnn_dir}/multisoftmax_pt_bn_init_stage2_fw${thresh}_cop${num_copies}
            bnam_dnn_dir=${dnn_dir}/multisoftmax_pt_bn_stage2_fw${thresh}_cop${num_copies}
              
            bnam_fmllr_feats_cop_dir=${bnam_fmllr_feats_cop_dir}_${tag}
            bnam_init_dnn_dir=${bnam_init_dnn_dir}_${tag}
            bnam_dnn_dir=${bnam_dnn_dir}_${tag}
                    
            if [[ ! -f "${bnam_dnn_dir}/final.nnet" ]]; then
            # initialize an AM-DNN with fMLLR+BN feats obtained from multilingual data
            ./run_dnn_adapt_to_multi_dt.sh --stage 3 --train-dbn false --hid-layers $((hid_layers_bn_stage2 - 2)) --hid-dim $hid_dim_bn_stage2 \
              --bn-dim $bn_dim \
              --splice ${splice_bn_stage2} --splice-step ${splice_step_bn_stage2} \
              --skip-decode true \
              "${TRAIN_LANG}" "${TEST_LANG}" \
              ${hmm_dir} ${ali_dt_dir} \
              ${bn_fmllr_feats_dir} \
              ${bnam_init_dnn_dir} ${bnam_init_dnn_dir} || exit 1;        
            # train the AM-DNN
            ./run_dnn_multilingual.sh --dnn-init "${bnam_init_dnn_dir}/final.nnet" --data-type-csl "pt:dt"  --lang-weight-csl "1.0:1.0"  \
              --threshold-csl "${thresh}:0.0" --lat-dir-csl "${lats_pt_dir}:-" --dup-and-merge-csl "${num_copies}>>1:0>>0" \
              --splice ${splice_bn_stage2} --splice-step ${splice_step_bn_stage2} \
              "${TEST_LANG}:${UNILANG_CODE}" "${ali_pt_dir}:${ali_dt_dir}" \
              "${bn_fmllr_feats_dir}/${TEST_LANG}/train:${bn_fmllr_feats_dir}/${UNILANG_CODE}/train" ${bnam_fmllr_feats_cop_dir} \
              ${bnam_dnn_dir} || exit 1;
            else
              echo "Model already exist in: ${bnam_fmllr_feats_cop_dir}/final.nnet . Skip"    
            fi        
          done # splice
          wait
        done # splice step
      else
        for alpha in 0.2 0.4 0.6 0.8 1.0; do
          etag=fw${thresh}_cop${num_copies}${num_copies_dt}_alpha${alpha}
          nnet_outdir=${dnn_dir}/multisoftmax_pt_$etag
          mkdir -p $nnet_outdir
          [ -f ${nnet_outdir}/final.nnet ] && echo "${nnet_outdir}/final.nnet exists. Skipping this run" && continue         
          (./run_dnn_multilingual.sh --dnn-init "${dnn_dir}/monosoftmax_dt/final.nnet" \
            --objective-csl "xent:xent" --data-type-csl "pt:dt"  --lang-weight-csl "1.0:${alpha}"  \
            --renew-nnet-type "parallel" \
            --threshold-csl "${thresh}:0.0" \
            --lat-dir-csl "${lats_pt_dir}:-" \
            --dup-and-merge-csl "${num_copies}>>1:${num_copies_dt}>>2" \
            ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
            "${TEST_LANG}:${UNILANG_CODE}" "${ali_pt_dir}:${ali_dt_dir}" \
            "${data_fmllr_dir}/${TEST_LANG}/train:${data_fmllr_dir}/${UNILANG_CODE}/train" ${data_fmllr_dir}/combined_$etag \
            ${nnet_outdir} 2>&1 | tee ${nnet_outdir}/run_dnn_multilingual.log || exit 1; ) &
        done
        wait
      fi
    done  # num_copies
  done  # num_copies dt  
done # thresh
fi
# =========================================

# =========================================
## Demonstrate the efficacy of a multisoftmax DNN trained with PT, DT, and ST labels (self-training)
nutts=4000
thresh=0.6
num_copies=0
unsup_dir_name="unsup_$nutts"
feat_unsup_dir=${data_fmllr_dir}/${TEST_LANG}/${unsup_dir_name}
decoding_mdl_dir=${dnn_dir}/multisoftmax_pt_fw${thresh}_cop${num_copies}/decode_block_1_dev_text_G_${TEST_LANG} # dnn mdl directory used to decode the unsup data
lats_unsup_dir=${dnn_dir}/multisoftmax_pt_fw${thresh}_cop${num_copies}/decode_${unsup_dir_name}_text_G_${TEST_LANG} # dir where lattices generated by decoding unsup data will be saved
if [[ $stage -le 50 ]]; then

# Now decode the unsupervised data using a reasonably well trained DNN model
local/get_unsup_lats.sh --nutts ${nutts} --unsup-dir-name ${unsup_dir_name} ${TEST_LANG} ${ali_pt_dir} ${feat_unsup_dir} \
  ${hmm_dir}/graph_text_G_${TEST_LANG}  ${decoding_mdl_dir}  ${lats_unsup_dir} || exit 1;

feat_pt_dir=${data_fmllr_dir}/${TEST_LANG}/train
feat_dt_dir=${data_fmllr_dir}/${UNILANG_CODE}/train

for nutts_small_unsup in 4000 ; do  # 4000 3000 2000 1000
  for thresh_pt in 0.6 ; do  # 0.6 0.7 0.8
    for thresh_unsup in 0.9; do # 0.7 0.9    

      for num_copies_unsup in 0 1 2 3 4; do # 0 1
        for num_copies_dt in 0; do # 0 2 4 6
          for num_copies_pt in 0 2 4; do # 0 2 4
          
            num_copies=(${num_copies_pt} ${num_copies_dt} ${num_copies_unsup})
            thresh=(${thresh_pt} 0.0 ${thresh_unsup})
            nutts_small=${nutts_small_unsup}
           
            unsupsmall_dir_tag="unsup_${nutts_small}" 
            feat_unsupsmall_dir=${data_fmllr_dir}/${TEST_LANG}/${unsupsmall_dir_tag}
           
            if [ "$nutts_small" -lt "$nutts" ]; then             
              utils/subset_data_dir.sh ${feat_unsup_dir} ${nutts_small} ${feat_unsupsmall_dir}
            fi
           
            etag=fw${thresh[0]}${thresh[1]}${thresh[2]}_cop${num_copies[0]}${num_copies[1]}${num_copies[2]}_unsup${nutts_small}
            nnet_outdir=${dnn_dir}/multisoftmax_pt_$etag
            mkdir -p $nnet_outdir
            [ -f ${nnet_outdir}/final.nnet ] && echo "${nnet_outdir}/final.nnet exists. Skipping this run" && continue
            ( ./run_dnn_multilingual.sh --dnn-init "${dnn_dir}/monosoftmax_dt/final.nnet" \
                --objective-csl "xent:xent:xent" --data-type-csl "pt:dt:semisup" --lang-weight-csl "1.0:1.0:1.0" \
                --renew-nnet-type "parallel" \
                --threshold-csl "${thresh[0]}:${thresh[1]}:${thresh[2]}" \
                --lat-dir-csl "${lats_pt_dir}:-:${lats_unsup_dir}" \
                 --dup-and-merge-csl "${num_copies[0]}>>1:${num_copies[1]}>>2:${num_copies[2]}>>1" \
                ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
                "${TEST_LANG}:${UNILANG_CODE}:${TEST_LANG}" "${ali_pt_dir}:${ali_dt_dir}:${ali_pt_dir}" \
                "${feat_pt_dir}:${feat_dt_dir}:${feat_unsupsmall_dir}" \
                ${data_fmllr_dir}/combined_$etag \
                ${nnet_outdir} ) &
          done # num_copies_pt
          wait
        done # num_copies_dt
      done # num_copies_unsup
      
    done # thresh_unsup
  done # thresh_pt
done # nutts_small_unsup

fi
# =========================================

# =========================================
## Train MTL with PT:DT:Autoencoder

#./run_dnn_multilingual.sh --dnn-init "exp/dnn4_pretrain-dbn_dnn/SW/monosoftmax_dt/final.nnet" --objective-csl "xent:xent:mse" --data-type-csl "pt:dt:unsup" --lang-weight-csl "1.0:1.0:0.2" --threshold-csl "0.6:0.0:0.0" --lat-dir-csl "exp/tri3cpt_ali/SW/decode_train:-:-" --dup-and-merge-csl "2>>1:1>>2:1>>3" --delta-order 2 #--renew-nnet-type "parallel"  --renew-nnet-opts "--parallel-softmax-opts ::--no-softmax "  "SW:AM_AR_CA_DI_HG_MD:SW" "exp/tri3cpt_ali/SW:exp/tri3c_ali/SW:exp/tri3cpt_ali/SW" #"data-fmllr-tri3c/SW/SW/train:data-fmllr-tri3c/SW/AM_AR_CA_DI_HG_MD/train:data-fmllr-tri3c/SW/SW/unsup_100" data-fmllr-tri3c/SW/combined_fw0.60.00.9_cop211_unsup100 #exp/dnn4_pretrain-dbn_dnn/SW/multisoftmax_pt_fw0.60.00.9_cop211_unsup100

nutts=6000
unsup_dir_name="unsup_$nutts"
feat_unsup_dir=${data_fmllr_dir}/${TEST_LANG}/${unsup_dir_name}
if [[ $stage -le 60 ]]; then

  # Extract the fMLLR features of unsup data using the PT adapted HMM model
  if [ ! -f ${ali_pt_dir}/decode_${unsup_dir_name}_${TEST_LANG}/lat.1.gz ]; then
    local/get_unsup_lats.sh --nutts ${nutts} --unsup-dir-name ${unsup_dir_name} --skip-decode true \
      ${TEST_LANG} \
      ${ali_pt_dir} \
      ${feat_unsup_dir} \
      ${hmm_dir}/graph_text_G_${TEST_LANG}  \
      dummy  \
      dummy || exit 1;
  fi
  
  feat_pt_dir=${data_fmllr_dir}/${TEST_LANG}/train
  feat_dt_dir=${data_fmllr_dir}/${UNILANG_CODE}/train

  for nutts_subset in 4000 ; do  # 4000 3000 2000 1000
    for thresh_pt in 0.6 0.7 0.8; do  # 0.6 0.7 0.8
      for num_copies_unsup in 0 1; do # 0 1
        for num_copies_dt in 0 2 4; do # 0 2 4 6
          for num_copies_pt in 0 2 4; do # 0 2 4
            num_copies=(${num_copies_pt} ${num_copies_dt} ${num_copies_unsup})
            thresh=(${thresh_pt} 0.0 0.0)
            feat_unsup_subset_dir=${data_fmllr_dir}/${TEST_LANG}/"unsup_${nutts_subset}"
            if [ "$nutts_subset" -lt "$nutts" ]; then
              invalid=1
              # If feat dir already exists, check if it is still valid
              [ -d ${feat_unsup_subset_dir} ] && utils/validate_data_dir.sh --no-text ${feat_unsup_subset_dir} && invalid=$?
              # If invalid, create the subset dir
              [ $invalid != 0 ] && utils/subset_data_dir.sh ${feat_unsup_dir} ${nutts_subset} ${feat_unsup_subset_dir}
            fi
            
            for alpha in 0.2 0.4 0.6 0.8 1.0; do
              etag=fw${thresh[0]}${thresh[1]}${thresh[2]}_cop${num_copies[0]}${num_copies[1]}${num_copies[2]}_unsup${nutts_subset}_alpha${alpha}
              nnet_outdir=${dnn_dir}/multisoftmax_pt_$etag
              mkdir -p $nnet_outdir
              [ -f ${nnet_outdir}/final.nnet ] && echo "${nnet_outdir}/final.nnet exists. Skipping this run" && continue
              ./run_dnn_multilingual.sh --dnn-init "${dnn_dir}/monosoftmax_dt/final.nnet" \
              --objective-csl "xent:xent:mse" --data-type-csl "pt:dt:unsup" --lang-weight-csl "1.0:1.0:${alpha}" \
              --renew-nnet-type "parallel" --renew-nnet-opts "--parallel-softmax-opts ::--no-softmax " \
              --threshold-csl "${thresh[0]}:${thresh[1]}:${thresh[2]}" \
              --lat-dir-csl "${lats_pt_dir}:-:-" \
              --dup-and-merge-csl "${num_copies[0]}>>1:${num_copies[1]}>>2:${num_copies[2]}>>1" \
              ${cmvn_opts:+ --cmvn-opts "$cmvn_opts"} --delta-order $delta_order --splice $splice --splice-step $splice_step \
              "${TEST_LANG}:${UNILANG_CODE}:${TEST_LANG}" "${ali_pt_dir}:${ali_dt_dir}:${ali_pt_dir}" \
              "${feat_pt_dir}:${feat_dt_dir}:${feat_unsup_subset_dir}" \
              ${data_fmllr_dir}/combined_$etag \
              ${nnet_outdir}
            done # alpha
          done # num_copies_pt
        done # num_copies_dt
      done # num_copies_unsup  
    done # thresh_pt
  done # nutts_subset

fi
