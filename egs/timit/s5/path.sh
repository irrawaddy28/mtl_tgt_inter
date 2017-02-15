#export KALDI_ROOT=`pwd`/../../..
#default settings for steps/nnet/{pretrain_dbn.sh, train.sh}
skip_cuda_check=false 
my_use_gpu=yes
corpus_dir=/media/data/workspace/corpus
export use_worldbet=false

#host dependendent settings 
if [ `hostname` = "ifp-48" ]; then
	export KALDI_ROOT=/ws/ifp-48_1/hasegawa/amitdas/work/SBS-kaldi-2015
	corpus_dir=/ws/rz-cl-2/hasegawa/amitdas/corpus
elif [ `hostname` = "ifp-30" ]; then
	export KALDI_ROOT=/media/data/workspace/gold/kaldi/kaldi-trunk
	skip_cuda_check=true
	my_use_gpu=no
elif [ `hostname` = "work" ]; then
	export KALDI_ROOT=`pwd`/../../..
	corpus_dir=/ws/corpus
else
	echo "Unidentified hostname `hostname`"; exit 1;
fi

export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$PWD:$PATH
export PATH=$PATH:/usr/local/cuda-7.5/bin
export LC_ALL=C
export IRSTLM=$KALDI_ROOT/tools/irstlm