#export KALDI_ROOT=`pwd`/../../..
skip_cuda_check=false 
my_use_gpu=yes
corpus_dir=/media/data/workspace/corpus

#host dependendent settings 
if [ `hostname` = "ifp-48" ]; then
	export KALDI_ROOT=/ws/ifp-48_1/hasegawa/amitdas/gold/kaldi/kaldi-trunk-061115
	corpus_dir=/ws/ifp-48_1/hasegawa/amitdas/corpus
elif [ `hostname` = "ifp-30" ]; then
	export KALDI_ROOT=/media/data/workspace/gold/kaldi/kaldi-trunk	
	skip_cuda_check=true
	my_use_gpu=no
elif [ `hostname` = "pac" ]; then
	export KALDI_ROOT=/media/data/workspace/gold/kaldi/kaldi-trunk
elif [ `hostname` = "login" ]; then
	export KALDI_ROOT=`pwd`/../../..
	corpus_dir=/export
else
	export KALDI_ROOT=`pwd`/../../..
	corpus_dir='/export/ws15-pt-data/amitdas/corpus/corpus'
	#echo "Unidentified hostname `hostname`"; exit 1;
fi

. ${corpus_dir}/babel/data/software/env.sh 
#SRILM=/usr/share/srilm/bin/i686-m64
SRILM=/export/ws15-pt-data/srilm/srilm-install/bin/i686-m64
export PATH=$SRILM:$PWD/utils/:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$PWD:$PATH
export LC_ALL=C

