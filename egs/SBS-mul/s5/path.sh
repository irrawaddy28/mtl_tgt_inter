if [[ `hostname` == "ifp-48" ]]; then
  export KALDI_ROOT=/ws/ifp-48_1/hasegawa/amitdas/work/SBS-kaldi-2015
  export N_BG=4
  export parallel_opts="--num-threads 2"
  export randomizer_size=131072   # per-job size = 2^17 = 131072 fills about 750 M in GPU
  export minibatch_size=512
elif  [[ `hostname` == "ifp-04" ]]; then
  export KALDI_ROOT=/ws/ifp-04_2/hasegawa/amitdas/work/SBS-kaldi-2015
  export N_BG=4
  export parallel_opts="--num-threads 6"
  export randomizer_size=524288   # per-job size = 2^19 = 524288 fills about 2.3 G in GPU
  export minibatch_size=512
elif  [[ `hostname` == "IFP-05" ]]; then
  export KALDI_ROOT=/ws/ifp-48_1/hasegawa/amitdas/work/SBS-kaldi-2015_`hostname`
  export N_BG=4
  export parallel_opts="--num-threads 6"
  export randomizer_size=262144   # per-job size = 2^18 = 262144 fills about 1.5 G in GPU
  export minibatch_size=512 
fi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$PWD:$PATH:/export/ws15-pt-data/python-3.4.3/bin
export LC_ALL=C
export IRSTLM=$KALDI_ROOT/tools/irstlm
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH  #/usr/local/cuda/lib64/stubs:
export SBS_DATADIR=/ws/rz-cl-2/hasegawa/amitdas/corpus/ws15-pt-data/data
