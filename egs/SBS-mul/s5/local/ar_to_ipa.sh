#!/bin/bash

#argument handling
while [ "$1" != "" ]; do
    case $1 in
	--utts)
	    shift
      	    utts=$1
            ;;
        --transdir)
            shift
            dir=$1
            ;;
	*)
	    echo "unknown argument" >&2
    esac
    shift
done

. ./path.sh

export CPLUS_INCLUDE_PATH=${KALDI_ROOT}/tools/openfst/include
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CPLUS_INCLUDE_PATH}/fst:${KALDI_ROOT}/tools/openfst/lib
export PATH=${SBS_DATADIR}/rsloan/phonetisaurus-0.8a/bin:${SBS_DATADIR}/prefix/bin:$PATH
export PYTHONPATH=${SBS_DATADIR}/rsloan/prefix/lib/python2.7/site-packages
python local/ar_to_ipa.py $dir $utts
