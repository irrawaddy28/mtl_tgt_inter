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

export CPLUS_INCLUDE_PATH=/export/ws15-pt-data/kaldi-trunk/tools/openfst/include
export LD_LIBRARY_PATH=/export/ws15-pt-data/kaldi-trunk/tools/openfst/include/fst:/export/ws15-pt-data/kaldi-trunk/tools/openfst/lib
export PATH=/export/ws15-pt-data/kaldi-trunk/tools/openfst/bin:/export/ws15-pt-data/rsloan/phonetisaurus-0.8a/bin:/export/ws15-pt-data/rsloan/prefix/bin:$PATH
export PYTHONPATH=/export/ws15-pt-data/rsloan/prefix/lib/python2.7/site-packages
python /export/ws15-pt-data/rsloan/dt_to_ipa.py $dir $utts