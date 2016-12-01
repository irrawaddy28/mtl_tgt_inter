#!/bin/bash

train_lang=$1
test_lang=$2

echo "Building universal train/test data..."

mkdir -p data/local/data

gen_set() {

    x=$1
    lang=$2

    wav_tmp=$(mktemp)
    utt2spk_tmp=$(mktemp)
    text_tmp=$(mktemp)
    
    for L in $lang; do
        cat data/$L/local/data/${x}_wav.scp >> $wav_tmp
        cat data/$L/local/data/${x}_utt2spk >> $utt2spk_tmp
        cat data/$L/local/data/${x}_text >> $text_tmp
    done

    sort -k1,1 $wav_tmp > data/local/data/${x}_wav.scp
    sort -k1,1 $text_tmp > data/local/data/${x}_text
    sort -t' ' -k1,1 $utt2spk_tmp > data/local/data/${x}_utt2spk
    
    ./utils/utt2spk_to_spk2utt.pl data/local/data/${x}_utt2spk > data/local/data/${x}_spk2utt || exit 1;
    
    mkdir -p data/$x
    
    cp data/local/data/${x}_wav.scp data/$x/wav.scp
    cp data/local/data/${x}_text data/$x/text
    cp data/local/data/${x}_utt2spk data/$x/utt2spk
    cp data/local/data/${x}_spk2utt data/$x/spk2utt
    
    rm ${wav_tmp}  ${utt2spk_tmp} ${text_tmp}

}

gen_set train "${train_lang}"
gen_set dev "${test_lang}"
gen_set eval "${test_lang}"
