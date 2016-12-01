#!/bin/bash


#declare -a configs=("cantonese" "conf/lang/101-cantonese-limitedLP.official.conf"
#"assamese" "conf/lang/102-assamese-limitedLP.official.conf"
#"bengali" "conf/lang/103-bengali-limitedLP.official.conf"
#"pashto" "conf/lang/104-pashto-limitedLP.official.conf"
#"turkish" "conf/lang/105-turkish-limitedLP.official.conf"
#"tagalog" "conf/lang/106-tagalog-limitedLP.official.conf"
#"vietnamese" "conf/lang/107-vietnamese-limitedLP.official.conf"
#"haitian" "conf/lang/201-haitian-limitedLP.official.conf"
#"lao" "conf/lang/203-lao-limitedLP.official.conf"
#"zulu" "conf/lang/206-zulu-limitedLP.official.conf"
#"tamil" "conf/lang/204-tamil-limitedLP.official.conf")

#nlangs=${#configs[@]}
#echo "Number of languages = $(( nlangs / 2 ))"
#nlangs=$(( nlangs - 1)) # array index starts from 0

#rm -rf data exp plp
#for k in $(seq 0 2 $nlangs); do
	#lang=${configs[$k]}
	#conf=${configs[$(( k + 1 ))]}
	#echo "lang = $lang, conf = $conf"
	#./run-1-main.sh --tri5-only "true" $conf;
	#./run-4-anydecode.sh --skip-kws true --tri5-only true --dir dev10h.uem;
	#local/score.sh --cmd "run.pl" data/dev10h.uem exp/tri5/graph exp/tri5/decode_dev10h.uem
	#rm -rf exp_all/$lang
	#mkdir -p exp_all/$lang
	#mv data exp plp exp_all/$lang	
#done

declare -a configs=("CNT" "ASM" "BNG" "PSH" "TUR" "TGL" "VTN" "HAI" "LAO" "ZUL" "TAM")
#declare -a configs=("CNT")
nlangs=${#configs[@]}
echo "Number of languages = $nlangs"
nlangs=$(( nlangs - 1 )) # array index starts from 0

for k in $(seq 0 $nlangs); do
	#(
	  lang=${configs[$k]}	
	  echo "Starting CD-GMM-HMM training on language = $lang"
	  ./run-1-main_langs.sh --tri5-only "true" $lang
	  #./run-4-anydecode_langs.sh --skip-kws true --tri5-only true --dir dev10h.uem $lang
	  echo "Completed CD-GMM-HMM training on language = $lang"
	#) &
done
wait;

exit 0;
