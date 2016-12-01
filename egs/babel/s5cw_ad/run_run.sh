#!/bin/bash

: << 'COMMENT'
declare -a configs=("bengali" "conf/lang/103-bengali-limitedLP.official.conf"
"assamese" "conf/lang/102-assamese-limitedLP.official.conf"
"cantonese" "conf/lang/101-cantonese-limitedLP.official.conf"
"pashto" "conf/lang/104-pashto-limitedLP.official.conf"
"tagalog" "conf/lang/106-tagalog-limitedLP.official.conf"
"turkish" "conf/lang/105-turkish-limitedLP.official.conf"
"vietnamese" "conf/lang/107-vietnamese-limitedLP.official.conf"
"haitian" "conf/lang/201-haitian-limitedLP.official.conf"
"lao" "conf/lang/203-lao-limitedLP.official.conf"
"zulu" "conf/lang/206-zulu-limitedLP.official.conf"
"tamil" "conf/lang/204-tamil-limitedLP.official.conf")
COMMENT

declare -a configs=("cantonese" "conf/lang/101-cantonese-limitedLP.official.conf"
"assamese" "conf/lang/102-assamese-limitedLP.official.conf"
"bengali" "conf/lang/103-bengali-limitedLP.official.conf"
"pashto" "conf/lang/104-pashto-limitedLP.official.conf"
"turkish" "conf/lang/105-turkish-limitedLP.official.conf"
"tagalog" "conf/lang/106-tagalog-limitedLP.official.conf"
"vietnamese" "conf/lang/107-vietnamese-limitedLP.official.conf"
"haitian" "conf/lang/201-haitian-limitedLP.official.conf"
"lao" "conf/lang/203-lao-limitedLP.official.conf"
"zulu" "conf/lang/206-zulu-limitedLP.official.conf"
"tamil" "conf/lang/204-tamil-limitedLP.official.conf")
nlangs=${#configs[@]}
echo "Number of languages = $(( nlangs / 2 ))"
nlangs=$(( nlangs - 1)) # array index starts from 0

rm -rf data exp plp
for k in $(seq 0 2 $nlangs); do
	lang=${configs[$k]}
	conf=${configs[$(( k + 1 ))]}
	echo "lang = $lang, conf = $conf"
	./run-1-main.sh --tri5-only "true" $conf;
	./run-4-anydecode.sh --skip-kws true --tri5-only true --dir dev10h.uem;
	local/score.sh --cmd "run.pl" data/dev10h.uem exp/tri5/graph exp/tri5/decode_dev10h.uem
	rm -rf exp_all/$lang
	mkdir -p exp_all/$lang
	mv data exp plp exp_all/$lang	
done

exit 0;
