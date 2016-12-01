#!/bin/bash -u
. ./path.sh
set -o errexit
set -o pipefail

tmpdir=local/tmp/DT
rawtransdir="/export/ws15-pt-data/data/transcripts/matched/dutch"
rawtransscp="$tmpdir/trans_raw.scp"
normtransscp="$tmpdir/trans_norm.scp"
engfsttxt="/export/ws15-pt-data/data/misc/engtag.fst.txt"
engfstbin="$tmpdir/engtag.fst.bin"
g2pfsttxt="conf/dutch/DT_G2P.fst.txt"
g2pfstbin="$tmpdir/DT_G2P.fst.bin"
tagdict="conf/dutch/numbers_dutch_pron.ipa.dict"
epssym="eps" # epsilon symbol used in FSTs
silword="<silence>"
l2tag="<EN>"

mkdir -p $tmpdir 

if [[ 0 == 1 ]]; then
# extract raw transcriptions and save them in scp format
ls -1 /export/ws15-pt-data/data/transcripts/matched/dutch|sed 's:.txt::' > $tmpdir/uttid.txt
perl local/sbs_extract_trans_DT.pl --d $rawtransdir --e "txt"   $tmpdir/uttid.txt > $rawtransscp

# tag special words with <EN>
# cat $rawtransscp |cut -d' ' -f2-|grep -iE "[0-9]" > conf/dutch/numbers_eng_pron.cmu.dict
perl local/transnorm.pl <(cat conf/dutch/numbers_eng_pron.cmu.dict|tr '[:upper:]' '[:lower:]') conf/eng/eng-ARPA2IPA.txt > conf/dutch/numbers_eng_pron.ipa.dict
# After this, I manually modified  the Eng pronunciations in "numbers_eng_pron.ipa.dict" to make them Dutch pronunciations which are
# saved in conf/dutch/numbers_dutch_pron.ipa.dict <-- this is the one we want to use 

# normalize transcriptions (don't normalize L2 words which are tagged using $l2tag)
transscptmp="$(mktemp)"
local/sbs_norm_trans_DT.pl --i $rawtransscp --sil "<silence>" --l2tag $l2tag > $transscptmp

# if words from "$tagdict" are present in transcription, then these words should be expanded to their phone sequences
# and tagged using L2 tags!! E.g. 4 -> <EN>v i r<EN>
local/sbs_norm_trans_DT.pl -r 1 --i $transscptmp  --l2tag "<EN>" --tagdict $tagdict > $normtransscp
fi

# create fst for English (L1 = Dutch, L2 = English) 
echo "Creating FST for English: vocab=$tmpdir/EN.vocab, fst=$engfstbin"
cat $engfsttxt| awk '{print $3}'|sed '/^ *$/d'|sort -u| awk 'BEGIN {ind = 0}; {print $1, ind; ind++}' >> $tmpdir/EN.vocab
fstcompile --isymbols=$tmpdir/EN.vocab --osymbols=$tmpdir/EN.vocab $engfsttxt $engfstbin
echo -e "Done\n"

# create fst for G2P Dutch
echo "Creating G2P FST for Dutch: grapheme_vocab=$tmpdir/DT.ortho.vocab, phone_vocab = $tmpdir/DT.phone.vocab, fst=$g2pfstbin"
echo "$epssym  0" > $tmpdir/DT.ortho.vocab
cat $g2pfsttxt|awk '{print $3}'|sed '/^ *$/d'|sort -u| grep -vi "$epssym"| awk 'BEGIN {ind = 1}; {print $1, ind; ind++}' >> $tmpdir/DT.ortho.vocab

echo "$epssym  0" > $tmpdir/DT.phone.vocab
cat $g2pfsttxt|awk '{print $4}'|sed '/^ *$/d'|sort -u| grep -vi "$epssym"| awk 'BEGIN {ind = 1}; {print $1, ind; ind++}' >> $tmpdir/DT.phone.vocab

fstcompile --isymbols=$tmpdir/DT.ortho.vocab --osymbols=$tmpdir/DT.phone.vocab $g2pfsttxt $g2pfstbin
#fstdraw  --isymbols=$tmpdir/DT.ortho.vocab --osymbols=$tmpdir/DT.phone.vocab $g2pfstbin | dot -Tpdf  > local/tmp/DT/DT_G2P.fst.pdf
echo -e "Done\n"

# create fsa for Dutch sentences (some sentences have L2 words)
echo "Creating FSA for dutch (mixed with L2) sentences"
mkdir -p $tmpdir/fsa
rm -rf $tmpdir/fsa/* 

cat $engfsttxt $g2pfsttxt <(echo "1 1 $l2tag $l2tag")|awk '{print $3}'|sed '/^ *$/d'|sort -u|grep -vi "$epssym"\
awk 'BEGIN {ind = 1}; {print $1, ind; ind++}' >> $tmpdir/trans.vocab
local/spell2fst.pl --odir $tmpdir/fsa --vocab $tmpdir/DT.ortho.vocab --sil "<silence>" --l2tag $l2tag < $normtransscp
exit 1;



# create union fst (U) = Union of G2P Dutch, Digits, English
fstunion $g2pfstbin $engfstbin | fstclosure - > $tmpdir/U.fst

# modify U.fst to add <EN> tags

# compose Dutch fsa with U.fst and print best path

# 
