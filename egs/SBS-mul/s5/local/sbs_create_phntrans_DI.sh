#!/bin/bash
# Script converts Dinka sentences to a sequence of IPA phones using a G2P Dinka FST.
# 
# The transcription file containing Dinka sentences must be in the following format:
# <utterance id> : <sentence>
# Example:
#		dinka_140817_354987-13 :  yee jeec käkä kɔc Riäk Macär cï keek ɣɔ̈n ka 
#		dinka_140817_354987-14 :  wän cï keek ke mat alɔŋ Bɛntïu raan kuath keek ee raan cɔl 
# 
# Steps:
# Step 1: Convert Windows encoding (UTF-16LE) to UTF-8 and remove any Windows special characters from the transcription file dinka.txt
# 		  > tr '[:upper:]' '[:lower:]' < <(iconv -f UTF-16LE -t UTF-8 < dinka.txt) | awk 'NR==1{sub(/^\xef\xbb\xbf/,"")}{print}' > dinka.utf8.txt
# Step 2: Remove Windows special characters from G2P FST
# 		  > dos2unix dinka_g2pmap.txt
# Step 3: Finally, generate the phone sequence
# 		  > ./sbs_create_phntrans_DI.sh  <(awk '{print $1}' dinka.utf8.txt) dinka_g2pmap.txt dinka.utf8.txt
#

if [ $# != 3 ]; then
   echo "Usage: sbs_create_phntrans_DI.sh [options] <uttlist> <g2pmap> <transcriptfile>"
   exit 1;
fi

tmpdir=/tmp/$$.tmp
mkdir -p $tmpdir

uttlist=$1
g2pmapfst=$2
transfile=$3

grep " " $g2pmapfst | cut -d' ' -f3 | sort | grep -v "eps" \
	| perl -ne 'print unless $a{$_}++' \
	| sed -e '/^$/d' -e 's///g' \
	| awk 'BEGIN {printf("eps 0\n")} {printf("%s %d\n", $0, NR)}' > $tmpdir/input.vocab

grep " " $g2pmapfst | cut -d' ' -f4 | sort | grep -v "eps" \
	| perl -ne 'print unless $a{$_}++' \
	| sed -e '/^$/d' -e 's///g' \
	| awk 'BEGIN {printf("eps 0\n")} {printf("%s %d\n", $0, NR)}' > $tmpdir/output.vocab

fstcompile --isymbols=$tmpdir/input.vocab --osymbols=$tmpdir/output.vocab $g2pmapfst > $tmpdir/g2p.fst
fstdraw    --isymbols=$tmpdir/input.vocab --osymbols=$tmpdir/output.vocab $tmpdir/g2p.fst | dot -Tpdf  > $tmpdir/G2P.fst.pdf

for id in `cat $uttlist`; do	
	idx=`basename $id .wav`
	str=`grep "$idx :" $transfile | sed 's/^[^:]*: *//g' | sed 's/ *$//g' | sed 's/,//g'` #| sed 's/[፠፡።፣፤፥፦፧፨]//g' ` #| sed 's/[፡?]//g'`
	str=`echo $str | sed -e 's/([^()]*)//g' | sed -e 's/\[[^][]*\]//g'`	
	for word in `echo $str`; do
		phonestring=`echo $word | perl local/make-acceptor.DI.pl \
				   | fstcompile --acceptor=true --isymbols=$tmpdir/input.vocab \
				   | fstarcsort --sort_type=olabel \
				   | fstcompose - $tmpdir/g2p.fst \
				   | fstshortestpath | fstprint --osymbols=$tmpdir/output.vocab \
				   | perl local/reverse_path.pl`
		if [[ -z $phonestring ]] ; then
			>&2 echo "Exception on word: $word"
		fi
		phonestring=`echo $phonestring | sed 's/eps//g'`
		echo -n $phonestring
		echo -n " "
	done
	echo
done 

rm -rf $tmpdir
