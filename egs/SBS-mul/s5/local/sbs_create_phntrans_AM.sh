#!/bin/bash
# Script converts Amharic sentences to a sequence of IPA phones using a G2P Amharic FST.
# 
# The transcription file containing Amharic sentences must be in the following format:
# <utterance id> : <sentence>
# Example:
# 		amharic_140908_359445-30 : ደህንነት ሰላማዊ እና መልካም አዲስ አመትን እመኛለሁ [music]
#  		amharic_140908_359445-3 : ለመላው ኢትዮጲያውያን ማህበረሰብ የደስታ ሰላማዊ እና የደህንነት 
# 
# Steps:
# Step 1: Convert Windows encoding (UTF-16LE) to UTF-8 and remove any Windows special characters from the transcription file amharic.txt
# 		  > iconv -f UTF-16LE -t UTF-8 < amharic.txt | tr -d '\r' | awk 'NR==1{sub(/^\xef\xbb\xbf/,"")}{print}' > amharic.utf8.txt
# Step 2: Remove Windows special characters from G2P FST
#         > dos2unix amharic_g2pmap.txt
# Step 3: Finally, generate the phone sequence
#		  > ./sbs_create_phntrans_AM.sh  <(awk '{print $1}' amharic.utf8.txt) amharic_g2pmap.txt amharic.utf8.txt
#

if [ $# != 3 ]; then
   echo "Usage: sbs_create_phntrans_AM.sh [options] <uttlist> <g2pmap> <transcriptfile>"
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

# There are 9 Amharic punctuations symbols: ፠፡።፣፤፥፦፧፨
# The corresponding Unicode code-points for punctuations are \u1360-\u1368
# We can convert these to utf-8 encoding using the python command below
# CHARS=$(python -c 'print u"\u1360\u1361\u1362\u1363\u1364\u1365\u1366\u1367\u1368".encode("utf8")')
# And then run a sed on the utf-8 encodings to remove punctuations from raw text: sed 's/['"$CHARS"']//g'
for id in `cat $uttlist`; do	
	idx=`basename $id .wav`	
	# sed 's/[፡?]//g'` does not work. Instead sed -e 's/፡//g' -e 's/?//g' works !!
	str=`grep "$idx :" $transfile | sed 's/^[^:]*: *//g' | sed 's/ *$//g' | sed -e 's/፡//g' -e 's/?//g' `
	str=`echo $str | sed -e 's/([^()]*)//g' | sed -e 's/\[[^][]*\]//g'`	
	for word in `echo $str`; do
		phonestring=`echo $word | perl local/make-acceptor.pl \
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
