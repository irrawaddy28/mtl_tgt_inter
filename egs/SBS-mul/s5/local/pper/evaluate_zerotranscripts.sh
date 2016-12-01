#!/bin/bash
# Evaluating "expected" accuracies wrt PTs, computed on
# on an evaluation set using decoded lattices on a target 
# language with no native transcriptions

finished="n"
loglikeoption="n"
weighted="n"

while [ $finished == "n" ]; do
	finished="y"
	[ "$1" == "--l" ] && loglikeoption="y" && shift && finished="n";
	[ "$1" == "--w" ] && weighted="y" && shift && finished="n";
done

if [[ $# -ne 5 ]]; then
	echo "Usage: script [--l = also compute loglikelihood ratios] [--w = weighted edit distance] <lattice rspecifier> <directory with PTs> <list of utterances> <phones vocab file> <output directory to write out FSTs (from ASR lattices)>";
	echo "Sample call:./evaluate_zerotranscripts.sh \"ark:gunzip -c /tmp/all.adapted.lats.gz|\" /export/ws15-pt-data2/data/pt-stable-7/held-out-SW data/lists/russian/dev data/RS/lang/phones.txt lat_fsts" 
	exit 1
fi

echo "$0 $@"
echo "weighted=$weighted"
PTprunewt=1

latrspecifier=$1
ptdir=$2
testids=$3
phonevocab=$4
odir=$5

# Converts kaldi lattices into the OpenFst format and writes
# out FSTs to odir
mkdir -p $odir
./convert_kaldilat2fst "$latrspecifier" "$odir"
>&2 echo "Finished converting lattices to FSTs";

tmpdir=tmp/$$.dir
mkdir -p $tmpdir

# Create an edit distance FST with insertion/deletion/sub costs=1
# using the phone vocabulary specified in $phonevocab
editfst=$tmpdir/edit.fst
perl create-editfst.pl < $phonevocab | fstcompile - > $editfst 

loglike=0;
eds=0;
numutts=`wc -l $testids | cut -d' ' -f1`
for uttid in `cat $testids`; do
	>&2 echo "Utterance $uttid"	
	uttid=`basename $uttid .wav`
	latfst=$odir/${uttid}.fst
	ptfst=$ptdir/${uttid}.lat.fst
	# Find shortest path within lattice and its score
	fstshortestpath $latfst > $tmpdir/path.${uttid}.fst

	# Create an unweighted acceptor using the best path from the lattice
	fstprint $tmpdir/path.${uttid}.fst | cut -f1-4 | perl -a -n -e 'chomp; if($#F <= 2) { print "$F[0]\n"; } else { print "$_\n"; }' | fstcompile - > $tmpdir/lat.${uttid}.fst
	if [[ $weighted == "y" ]]; then
		cp $ptfst $tmpdir/pt.pruned.${uttid}.fst
	else
		fstprune --weight=$PTprunewt $ptfst | fstprint | cut -f1-4 | perl -a -n -e 'chomp; if($#F <= 2) { print "$F[0]\n"; } else { print "$_\n"; }' | fstcompile - > $tmpdir/pt.pruned.${uttid}.fst
	fi

	# Create an unweighted acceptor using the best path from the PT for uttid
	fstshortestpath $ptfst | fstprint | cut -f1-4 | perl -a -n -e 'chomp; if($#F <= 2) { print "$F[0]\n"; } else { print "$_\n"; }' | fstcompile - > $tmpdir/pt.${uttid}.fst

	if [[ $loglikeoption == "y" ]]; then
		# Compute the score from the best path in the lattice
		# and when restricted to the best path from PT
		bestlatscore=`fstshortestdistance --reverse $tmpdir/path.${uttid}.fst | head -1 | cut -f2`

		reflatscore=`fstarcsort --sort_type=olabel $tmpdir/pt.${uttid}.fst | fstcompose - $editfst | fstarcsort --sort_type=olabel | fstcompose - $latfst | fstshortestdistance --reverse | head -1 | cut -f2`

		# Compute the difference between the scores from the best path and the "PT reference" path in the lattice
		loglikeratio=`echo "($reflatscore - $bestlatscore)" | bc -l`
		
		loglike=`echo "($loglike + $loglikeratio)" | bc -l`
		
		echo "Accumulated loglikelihoodratio $loglike bestlatscore=$bestlatscore and reflatscore=$reflatscore";
	fi

	# Computes the edit distance between the best path and the "PT reference" path in the lattice
	editscore=`fstarcsort --sort_type=olabel $editfst | fstcompose - $tmpdir/lat.${uttid}.fst | fstarcsort --sort_type=ilabel - | fstcompose $tmpdir/pt.pruned.${uttid}.fst - | fstshortestdistance --reverse | head -1 | cut -f2`
	#echo "fstarcsort --sort_type=olabel $editfst | fstcompose - $tmpdir/lat.${uttid}.fst | fstarcsort --sort_type=ilabel - | fstcompose $tmpdir/pt.pruned.${uttid}.fst - | fstshortestdistance --reverse "
	echo "edit score ($uttid) = $editscore"
	eds=`echo "($eds + $editscore)" | bc -l`
	echo "Accumulated edit distances $eds";
done

finaled=`echo "$eds / $numutts" | bc -l` 
finalmargin=`echo "$loglike / $numutts" | bc -l` 

>&2 echo "Avg. edit distance = $finaled and avg. loglikelihood ratio = $finalmargin over $numutts utterances in the data set $testids"

rm -rf $(dirname $tmpdir)
