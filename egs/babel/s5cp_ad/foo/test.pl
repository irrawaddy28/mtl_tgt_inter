#! /usr/bin/perl 

$textFileName = "text";
$vocabFile = "data/local/lexicon.txt";
$cmd="tmp=`mktemp` ; cp $textFileName \$tmp && perl utils/transnorm.pl --map-to-col -1  $textFileName $vocabFile > \$tmp ; echo \$tmp";
print "cmd = $cmd\n";
system($cmd);

print "tmp = $tmp\n";
