#!/usr/bin/perl
#
# LAB: JSALT Workshop, June 26th, 2015
# Building linear-chain acceptors
# a list of spellings

$argerr = 0;
$odir = "";
$vocab = "";
$sil = "";
$l2tag = "";
$l2flag = 0;

while ((@ARGV) && ($argerr == 0)) {
	if($ARGV[0] eq "--odir") {
		shift @ARGV;
		$odir = shift @ARGV;
	} elsif($ARGV[0] eq "--vocab") {
		shift @ARGV;
		$vocab = shift @ARGV;
	} elsif($ARGV[0] eq "--sil") {
		shift @ARGV;
		$sil = shift @ARGV;
	} elsif($ARGV[0] eq "--l2tag") {
		shift @ARGV;
		$l2tag = shift @ARGV;	
	} else {
		$argerr = 1;
	}
}

$argerr = 1 if ($odir eq "" || $vocab eq "");

if($argerr == 1) {
	print "Usage: ./spell2fst.pl --odir <output directory to store acceptors> --vocab <vocabulary file>\n";
	print "Input: Test file\n";
	exit 1;
}

system("mkdir -p $odir");

print "Computing acceptors...\n";
$count = 0;
while(<STDIN>) {
	chomp;
	@fields = split(/\s+/);
	$id = $fields[0]; #$ref = $fields[1]; #$misspell = $fields[2];	
	shift @fields;	
	$filename = "$odir/$id.txt";
	$binfile = "$odir/$id.fst";
	open(FILE, ">$filename");
	$state = 0;
	#print "id = $id: @fields ";
	foreach my $ref (@fields) {
		#print "ref = $ref ";		
		if ($ref =~ /$sil/) {			
			print FILE "$state\t",$state+1,"\t$ref\n"; 
		} else {
			
			if ($ref =~ /$l2tag(.*)$l2tag/) {
				$l2flag = 1;
				$ref = $1;
				print FILE "$state\t",$state+1,"\t$l2tag\n";
			}
			
			@lets = split(//,$ref);		
			for($l = 0; $l <= $#lets; $l++) {
				print FILE "$state\t",$state+1,"\t$lets[$l]\n"; #one letter per arc
				$state++;
			}
			
			if ($l2flag == 1) {
				print FILE "$state\t",$state+1,"\t$l2tag\n";
				$l2flag = 0;
			}
		}
    }
    print FILE "$state\n"; #final state
	close(FILE);	
	system("fstcompile --acceptor=true --isymbols=$vocab $filename $binfile"); 
	$count++;
	print "Finished building $count acceptors\n" if($count % 100 == 0);
}

