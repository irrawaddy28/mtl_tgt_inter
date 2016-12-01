#!/usr/bin/perl

binmode STDIN, ':utf8';
binmode STDOUT, ':utf8';

while(<STDIN>){
	chomp;
	next if($_ =~ /^\s*$/);
	@lets = split(//);
	$state = 0;
	for($l = 0; $l <= $#lets; $l++) {
		print "$state\t",$state+1,"\t$lets[$l]\n";
		$state++;
	}
	print "$state\n";
}

