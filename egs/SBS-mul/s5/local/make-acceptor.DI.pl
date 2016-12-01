#!/usr/bin/perl

binmode STDIN, ':utf8';
binmode STDOUT, ':utf8';

while(<STDIN>){
	chomp;
	next if($_ =~ /^\s*$/);
	$_ =~ s/\x{0254}\x{0308}/ 11 /g; #handling character clusters first
	$_ =~ s/\x{025b}\x{0308}/ 22 /g;
	$_ =~ s/([\x{0254}\x{00f6}\x{00ef}\x{00e4}\x{00eb}\x{025b}\x{0194}\x{014b}\x{0263}])/ $1 /g;
	$_ =~ s/([a-zA-Z\s])/ $1 /g; #handling all alpha characters (after handing all the ext. Latin characters)
	$_ =~ s/ 11 / \x{0254}\x{0308} /g; #replacing character clusters
	$_ =~ s/ 22 / \x{025b}\x{0308} /g;
	$_ =~ s/\s+/ /g; $_ =~ s/^\s+//g; $_ =~ s/\s+$//g;
	@lets = split(/ /);
	$state = 0;
	for($l = 0; $l <= $#lets; $l++) {
		print "$state\t",$state+1,"\t$lets[$l]\n";
		$state++;
	}
	print "$state\n";
}

