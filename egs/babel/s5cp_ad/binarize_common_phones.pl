#! /usr/bin/perl
use Data::Dumper qw(Dumper);
my $usage = "...\n";

(@ARGV >= 2) || die "$usage";

my $nfiles = @ARGV - 1;

#print "ARGV = @ARGV\n";

## Read the global list of symbols
my %SYMG = ();
@symg = <STDIN>;
foreach (@symg) {
        $_ =~ s/[\r\n\s]+//g;
        $SYMG{$_} = 1;
}

## Read the smaller subsets of symbols from each file
shift @ARGV;
my @mat;
foreach my $j (0..$nfiles-1) {	
	my $thisf = $ARGV[$j];
	open(F, "<$thisf") || die "Could not open file $thisf";
	
	my %SYMF = ();
	while(<F>) {
		$_ =~ s/[\r\n\s]+//g;
		#@unicodes = map {ord} split(//, $_);
		#print "$_, len = ", length($_), " , ", "@unicodes", "\n";
		$SYMF{$_} = 1;		
	}
	
	my $i = 0;
	for (sort keys %SYMG) {				
		if (exists $SYMF{$_}) {
			$mat[$i][$j] = 1;
			#print "$_ = 1\n";
		} else {
			$mat[$i][$j] = 0;
			#print "$_ = 0\n";
		}
		$i++;
	}	
}

$i = 0;
for (sort keys %SYMG) {
		  printf "%-12s\t", $_;
		  my $rowsum = 0;		  
		  foreach $j (0..$nfiles-1) { 
			  $rowsum += $mat[$i][$j];
			  printf "%d\t", $mat[$i][$j];
		  }
		  printf "%d\t", $rowsum;
		  printf "\n";
		  $i++;
}

printf "%-12s\t", ";TOT";
foreach $j (0..$nfiles-1) { 
	my $colsum = 0;	
	$i = 0;	
	for (sort keys %SYMG) {
		$colsum += $mat[$i][$j];
		$i++;
	}
	printf "%d\t", $colsum;		
}
printf "\n";
#print Dumper \@mat;
