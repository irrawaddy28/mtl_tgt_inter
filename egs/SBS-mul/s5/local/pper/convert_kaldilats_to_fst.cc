// Author: Preethi Jyothi
// November 2011

// Common STL header files
#include <algorithm>
#include <functional>
#include <iostream>
#include <locale>
#include <string>
#include <vector>
#include <time.h>

// KALDI header files
#include "util/kaldi-io.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "fstext/lattice-utils-inl.h"
#include "fstext/lattice-utils.h"

// OpenFST header files
#include <fst/compose.h>
#include <fst/fstlib.h>
//#include "./my_utils.h"

using namespace std;
using namespace fst;
using namespace kaldi;

int main(int argc, char** argv) {

  try {
  using namespace kaldi;

  typedef ArcTpl<CompactLatticeWeight> A;
  typedef StdArc SA;
  typedef SA::Weight SWt;
  typedef A::StateId S;
  typedef A::Weight W;
  typedef kaldi::int32 int32;

  if (argc <= 2) {
    cerr << "Usage : cmd <lattice-rspecifier> <output directory to write fsts>" << endl;
    exit(0);
  }

  std::string output_dir(argv[argc-1]);
  std::string lattice_rspecifier(argv[argc-2]);

  fst::VectorFst<LatticeArc> lfst;
  BaseFloat acoustic_scale = 0.083333;
  
  SequentialCompactLatticeReader lreader(lattice_rspecifier);
 
  int fst_num = 0;
  for(; !lreader.Done(); lreader.Next()) {
    string key = lreader.Key();
    const CompactLattice &clat1 = lreader.Value();
    CompactLattice &clat = const_cast<CompactLattice&>(clat1);
	Lattice lat;

    if(acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);

    RemoveAlignmentsFromCompactLattice(&clat);
    fst::StdVectorFst std_fst, std_fst_out;
    char t_str[200];
    sprintf(t_str, "%s/%s.fst", output_dir.c_str(), key.c_str());
    fst_num++;

	fst::ConvertLattice(clat, &lat);
	fst::ConvertLattice(lat, &std_fst);
	fst::Project(&std_fst, fst::PROJECT_OUTPUT);
	fst::RmEpsilon(&std_fst);

    std_fst.Write(t_str);
  }

  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
