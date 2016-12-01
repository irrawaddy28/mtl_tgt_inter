#!/usr/bin/env python3

import sys

pron_dict = []
f = open(sys.argv[1])
for line in f:
    phrase, prons = line.strip().split('\t')
    pron, *_ = prons.split(', ')
    pron_dict.append((phrase, pron))
f.close()

for arg in sys.argv[2:]:
    f = open(arg)
    for line in f:
        segs = []
        while line:
            if line[0] in ['，', '。', ' ', '、', '？', '-', '.', '“', '”', ',', '：']:
                line = line[1:]
                continue
    
            matches = []
            for ph, p in pron_dict:
                if line.startswith(ph):
                    matches.append((ph, p))
    
            if matches:
                ph, p = max(matches, key=lambda t: len(t[0]))
                segs.append(p)
                line = line[len(ph):]
            else:
                print(line, file=sys.stderr)
                exit(1)
    
        print(' '.join(segs))
    f.close()
