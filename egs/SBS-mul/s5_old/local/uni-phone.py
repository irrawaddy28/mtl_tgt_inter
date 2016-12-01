#!/usr/bin/env python3

import sys

lang2uni = {
    'ä': 'a',
    'ai': 'aɪ',
    'au': 'aʊ',
    'ɛː': 'ɛ',
    'g': 'ɡ',
    'ɔː': 'ɔ',
    'yː': 'y',
    'ɑː': 'ɑ'
}

for line in sys.stdin:
    result = []
    for p in line.split():
        result.append(lang2uni[p] if p in lang2uni else p)
    print(' '.join(result))

