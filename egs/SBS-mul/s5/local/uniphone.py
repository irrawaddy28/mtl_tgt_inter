#!/usr/bin/env python3

import sys

uniphone_map = {
    'ä': 'a',
    'aːː': 'aː',
    'aɪ̯': 'aɪ',
    'au': 'aʊ',
    'ɑː': 'aː',
    'ɑr': 'ɑɻ',
    'ɑʊ̯': 'aʊ',
    'ɒː': 'ɒ',
    'b̤': 'b ɦ',
    'ɓ': 'b',
    'ç': 'ɕ',
    'cː': 'c',
    'd̪': 'd',
    'd̪̤': 'd ɦ',
    'dz': 'd z',
    'dzː': 'dː z',
    'dˤ': 'd ʕ',
    'ðˤ': 'ð ʕ',
    'dˤː': 'dː ʕ',
    'ɖ': 'd ɻ',
    'ɗ': 'ɟ',
    'ɗʒ': 'ɟʝ',
    'eɪ̯': 'eɪ',
    'ɛː': 'ɛ',
    'ɛi': 'eɪ',
    'ɤ': 'ə',
    'gː': 'ɡ',
    'ɡʰ': 'ɡ ɦ',
    'ɠ': 'ɡ',
    'ħ': 'hː',
    'ħː': 'hː',
    'i̯': 'i',
    'iːː': 'iː',
    'ɨ': 'ɪ',
    'juː': 'j uː',
    'ɟː': 'ɟ',
    'ɟ̤ʝ': 'ɟʝ ɦ',
    'lːː': 'lː',
    'mb': 'm b',
    'ɱ': 'm',
    'ɱv': 'm v',
    'nd': 'n d',
    'n̠d̠ʒ': 'n dʒ',
    'nz': 'n z',
    'ɲː': 'ɲ',
    'ŋɡ': 'ŋ ɡ',
    'oː': 'oʊ',
    'øː': 'œ',
    'œy': 'œ y',
    'oʊ̯': 'oʊ',
    'pː': 'p',
    'qː': 'q',
    'ɽ': 'r',
    'ɽʰ': 'r ɦ',
    'sˤ': 's ʕ',
    'sˤː': 'sː ʕ',
    'ʂ': 'ʃ',
    't̪': 't',
    'tɕ': 'ɟʝ',
    'tɕʰ': 'c ɕ',
    't̪ʰ': 'tʰ',
    'tsː': 'ts',
    'tsʰ': 'ts',
    'tʃː': 'tʃ',
    'tˤ': 't ʕ',
    'tˤː': 'tː ʕ',
    'ʈ': 't ɻ',
    'ʈʰ': 'tʰ ɻ',
    'ʈʂ': 'tʃ',
    'ʈʂʰ': 'tʃʰ',
    'u̯': 'u',
    'ɥ': 'y',
    'vː': 'v',
    'wː': 'w',
    'wːː': 'w',
    'xː': 'x',
    'y̯': 'y',
    'yː': 'y',
    'zˤ': 'z ʕ',
    'ʒː': 'ʒ',
    'ʕː': 'ʕ',
    'θː': 'θ',
    'ɔː': 'ɔ',
    'uɪ': 'u ɪ',
    'ʷa': 'w a',    # AM to multilingual
    't\'': 'tʰ',    # AM to multilingual
    'p\'': 'pʰ',    # AM to multilingual
    'tʃ\'': 'tʃʰ',  # AM to multilingual
    'ts\'': 'ts',   # AM to multilingual
    'à̤' : 'a',		# DI to multilingual
    'à̤ː' : 'aː',	# DI to multilingual
    'e'  : 'eː',	# DI to multilingual
    'è̤'  : 'eː', 	# DI to multilingual
    'è̤ː' : 'eː',	# DI to multilingual
    'ɛ̤̆'  : 'ɛ',		# DI to multilingual
	'ĭ̤'  :	'i',	# DI to multilingual
	'ĭ̤ː' :	'iː',	# DI to multilingual
	'ò̤'  :  'o',	# DI to multilingual
	'ɔ̤̀'  :	'ɔ',	# DI to multilingual
	'd̪'	 :	'd',	# DI to multilingual
	'n̪'  :	'n',	# DI to multilingual
	't̪'	 :	't',	# DI to multilingual
}

for line in sys.stdin:
    parts = line.split()
    print(' '.join([uniphone_map.get(e, e) for e in parts]))

