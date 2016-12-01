##Paul Hager
## Native Transcription with numerals to Native Transcription with written out numerals
## 7.9.15
## sbs_data_preprocess_UR.py

import string
import sys
import codecs

## Script should be run with the following options:
## python script.py <list of utterance IDs> <numeral dictionary> <data directory> <output directory>


def main(argv):
        utterance_list = argv[0]
	numeral_dictionary_file = argv[1]
	data_dir = argv[2]
	OUTPUT_DIRECTORY = argv[3]
	utterance_list = open(utterance_list, 'r')

	## codecs code is borrowed from Spencer Green
        ## See more at: http://www.spencegreen.com/2008/12/19/python-arabic-unicode/#sthash.BkRHT3pK.dpuf

	#Create a numeral-to-Urdu-script dictionary
	numeral_graph_dict = {}
	numeral_graph_dict_filename = numeral_dictionary_file
	FILE = codecs.open(numeral_graph_dict_filename,'r',encoding='utf-8')
	for line in FILE:
		items = line.split()
		numeral_graph_dict[items[0]] = items[1]

	#Rewrite the transcripts with numerals written out in Urdu.
	for utterance in utterance_list:
                utterance_ID = utterance[:-5]
                filename = data_dir + '/' + utterance_ID + '.txt'
		IN_FILE = codecs.open(filename,'r', encoding='utf-8')
		words = IN_FILE.readline().split()
		buffer = ''
		
		for word in words:
			new_word = ''
			for char in word:
				if char in numeral_graph_dict.keys():
					new_word += numeral_graph_dict[char]
				else:
					new_word += char
			buffer += new_word + ' '
		
		out_filename = OUTPUT_DIRECTORY + '/' + utterance_ID + '.txt'
		OUT_FILE = codecs.open(out_filename,'w',encoding='utf-8')
		OUT_FILE.write(buffer)
		OUT_FILE.close()

if __name__ == "__main__":
        main(sys.argv[1:])

