from nltk.corpus import wordnet
from itertools import chain
from utils import load_vocab_for_single_col
import logging
import argparse
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO,filename='wd.log')

def build_dictionary_wordnet(template_file, syn_file):
    synonyms = {}
    vocabulary = set()

    with open(template_file) as IN:
        for line in IN:
            l = line.strip().split()
            for word in l:
                vocabulary.add(word)

    for word in vocabulary:
            synsets = wordnet.synsets(word)
            if len(synsets) == 0:
                logging.info('Not find: %s'%word)
                continue
            synonyms[word] = set(chain.from_iterable([syn.lemma_names() for syn in synsets]))

    fout = open(syn_file,'w')
    for key in sorted(synonyms.keys()):
        for syn in synonyms[key]:
            if syn in vocabulary and key in vocabulary:
                fout.write('%s\t%s\n'%(key,syn))
    fout.close()

def filter_wordnet(wdfile):
    #filter_wordnet('../gen_data/synonyms.wd')
    fout = open(wdfile+'.filter','w')
    with open(wdfile) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) != 2 or len(arr[0]) < 4 or len(arr[1]) < 4:
                continue
            arr[1] = arr[1].lower()
            if arr[1].find('_') != -1:
                continue
            fout.write('%s\t%s\n'%(arr[0],arr[1]))
    fout.close()

    
def filter_by_freq(filename, freq=100):
    vocab = load_vocab_for_single_col('../gen_data/vocab.1b')
    fout = open(filename+'.f'+str(freq),'w')
    with open(filename) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                logging.info("error line: %s"%('\t'.join(arr)))
                continue
            if arr[0] not in vocab or arr[1] not in vocab:
                logging.info("not in vocab : %s"%('\t'.join(arr)))
                continue
            if vocab[arr[0]] < freq and vocab[arr[1]] < freq:
                logging.info("low freq: %s"%('\t'.join(arr)))
            else:
                fout.write(line)
    fout.close()

def complete_pair_dictionary(filename):
    dictionary = {}
    with open(filename) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) != 2:
                logging.info("error line: %s"%('\t'.join(arr)))
                continue
            if arr[0] not in dictionary:
                dictionary[arr[0]] = set()
            if arr[1] not in dictionary:
                dictionary[arr[1]] = set()
            dictionary[arr[0]].add(arr[1])
            dictionary[arr[1]].add(arr[0])
    fout = open(filename+'.pair','w')
    for key in sorted(dictionary.keys()):
        for word in dictionary[key]:
            fout.write('%s\t%s\n'%(key, word))
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-logs', help = 'template_file', type = str, default = './data/HDFS_clean.log')
    parser.add_argument('-syn_file',help = 'syn_file', type = str, default = './results/syns.txt')
    args = parser.parse_args()

    logs = args.logs
    syn_file =  args.syn_file

    build_dictionary_wordnet(logs, syn_file)
    print('input:', logs)
    print('syn_file', syn_file)
