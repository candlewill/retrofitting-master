import argparse
import gzip
import math
import numpy
import re
import sys
import gensim
from copy import deepcopy

isNumber = re.compile(r'\d+.*')


def norm_word(word):
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()


''' Read all the word vectors and normalize them '''


# def read_word_vecs(filename):
#     wordVectors = {}
#     if filename.endswith('.gz'):
#         fileObject = gzip.open(filename, 'r')
#     else:
#         fileObject = open(filename, 'r')
#
#     for line in fileObject:
#         line = line.strip().lower()
#         word = line.split()[0]
#         wordVectors[word] = numpy.zeros(len(line.split()) - 1, dtype=float)
#         for index, vecVal in enumerate(line.split()[1:]):
#             wordVectors[word][index] = float(vecVal)
#         ''' normalize weight vector '''
#         wordVectors[word] /= math.sqrt((wordVectors[word] ** 2).sum() + 1e-6)
#
#     sys.stderr.write("Vectors read from: " + filename + " \n")
#     return wordVectors

def read_word_vecs(filename):
    model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=False)
    wordVectors = dict()
    for key in model.vocab.keys():
        v = numpy.array(model[key])
        wordVectors[str(key).lower()] =v/(math.sqrt((v**2).sum()+1e-6))

    # for line in fileObject:v
    #     line = line.strip().lower()
    #     word = line.split()[0]
    #     wordVectors[word] = numpy.zeros(len(line.split()) - 1, dtype=float)
    #     for index, vecVal in enumerate(line.split()[1:]):
    #         wordVectors[word][index] = float(vecVal)
    #     ''' normalize weight vector '''
    #     wordVectors[word] /= math.sqrt((wordVectors[word] ** 2).sum() + 1e-6)

    sys.stderr.write("Vectors read from: " + filename + " \n")
    return wordVectors

''' Write word vectors to file '''


def print_word_vecs(wordVectors, outFileName):
    sys.stderr.write('\nWriting down the vectors in ' + outFileName + '\n')
    outFile = open(outFileName, 'w', encoding='utf-8')
    for word, values in wordVectors.items():
        outFile.write(word + ' ')
        for val in wordVectors[word]:
            outFile.write('%.4f' % (val) + ' ')
        outFile.write('\n')
    outFile.close()


''' Read the PPDB word relations as a dictionary '''


def read_lexicon(filename, wordVecs):
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon


''' Retrofit word vectors to a lexicon '''


def retrofit(wordVecs, lexicon, numIters):
    newWordVecs = deepcopy(wordVecs)
    wvVocab = set(newWordVecs.keys())
    loopVocab = wvVocab.intersection(set(lexicon.keys()))
    for it in range(numIters):
        # loop through every node also in ontology (else just use data estimate)
        for word in loopVocab:
            wordNeighbours = set(lexicon[word]).intersection(wvVocab)
            numNeighbours = len(wordNeighbours)
            # no neighbours, pass - use data estimate
            if numNeighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            newVec = numNeighbours * wordVecs[word]
            # loop over neighbours and add to new vector (currently with weight 1)
            for ppWord in wordNeighbours:
                newVec += newWordVecs[ppWord]
            newWordVecs[word] = newVec / (2 * numNeighbours)
    return newWordVecs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
    parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
    parser.add_argument("-o", "--output", type=str, help="Output word vecs")
    parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
    args = parser.parse_args()

    wordVecs = read_word_vecs(args.input)
    lexicon = read_lexicon(args.lexicon, wordVecs)
    numIter = int(args.numiter)
    outFileName = args.output

    ''' Enrich the word vectors using ppdb and print the enriched vectors '''
    print_word_vecs(retrofit(wordVecs, lexicon, numIter), outFileName)

'''
Examples script:

python retrofit.py -i D:\Word_Embeddings\English\glove.6B\glove.6B.300d.txt -l lexicons\wordnet-synonyms+.txt -n 10 -o D:\Wor
d_Embeddings\English\glove.6B\GloVe_out_vec_file.txt

python retrofit.py -i D:\Word_Embeddings\English\GoogleNews-vectors-negative300.bin -l lexicons\wordnet-synonyms+.txt -n 10 -o D:\Wor
d_Embeddings\English\word2vec_out_vec_file.txt

python retrofit.py -i "D:\Word_Embeddings\English\simplified_word2vecs (with-header).txt" -l lexicons\wordnet-synonyms+.txt -n 10 -o D:\Wor
d_Embeddings\English\word2vec_out_vec_file.txt
'''