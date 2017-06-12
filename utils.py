import collections
import numpy as np
from nltk import word_tokenize
import random

#TODO: Handle train/test/validation data
    #TODO: Datashuffling 2/3, 1/6, 1/6
    #TODO: Store them separately
#TODO: Write _make_batches
#TODO: Define method to input <EOS> tokens
#TODO: Global variable called numval batches, and num_testbatches
class DataLoader(object):
    def __init__(self, filename, batch_size, num_steps):
        '''
        :param filename: path to corpus
        :param batch_size: size of minibatches
        :return: object with useful tools for handling corpus
        '''
        self.window=5
        self.num_steps = num_steps
        filereader = open(filename, 'r')
        lines=[]
        counts=Counter()
        #extract sentences
        for line in filereader.readlines():
            tokenized = word_tokenize(line.lower())
            counts.update(tokenized)
            lines.append(tokenized)
        #consolidate vocabulary
        for w in counts:
            if counts[w] > 3:
                self.vocab.append(w)
        self.vocab=[]
        self.V=len(vocab)
        self.vocab.append("<UNK>")
        self.vocab.append("#")

        #filter low frequqency words
        for line in lines[]:
            for w in line:
                if w not in self.vocab:
                    w="<UNK>"
        #set a random seed to keep the split
        random.seed(31415)
        #shuffle the lines and split them in train test and validation
        random.shuffle(lines)
        num_sents= len (lines)
        train_sents, test_sents, val_sents = lines[:floor(numsents*0.7)] ,lines[floor(numsents*0.7):floor(numsents*0.85)],lines[floor(numsents*0.85):]

        self.idx2word = dict(enumerate(self.vocab))
        self.word2idx = {value:key for (key, value) in self.idx2word.items()}

        # A dictionary that contains batch information as a triple consisting of:
        # The list of batch matrices
        # The list of label vectors
        # The index where the next batchshould be extracted
        self.all_batch_triples={}

        #populate the triplets via the _make_batches method
        self.allbatch_triples["train"]=  self._make_batches(batch_size, train_sents)
        self.all_batch_triples["test"] = self._make_batches(batch_size, test_sents)
        self.all_batch_triples["validation"] = self._make_batches(batch_size, val_sents)


        self.num_batches = 0

    def _cast_index(self):
        '''
        :param sequence: input list of tokens to be cast as their indexes
        :return: list of same length of sequence of indices corresponding to tokens
        '''
        return [self.word2idx[w] for w in self.data if w in self.word2idx]

    def _make_batches(self, batch_size, sentences, shuffle=False):
        '''
        :param batch_size: size of minibatches
        :return: object consisting of all batches in corpus to be loaded with _load_batches
        '''
        n=self.window
        batches=[]
        labels=[]
        for sentence in sentences:
            current_batch=[]
            current_labels=[]
            #Pad sentence with out of sentence tokens
            sentence=["#"]*(n-1) +sentence + ["#"]*(n-1)
            # compile the matrix
            for begin_ind in range(len(sentence)-n-1):
                chunk= sentence[begin_ind:begin_ind+n]
                chunk_ids=[ word2idx[word] for word in chunk]
                current_batch.append(chunk_ids)
                current_labels.append(word2idx[sentence[begin_ind+n+1]])
                if len(current_batch) == batch_size:
                    batches.append(current_batch)
                    labels.append(current_labels)
                    current_labels=[]
                    current_batch=[]
        return (batches, labels, 0)

    def _load_next_batch(self, kind="train"):
        if self.all_batch_triples[kind][2] > len(self.all_batch_triples[kind][0]){
            #start a new epoch
            self.all_batch_triples[kind][2]=0
            return False
        }
        return self.all_batch_triples[kind][0][self.all_batch_triples[kind][2]], self.all_batch_triples[kind][1][self.all_batch_triples[kind][2]]

class SmallConfig:
    pass

class MediumConfig:
    pass

class LargeConfig:
    pass
