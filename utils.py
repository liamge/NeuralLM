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
            tokenized.append('<EOS>')
            lines.append(tokenized)
        #consolidate vocabulary
        for w in counts:
            if counts[w] > 3:
                self.vocab.append(w)
        self.vocab=[]
        self.V=len(vocab)
        self.vocab.append("<UNK>")

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
        self.train_sents, self.test_sents, self.val_sents = lines[:floor(numsents*0.7)] ,lines[floor(numsents*0.7):floor(numsents*0.85)],lines[floor(numsents*0.85):]

        self.idx2word = dict(enumerate(self.vocab))
        self.word2idx = {value:key for (key, value) in self.idx2word.items()}


        self.train_batches, self.train_labels = self._make_batches(batch_size, train_sents)
        self.test_batches, self.test_labels = self._make_batches(batch_size, test_sents)
        self.val_batches, self.val_labels = self._make_batches(batch_size, val_sents)


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
        for sentence in sentences:
            current_batch=[]
            for begin_ind in range(len(sentence)-n):
                chunk= sentence[begin_ind:begin_ind+n]
                chunk_ids=[ word2idx[word] for word in chunk]
                current_batch.append(chunk_ids)
                if len(current_batch)== batch_size:
                    batches.append(current_batch)
                    current_batch=[]
        current_batch= current_batch
        return batches
    def _load_next_batch(self, kind="train"):
        # If last batch reset to new epoch
        #TODO: Distinguish between the different kinds of batches
        if self.num_batches == len(self.batches)-1:
            temp = self.num_batches
            self.num_batches = 0
            return self.batches[temp], self.labels[temp]
        else:
            temp = self.num_batches
            self.num_batches += 1
            return self.batches[temp], self.labels[temp]

class SmallConfig:
    pass

class MediumConfig:
    pass

class LargeConfig:
    pass
