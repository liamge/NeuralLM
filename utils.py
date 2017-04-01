import collections

#TODO: Handle train/test/validation data
#TODO: Write _make_batches
#TODO: Define method to input <EOS> tokens

class DataLoader(object):
    def __init__(self, filename, batch_size, num_steps):
        '''
        :param filename: path to corpus
        :param batch_size: size of minibatches
        :return: object with useful tools for handling corpus
        '''
        self.num_steps = num_steps
        f = open(filename, 'r').read()
        tokenized = f.lower().split()
        counts = collections.Counter(tokenized)
        self.data = []
        for w in tokenized:
            if counts[w] > 3:
                self.data.append(w)
            else:
                self.data.append('<UNK>')
        self.V = set(self.data)
        self.N = len(self.data)
        self.idx2word = dict(zip(enumerate(self.V)))
        self.word2idx = {value:key for (key, value) in self.idx2word.items()}
        self.batches, self.labels = _make_batches(batch_size)
        self.num_batches = 0

    def _cast_index(self):
        '''
        :param sequence: input list of tokens to be cast as their indexes
        :return: list of same length of sequence of indices corresponding to tokens
        '''
        return [self.word2idx[w] for w in self.data if w in self.word2idx]

    def _make_batches(self, batch_size):
        '''
        :param batch_size: size of minibatches
        :return: object consisting of all batches in corpus to be loaded with _load_batches
        '''
        batch_len = self.N // batch_size
        epoch_size = (batch_len - 1) // self.num_steps
        assert epoch_size > 0, "Batch size caused epoch size to be 0, please try smaller batch size"
        pass

    def _load_next_batch(self):
        # If last batch reset to new epoch
        if self.num_batches == len(self.batches)-1:
            temp = self.num_batches
            self.num_batches = 0
            return self.batches[temp], self.labels[temp]
        else:
            temp = self.num_batches
            self.num_batches += 1
            return self.batches[temp], self.labels[temp]
