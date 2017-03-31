import tensorflow as tf
import numpy as np

'''
Model based off of http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf.
Works as a simplistic baseline to compare alternative models against.
TODO: Make computational graph
TODO: Define train function
TODO: Define generate function (straightforward, make forward pass and use np.argmax to select predicted index, then append idx2word[idx] to running sentence until it returns '<EOS'> or reaches a set length
'''

class Model(object):
    def __init__(self, config):
        self.graph = self._make_graph


    def _make_graph(self):
        pass

    def _train(self):
        pass

    def _generate(self):
        pass
