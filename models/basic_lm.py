import tensorflow as tf
import numpy as np
import functools

'''
Model based off of http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf.
Works as a simplistic baseline to compare alternative models against.
TODO: Make computational graph
TODO: Define train function
TODO: Define generate function (straightforward, make forward pass and use np.argmax to select predicted index, then append idx2word[idx] to running sentence until it returns '<EOS'> or reaches a set length
TODO: TSNE Projections for visualization
'''

def define_scope(function):
    # Decorator based off https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + funciton.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Model(object):
    def __init__(self, conf, data_loader):
        self.conf = conf
        self.data = data_loader
        self.prediction
        self.optimize
        self.error

    @define_scope
    def prediction(self):
        #TODO: Allow for the freezing of weights/loading of pretrained vectors


        train_input = tf.placeholder(tf.int32, shape=[self.conf.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[self.conf.batch_size, 1])

        if self.conf.embed_trainable == False:
            embedding = tf.Variable(tf.random_uniform([self.data.V, self.conf.embed_dim], -1.0, 1.0),
                                trainable=False,
                                name='embed_layer')
        else:
            embedding = tf.Variable(tf.constant(0.0, shape=[self.data.V, self.conf.embed_dim]), name='embed_layer')

        embed = tf.nn.embedding_lookup(embedding, train_input)

        b = tf.Variable(tf.constant([len(self.conf.V)], 1.0), name='bias1')
        #Allow for direct connections to softmax layer
        if self.conf.direct_connections == True:
            W = tf.Variable(tf.random_normal([len(self.data.V), self.conf.batch_size], -1.0, 1.0))
        else:
            W = tf.Variable(np.zeros([len(self.data.V), self.conf.batch_size]), trainable=False)
        pass    

    @define_scope
    def optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        optimizer = tf.train.GradientDescentOptimiser(self.conf.lr)
        return optimizer.minimize(-logprob)

    @define_scope
    def error(self):
        pass

    def _generate(self):
        pass
