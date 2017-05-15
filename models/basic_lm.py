import tensorflow as tf
import numpy as np
import functools

'''
Model based off of http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf.
Works as a simplistic baseline to compare alternative models against.
TODO: Define generate function (straightforward, make forward pass and use np.argmax to select predicted index, then append idx2word[idx] to running sentence until it returns '<EOS'> or reaches a set length
TODO: TSNE Projections for visualization
TODO: Allow GPU use in args
'''

class BasicNeuralLM:
    def __init__(self, conf, data_loader):
        self.conf = conf
        self.data = data_loader

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _build_placeholders(self):
        self.train_input = tf.placeholder(tf.float32, shape=[None, self.conf.hidden_dim])
        self.train_labels = tf.placeholder(tf.float32, shape=[None])

    def _build_embeddings(self):
        #TODO: Allow for loading of pretrained vectors
        with tf.device('/cpu:0'):
            with tf.name_scope('embed'):
                if self.conf.trainable == False:
                    self.embed_matrix = tf.Variable(tf.random_uniform([self.data.V, self.conf.embed_dim], -1.0, 1.0),
                                                    trainable=False,
                                                    name='embed_matrix')
                else:
                    self.embed_matrix = tf.Variable(tf.constant(0.0, shape=[self.data.V, self.conf.embed_dim]),
                                                    name='embed_matrix')

    def _build_loss(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('loss'):
                # Output bias
                b = tf.Variable(tf.constant([len(self.data.V)], 1.0), name='bias1')
                # Allow for direct connections to softmax layer (input to output)
                if self.conf.direct_connections == True:
                    W = tf.Variable(tf.random_normal([self.conf.embed_dim, len(self.data.V)], -1.0, 1.0))
                else:
                    W = tf.Variable(np.zeros([self.conf.batch_size, len(self.data.V)]), trainable=False)

                # Input to hidden
                H = tf.Variable(tf.random_uniform([self.conf.num_steps, self.conf.hidden_dim], -1.0, 1.0))
                # Hidden bias
                d = tf.Variable(tf.random_uniform([self.conf.hidden_dim]))
                # Hidden to output
                U = tf.Variable(tf.random_uniform([self.conf.hidden_dim, len(self.data.V)]))

                input = tf.nn.embedding_lookup(self.embed_matrix, self.train_input)

                hidden = tf.tanh(tf.matmul(input, H) + d)
                hidden2out = tf.matmul(hidden, U) + b
                self.logits = tf.matmul(input, W) + hidden2out

                self.loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, labels=self.train_labels)

    def _build_optimizer(self):
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.conf.lr).minimize(self.loss)

    def _build_summaries(self):
        # For tensorboard visualization, run main.py then in command line prompt type:
        # tensorboard --logdir="./graphs" --port 6006
        # then open browser to http://localhost:6006/
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _build_graph(self):
        self._build_placeholders()
        self._build_embeddings()
        self._build_loss()
        self._build_optimizer()
        self._build_summaries()