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
TODO: Allow GPU use in args
'''

class BasicNeuralLM(object):
    def __init__(self, conf, data_loader):
        self.conf = conf
        self.data = data_loader

        #TODO: Allow for the loading of pretrained vectors
        #TODO: Put placeholders in their own function?
        self.train_input = tf.placeholder(tf.float32, shape=[None, self.conf.hidden_dim])
        self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])

        #TODO: Put embedding in it's own function?
        if self.conf.trainable == False:
            embedding = tf.Variable(tf.random_uniform([self.data.V, self.conf.embed_dim], -1.0, 1.0),
                                trainable=False,
                                name='embed_layer')
        else:
            embedding = tf.Variable(tf.constant(0.0, shape=[self.data.V, self.conf.embed_dim]), name='embed_layer')

        #TODO: Put loss in it's own function?
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

        input = tf.nn.embedding_lookup(embedding, self.train_input)

        hidden = tf.tanh(tf.matmul(input, H) + d)
        hidden2out = tf.matmul(hidden, U) + b
        self.logits = tf.matmul(input, W) + hidden2out

        self.loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, labels=self.train_labels)

        # Optimize step
        self.optimizer = tf.train.GradientDescentOptimizer(conf.lr).minimize(self.loss)

    def _train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            average_loss = 0.0
            for i in range(self.conf.num_epochs):
                batch_data, batch_labels = self.data.next()
                loss_batch, _ = sess.run([self.loss, self.optimizer], feed_dict={self.train_input: batch_data,
                                                                                 self.train_labels: batch_labels})
                average_loss += loss_batch
                if (i + 1) % 2000 == 0:
                    print("Average loss at step {}: {:5.1f}".format(i + 1, average_loss/(i+1)))
