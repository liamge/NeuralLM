import tensorflow as tf
import numpy as np
import os
import argparse

from tensorflow.contrib.tensorboard.plugins import projector
from models.basic_lm import BasicNeuralLM
from utils import DataLoader, SmallConfig, MediumConfig, LargeConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', help='file containing data for training or testing', required=True, dest='datafile')
    parser.add_argument('-c', '--conf', help='choice of configuration for model, big, medium, or small', dest='conf', required=True)
    return vars(parser.parse_args())

def train_model(model, visualize_embeddings=True):
    '''
    :param model: model object which has had model.build_graph() already run
    :param num_train_steps: number of steps to train model
    :return: n/a
    '''
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

        #if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(sess, ckpt.model_checkpoint_path)

        initial_step = model.global_step.eval()
        for i in range(initial_step + model.conf.num_epochs):
            epoch_loss = 0.0
            # Training loop for epoch
            total_loss = 0.0
            for index in range(model.data.num_batches["train"]):
                batch_X, batch_y_ = model.data._load_next_batch()
                feed_dict = {model.train_input: batch_X, model.train_labels: batch_y_}
                loss_batch, _ = sess.run([model.loss, model.optimizer],
                                                feed_dict=feed_dict)
                total_loss += loss_batch
                epoch_loss += loss_batch
                if (index + 1) % model.conf.skip_step == 0:
                    total_loss = sum(total_loss)
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / model.conf.skip_step))
                    total_loss = 0.0
                    saver.save(sess, 'checkpoints/language-model', index)
            # Validation loop for epoch
            val_loss = 0.0
            for index in range(model.data.num_batches["validation"]):
                batch_X, batch_y_ = model.data._load_next_batch()
                feed_dict = {model.train_input: batch_X, model.train_labels: batch_y_}
                loss_batch, _ = sess.run([model.loss, model.optimizer],
                                                  feed_dict=feed_dict)
                total_loss += loss_batch
            print("Average validation set loss: {}".format(val_loss / model.data.num_batches["validation"]))
            print("Epoch {} complete with average error of {} on training set".format(i+1, sum(epoch_loss) / model.data.num_batches["train"]))

        if visualize_embeddings:
            final_embed_matrix = sess.run(model.embed_matrix)
            embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter('processed')

            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            embedding.metadata_path = 'processed/vocab_1000.tsv'
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, 'processed/model3.ckpt', 1)

def generate_sequence(model, seed, max_seq_len=45):
    '''
    :param model: model object which has had model.build_graph() already run
    :param seed: list of num_steps strings
    :return: list of strings
    '''
    assert len(seed) == model.conf.num_steps, "Error: seed is of incorrect length, must provide list of {} strings".format(model.conf.num_steps)
    seq = []
    for w in seed:
        if w not in model.data.V:
            seq.append(model.data.word2idx['<UNK>'])
        else:
            seq.append(model.data.word2idx[w])


    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        current = seq[-1]
        while current != '<EOS>' or len(seq) <= max_seq_len:
            # Lazy solution here where we just pass the input sequence and a matrix of zeros so its the right shape
            input_matrix = np.zeros([model.conf.batch_size, model.conf.num_steps])
            input_matrix[0, :] = seq[-model.conf.num_steps]
            feed_dict = {model.train_input: input_matrix}
            current = sess.run(tf.argmax(tf.nn.softmax(model.logits)), feed_dict=feed_dict)
            seq.append(current[0])

    return [model.data.idx2word[idx] for idx in seq]


if __name__ == "__main__":
    args = parse_args()
    if args['conf'] == 'large':
        conf = LargeConfig()
    elif args['conf'] == 'medium':
        conf = MediumConfig()
    elif args['conf'] == 'small':
        conf = SmallConfig()
    else:
        print('Error: Configuration {} not accepted, please use either large, medium, or small configs.'.format(args['conf']))
    data = DataLoader(args['datafile'], conf.batch_size, conf.num_steps)

    model = BasicNeuralLM(conf, data)
    model.build_graph()

    train_model(model)