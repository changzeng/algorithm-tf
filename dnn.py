# encoding: utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


class DnnTF(object):
    def __init__(self, batch_size=128, layer_out_dim_list=None, learning_rate=0.1):
        if layer_out_dim_list is None:
            layer_out_dim_list = [784, 128, 10]
        self.batch_size = batch_size
        self.layer_out_dim_list = layer_out_dim_list
        self.learning_rate = learning_rate
        self.mnist = input_data.read_data_sets('./data/mnist-original.mat')

        self.graph = tf.Graph()
        self.inputs = None
        self.labels = None
        self.output_layer_val = None
        self.predict_ids = None
        self.predict_val = None
        self.loss = None
        self.train_op = None
        self.acc = None

        self.build_graph()

    def next_batch_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return self.mnist.train.next_batch(batch_size)

    def next_feed_dict(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        inputs, labels = self.next_batch_data(batch_size)
        return {self.inputs: inputs, self.labels: labels}

    def build_graph(self):
        with self.graph.as_default():
            with tf.name_scope("input"):
                self.inputs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.layer_out_dim_list[0]])
                self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

            input_val = self.inputs
            for ind in range(1, len(self.layer_out_dim_list)):
                with tf.name_scope("middle_layer_{0}".format(ind)):
                    input_dim = self.layer_out_dim_list[ind-1]
                    output_dim = self.layer_out_dim_list[ind]

                    weights = tf.get_variable(dtype=tf.float32, shape=[input_dim, output_dim],
                                              name="layer_{0}_weights".format(ind))
                    bias = tf.get_variable(dtype=tf.float32, shape=[output_dim], name="layer_{0}_bias".format(ind))

                    output_val = tf.matmul(input_val, weights) + bias

                    input_val = output_val

            with tf.name_scope("output_layer"):
                self.output_layer_val = output_val

                predict = tf.nn.top_k(self.output_layer_val, 1)
                self.predict_ids = tf.reshape(predict.indices, [self.batch_size])
                self.predict_val = tf.reshape(predict.values, [self.batch_size])

            with tf.name_scope("update"):
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.output_layer_val,
                        labels=self.labels
                    )
                )
                self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            with tf.name_scope("evaluate"):
                self.acc = tf.reduce_mean(
                    tf.cast(
                        tf.equal(self.labels, tf.cast(self.predict_ids, tf.int32))
                        , tf.float32)
                )

    def train(self):

        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            for ind in range(2000):
                _, loss, acc, predict, predict_val = sess.run(
                    [self.train_op, self.loss, self.acc, self.predict_ids, self.predict_val],
                    feed_dict=self.next_feed_dict()
                )
                # print("labels:\n", labels)
                # print("predict:\n", predict)
                # print("predict_val:\n", predict_val)
                if (ind+1) % 100 == 0:
                    test_acc = sess.run(self.acc, feed_dict=self.next_feed_dict())
                    print("train batch: {0}, train acc: {1:.2f}, train loss: {2:.2f}, test acc: {3:.2f}"
                          .format(ind+1, acc, loss, test_acc))
            self.visualize(sess)

    def visualize(self, sess):
        batch_size = 100
        col_size = int(np.sqrt(batch_size))
        inputs, labels = self.next_batch_data()
        predicts = sess.run(self.predict_ids, feed_dict={self.inputs: inputs, self.labels: labels})
        plt.figure(figsize=(6, 8))
        for ind in range(batch_size):
            pic = np.reshape(inputs[ind, :], (28, 28))
            number = labels[ind]
            predict = predicts[ind]

            plt.subplot(col_size, col_size, ind+1)
            plt.imshow(pic, cmap='Greys')
            plt.title("{0}".format(predict))
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dnn = DnnTF(batch_size=256)
    dnn.train()

