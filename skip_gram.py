# encoding: utf-8

import tensorflow as tf


class SkipGramTF(object):
    def __init__(self, dim=128, dict_size=1000, batch_size=128):
        self.dim = dim
        self.dict_size = dict_size
        self.batch_size = batch_size

        self.inputs = None
        self.labels = None

        self.build_graph()

    def build_graph(self):
        with tf.name_scope("input_layer"):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
            self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

        with tf.name_scope("middle_layer"):
            weights1 = tf.get_variable(dtype=tf.float32, shape=[self.dict_size, self.dim],
                                       initializer=tf.truncated_normal_initializer(stddev=1), name="weights1")
            bias1 = tf.get_variable(dtype=tf.float32, shape=[self.dim], initializer=tf.constant_initializer(0.1),
                                    name="bias1")
            output1 = tf.nn.embedding_lookup(weights1, self.inputs) + bias1

            weights2 = tf.get_variable(dtype=tf.float32, shape=[self.dim, self.dict_size],
                                       initializer=tf.truncated_normal_initializer(stddev=1), name="weights2")
            bias2 = tf.get_variable(dtype=tf.float32, shape=[self.dict_size], initializer=tf.constant_initializer(0.1),
                                    name="bias2")
            self.output_layer_val = tf.matmul(output1, weights2) + bias2

        with tf.name_scope("update"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output_layer_val,
                labels=self.labels
            )
            self.train = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

        with tf.name_scope("evaluate"):
            self.acc = tf.reduce_mean(
                tf.cast(
                    tf.nn.in_top_k(self.output_layer_val, self.labels, 20)
                    , tf.int32
                )
                , axis=1
            )

