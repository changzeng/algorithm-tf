# encoding: utf-8

# tensorflow实现各种距离

import tensorflow as tf


class Distance(object):
    def __init__(self, dim=2, batch_size=128):
        self.dim = dim
        self.batch_size = batch_size

        self.vec1 = None
        self.vec2 = None
        self.euler_dist = None
        self.manhattan_dist = None
        self.chebyshev_dist = None
        self.cosine_dist = None

        self.build_graph()

    def build_graph(self):
        with tf.name_scope("input_layer"):
            self.vec1 = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.dim])
            self.vec2 = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.dim])

        with tf.name_scope("euler_distance"):
            self.euler_dist = tf.reshape(
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(self.vec1 - self.vec2)
                        , axis=1)
                )
                , [self.batch_size]
            )

        with tf.name_scope("manhattan_distance"):
            self.manhattan_dist = tf.reshape(
                tf.reduce_sum(
                    tf.abs(self.vec1 - self.vec2)
                    , axis=1
                )
                , [self.batch_size]
            )

        with tf.name_scope("chebyshev_distance"):
            self.chebyshev_dist = tf.reshape(
                tf.nn.top_k(
                    tf.abs(self.vec1 - self.vec2)
                    , 1
                ).values
                , [self.batch_size]
            )

        with tf.name_scope("standardized_euler_distance"):
            pass

        with tf.name_scope("cosine_distance"):
            vec1_norm = tf.sqrt(tf.reduce_sum(tf.square(self.vec1), axis=1))
            vec2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.vec2), axis=1))
            self.cosine_dist = tf.divide(
                tf.reduce_sum(
                    tf.multiply(self.vec1, self.vec2)
                    , axis=1
                )
                , tf.multiply(vec1_norm, vec2_norm)
            )
