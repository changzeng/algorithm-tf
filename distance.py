# encoding: utf-8

# tensorflow实现各种距离

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DistanceTF(object):
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


class DistanceVisualize(object):
    @classmethod
    def cal_covariance(cls, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.matmul(vec1-np.mean(vec1), vec2-np.mean(vec2))/len(vec1)

    @classmethod
    def cal_pearson(cls, vec1, vec2):
        std1, std2 = np.std(vec1), np.std(vec2)
        return cls.cal_covariance(vec1, vec2)/(std1*std2)

    def covariance(self):
        pass

    def pearson(self):
        """
        能够忽略两个变量间的线性关系，关注整体的变化趋势。
        试用场景：有的人倾向于高评分，有的人倾向于低评分，但对物品的评分趋势一致就可认为是相似用户
        :return:
        """
        plt.figure(figsize=(10, 5))

        def plot(h, w, ind, x, y):
            val = self.cal_pearson(x, y)
            plt.subplot(h, w, ind)
            plt.scatter(x, y)
            plt.title("{0:.2f}".format(val))

        vec_size = 100
        vec1 = np.random.random(vec_size)
        plot(2, 3, 1, vec1, vec1*2+10)
        plot(2, 3, 2, vec1, vec1 + np.random.random(vec_size))
        plot(2, 3, 3, vec1, np.power(vec1, 4))
        plot(2, 3, 4, vec1, np.square(vec1) + 2)
        plot(2, 3, 5, vec1, np.sqrt(vec1) + 2)
        plot(2, 3, 6, vec1, np.log(vec1))

        # plt.suptitle("pearson")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dist = DistanceVisualize()
    dist.pearson()
