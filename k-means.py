# encoding: utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


class KMeansTF(object):
    def __init__(self, dim=2, num_cluster=5, batch_size=128, epoch=10, average_batch_size=256, threshold=0.01):
        self.dim = dim
        self.num_cluster = num_cluster
        self.graph = tf.Graph()
        self.batch_size = batch_size
        self.epoch = epoch
        self.average_batch_size = average_batch_size
        self.threshold = threshold

        self.inputs = None
        self.cluster_centers = None
        self.distance = None
        self.predict_center_ids = None
        self.num_list = None
        self.sum_list = None
        self.center_list = None

        self.build_graph()

        self.batch_list = []
        self.total_batch_num = 1000

    def build_graph(self):
        with self.graph.as_default():
            # 训练数据输入
            self.inputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.dim), name="inputs")
            # 聚类中心
            self.cluster_centers = tf.get_variable(dtype=tf.float32, shape=(self.num_cluster, self.dim),
                                                   name="cluster_centers")
            # 计算输入数据到各个中心点的距离
            center_distance = tf.reshape(
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            tf.expand_dims(self.inputs, axis=1) - self.cluster_centers
                        )
                        , axis=2
                    )
                )
                , (self.batch_size, self.num_cluster)
            )
            closest_center = tf.nn.top_k(-1*center_distance, 1)
            indices = closest_center.indices

            # 最近中心点的距离
            self.distance = -1*closest_center.values
            # 最近中心点编号
            self.predict_center_ids = tf.reshape(indices, [self.batch_size])

            # 将属于同一中心点的数据求和
            num_list, sum_list = [], []
            for c_id in range(self.num_cluster):
                is_belong = tf.equal(self.predict_center_ids, c_id)
                nums = tf.reduce_sum(tf.cast(is_belong, tf.int32), axis=0)
                sums = tf.reduce_sum(tf.gather_nd(self.inputs, tf.where(is_belong))
                                     , axis=0)

                num_list.append(nums)
                sum_list.append(sums)

            self.num_list = num_list
            self.sum_list = sum_list

    def predict(self):
        with tf.Session(graph=self.graph) as sess:
            predict_list = []
            for batch in self.train_batch_iterator():
                predict_ids = sess.run(self.predict_center_ids,
                                       feed_dict={self.inputs: batch, self.cluster_centers: self.center_list})
                predict_list.append(predict_ids)
        return np.concatenate(predict_list)

    @classmethod
    def calculate_distance(cls, inputs, centers):
        result = []
        for ind in range(inputs.shape[0]):
            vec_a = inputs[ind, :]
            dist_list = []
            for c_ind in range(centers.shape[0]):
                vec_b = centers[c_ind, :]
                dist = np.sum((vec_a - vec_b)**2)
                dist_list.append(np.sqrt(dist))
            result.append(dist_list)
        return result

    def train(self):
        center_list = np.random.rand(self.num_cluster, self.dim)
        with tf.Session(graph=self.graph) as sess:
            for epoch in range(self.epoch):
                print("epoch: {0}".format(epoch))
                cluster_size = np.zeros(self.num_cluster)
                vec_sum = np.zeros((self.num_cluster, self.dim))
                for batch in self.train_batch_iterator():
                    num_list, sum_list, center_ids, distance = sess.run(
                        [self.num_list, self.sum_list, self.predict_center_ids, self.distance],
                        feed_dict={self.inputs: batch, self.cluster_centers: center_list}
                    )
                    # tf_dist_list = list(zip(center_ids, distance))
                    # dist_list = self.calculate_distance(batch, center_list)
                    cluster_size += num_list
                    vec_sum += sum_list
                # 更新中心点
                nxt_center_list = vec_sum / np.reshape(cluster_size, (self.num_cluster, 1))
                diff = np.sum(np.abs(center_list-nxt_center_list)) / self.num_cluster
                center_list = nxt_center_list
                if diff <= self.threshold:
                    break

        self.center_list = center_list

    def train_batch_iterator(self):
        if len(self.batch_list) == 0:
            for batch_ind in range(self.total_batch_num):
                batch = np.random.rand(self.batch_size, self.dim)
                self.batch_list.append(batch)
                yield batch
        else:
            for batch in self.batch_list:
                yield batch

    def plot_result(self):
        total_data = np.concatenate(self.batch_list)
        x, y = total_data[:, 0], total_data[:, 1]

        plt.figure(figsize=(10, 12))
        plt.subplot(2, 1, 1)
        sk_model = KMeans(n_clusters=self.num_cluster)
        sk_model.fit(total_data)
        sk_model_predict = sk_model.predict(total_data)
        sk_model_center_list = sk_model.cluster_centers_

        plt.scatter(x, y, c=sk_model_predict)

        center_x = sk_model_center_list[:, 0]
        center_y = sk_model_center_list[:, 1]
        color = [i+2 for i in range(self.num_cluster)]
        size = [50]*self.num_cluster
        plt.scatter(center_x, center_y, c=color, s=size, edgecolors='k', marker='o')
        plt.title("sklearn result")

        plt.subplot(2, 1, 2)
        predict_center_ids = self.predict()
        plt.scatter(x, y, c=predict_center_ids)

        center_x = self.center_list[:, 0]
        center_y = self.center_list[:, 1]
        color = [i+2 for i in range(self.num_cluster)]
        size = [50]*self.num_cluster
        plt.scatter(center_x, center_y, c=color, s=size, edgecolors='k', marker='o')
        plt.title("tensorflow result")

        plt.tight_layout()
        plt.show()


def main():
    model = KMeansTF(epoch=20)
    model.train()
    model.plot_result()


if __name__ == "__main__":
    main()
