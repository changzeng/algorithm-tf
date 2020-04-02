# encoding: utf-8

import tensorflow as tf

from random import shuffle
from utils import generate_sample_list_chinese


class SkipGramTF(object):
    def __init__(self, dim=128, batch_size=128, top_k=5, min_count=10):
        self.dim = dim
        self.batch_size = batch_size
        self.top_k = top_k
        self.min_count = min_count

        self.train_data = None
        self.test_data = None
        self.word_ind_dict = None
        self.word_ind_reverse_dict = None
        self.dict_size = None
        self.saver = None
        self.init_data()

        self.graph = tf.Graph()
        self.inputs = None
        self.labels = None
        self.weights1 = None
        self.bias1 = None
        self.layer1_output = None
        self.output_layer_val = None
        self.loss = None
        self.train_op = None
        self.acc = None
        self.layer1_sim_res = None
        self.layer2_sim_res = None
        self.best_acc = None

        self.build_graph()

    def init_data(self):
        word_ind_dict, total_data = generate_sample_list_chinese(min_count=self.min_count)
        print("dict size: {0}, sample num: {1}".format(len(word_ind_dict), len(total_data)))
        shuffle(total_data)
        split_ind = int(len(total_data)*0.8)
        self.train_data, self.test_data = total_data[:split_ind], total_data[split_ind:]
        self.word_ind_dict = word_ind_dict
        self.dict_size = len(word_ind_dict)
        self.word_ind_reverse_dict = {}
        for k, v in self.word_ind_dict.items():
            self.word_ind_reverse_dict[v] = k

    def build_graph(self):
        with self.graph.as_default():
            with tf.name_scope("input_layer"):
                self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
                self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

            with tf.name_scope("middle_layer"):
                weights1 = tf.get_variable(dtype=tf.float32, shape=[self.dict_size, self.dim],
                                           initializer=tf.truncated_normal_initializer(stddev=1), name="weights1")
                bias1 = tf.get_variable(dtype=tf.float32, shape=[self.dim], initializer=tf.constant_initializer(0.1),
                                        name="bias1")
                output1 = tf.nn.embedding_lookup(weights1, self.inputs) + bias1
                self.weights1, self.bias1, self.layer1_output = weights1, bias1, output1

                weights2 = tf.get_variable(dtype=tf.float32, shape=[self.dim, self.dict_size],
                                           initializer=tf.truncated_normal_initializer(stddev=1), name="weights2")
                bias2 = tf.get_variable(dtype=tf.float32, shape=[self.dict_size], initializer=tf.constant_initializer(0.1),
                                        name="bias2")
                self.output_layer_val = tf.matmul(output1, weights2) + bias2

            with tf.name_scope("update"):
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.output_layer_val,
                        labels=self.labels
                    )
                )
                self.train_op = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

            with tf.name_scope("evaluate"):
                self.acc = tf.reduce_mean(
                    tf.cast(
                        tf.nn.in_top_k(self.output_layer_val, self.labels, self.top_k)
                        , tf.float32
                    )
                )
                norm_emb = tf.divide(
                    self.weights1
                    , tf.reshape(
                        tf.sqrt(
                            tf.reduce_sum(
                                tf.square(self.weights1)
                                , axis=1
                            )
                        )
                        , (self.dict_size, 1)
                    )
                )
                select_emb = tf.expand_dims(
                    tf.nn.embedding_lookup(self.weights1, self.inputs)
                    , axis=1
                )
                closest = tf.nn.top_k(
                    tf.reduce_sum(tf.multiply(select_emb, norm_emb), axis=2)
                    , self.top_k
                )
                self.layer1_sim_res = tf.concat(
                    [
                        tf.cast(tf.expand_dims(closest.indices, axis=2), tf.float32),
                        tf.cast(tf.expand_dims(closest.values, axis=2), tf.float32)
                    ], axis=2
                )
                layer2_closest = tf.nn.top_k(self.output_layer_val, self.top_k)
                self.layer2_sim_res = tf.concat(
                    [
                        tf.cast(tf.expand_dims(layer2_closest.indices, axis=2), tf.float32),
                        tf.cast(tf.expand_dims(layer2_closest.values, axis=2), tf.float32)
                    ], axis=2
                )

            self.saver = tf.train.Saver(max_to_keep=2)

    def train(self):
        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            global_step, avg_loss, avg_acc = 0, 0, 0
            print_batch_num = 1000
            for epoch in range(1000):
                for ind, batch in enumerate(self.batch_iterator()):
                    global_step += 1
                    _, loss, train_acc = sess.run(
                        [self.train_op, self.loss, self.acc]
                        , feed_dict={self.inputs: batch[0], self.labels: batch[1]}
                    )
                    avg_loss += loss
                    avg_acc += train_acc
                    if global_step % print_batch_num == 0:
                        avg_loss, avg_acc = avg_loss/print_batch_num, avg_acc/print_batch_num
                        print("train --> epoch: {0}, complete batch: {1}, step: {2}, loss: {3:.2f}, top20 acc: {4:.2f}"
                              .format(epoch+1, ind+1, global_step, avg_loss, avg_acc))
                        avg_loss, avg_acc = 0, 0

                    # if global_step % 20000 == 0:
                    #     self.test(sess)

                    if global_step % 40000 == 0:
                        self.show_sim_word(sess)

                    if global_step % 50000 == 0:
                        print("saving model...")
                        self.saver.save(sess, "models/skip_gram/skip_gram", global_step=global_step)

    def format_res(self, sim_res):
        dst_list = []
        for dst_ind, dst_score in sim_res:
            dst_word = self.word_ind_reverse_dict[int(dst_ind)]
            dst_list.append("{0} {1:.2f}".format(dst_word, dst_score))
        return dst_list

    def show_sim_word(self, sess):
        ind_list = list(range(self.dict_size))
        shuffle(ind_list)
        select_ind_list = ind_list[:self.batch_size]
        layer1_sim_res, layer2_sim_res = sess.run(
            [self.layer1_sim_res, self.layer2_sim_res]
            , feed_dict={self.inputs: select_ind_list}
        )
        for ind, select_ind in enumerate(select_ind_list):
            if ind >= 20:
                break
            sim_res1, sim_res2 = layer1_sim_res[ind], layer2_sim_res[ind]
            src_word = self.word_ind_reverse_dict[select_ind]
            dst_list1 = self.format_res(sim_res1)
            dst_list2 = self.format_res(sim_res2)
            print("{0}\n{1}\n{2}".format(src_word, ",".join(dst_list1), ",".join(dst_list2)))

    def test(self, sess):
        avg_acc, test_batch_num = 0.0, 0
        for ind, batch in enumerate(self.batch_iterator(is_test=True)):
            test_batch_num += 1
            acc = sess.run(
                self.acc
                , feed_dict={self.inputs: batch[0], self.labels: batch[1]}
            )
            avg_acc += acc
        avg_acc /= test_batch_num
        print("test --> batch num: {0}, top20 acc: {1:.2f}".format(test_batch_num, avg_acc))

    def batch_iterator(self, is_test=False):
        if is_test:
            data = self.test_data
        else:
            data = self.train_data

        shuffle(data)
        batch = []
        for ind, sample in enumerate(data):
            batch.append(sample)
            if (ind+1) % self.batch_size == 0:
                inputs, labels = zip(*batch)
                batch = []
                yield inputs, labels


if __name__ == "__main__":
    model = SkipGramTF(min_count=20)
    model.train()

