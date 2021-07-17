#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/10/26 0026 10:45
# @Author : zhe lang
# @Site : 
# @File : word2vec_model.py
# @Software:


import data_preprocess
import numpy as np
import tensorflow as tf

import collections
import csv
import math
import os
import random
import time
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from adjustText import adjust_text


class CbowModel(object):
    def __init__(self, input_data, input_count, input_dict, input_rdict, vocabulary_num, vocabulary_size = data_preprocess.VOCABULARY_SIZE,
                 batch_size=128, embedding_size=128, window_size=2, num_sampled=5, valid_size=32, valid_window=50):
        self.__asm_encoding_list = input_data
        self.__asm_count_list = input_count
        self.__asm_encoding_dict = input_dict # value:assembly ; key:encoding
        self.__asm_encoding_rdict = input_rdict # key:encoding ; value:assembly

        self.__data_index = 0
        self.__vocabulary_num = vocabulary_num
        self.__vocabulary_size = vocabulary_size

        self.__batch_size = batch_size
        self.__embedding_size = embedding_size
        self.__window_size = window_size
        self.__num_sampled = num_sampled
        self.__valid_size = valid_size
        self.__valid_window = valid_window

        self.__train_step_num = 137500 * 10
        self.__cbow_loss_interval = 137500
        # self.__train_step_num = 135000 * 10
        # self.__cbow_loss_interval = 135000
        # self.__train_step_num = 820000 * 5
        # self.__cbow_loss_interval = 820000 # 820189
        self.__cbow_loss_list = []
        self.__cbow_loss_csv_path = './cbow_losses_%s.csv' % time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.__cbow_loss_png_path = './cbow_losses_%s.png' % time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.__cbow_embeddings_tsne_path = './cbow_embeddings_tsne_%s.png' % time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.__cbow_embeddings = None
        self.__cbow_embeddings_path = './cbow_embeddings.npy'


    def __del__(self):
        pass


    def __generate_batch_cbow(self, batch_size, window_size):
        span = 2 * window_size + 1 # [ skip_window target skip_window ]
        batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        buffer = collections.deque(maxlen=span)
        sep_id = self.__asm_encoding_dict[' ']
        for i in range(batch_size):
            while sep_id in self.__asm_encoding_list[self.__data_index: self.__data_index + span]:
                cur_window = self.__asm_encoding_list[self.__data_index: self.__data_index + span]
                self.__data_index = (self.__data_index + 1 + cur_window.index(sep_id)) % len(self.__asm_encoding_list)
                if self.__data_index + span >= len(self.__asm_encoding_list):
                    self.__data_index = 0
            buffer.extend(self.__asm_encoding_list[self.__data_index: self.__data_index + span])
            self.__data_index = (self.__data_index + 1) % len(self.__asm_encoding_list)

            target = window_size
            col_idx = 0
            for j in range(span):
                if j == span // 2:
                    continue
                batch[i, col_idx] = buffer[j]
                col_idx += 1
            labels[i, 0] = buffer[target]

        # for _ in range(span):
        #     buffer.append(self.__asm_encoding_list[self.__data_index])
        #     self.__data_index = (self.__data_index + 1) % len(self.__asm_encoding_list)
        #
        # for i in range(batch_size):
        #     target = window_size
        #     target_to_avoid = [window_size]
        #     col_idx = 0
        #     for j in range(span):
        #         if j == span // 2:
        #             continue
        #         batch[i, col_idx] = buffer[j]
        #         col_idx += 1
        #     labels[i, 0] = buffer[target]
        #     buffer.append(self.__asm_encoding_list[self.__data_index])
        #     self.__data_index = (self.__data_index + 1) % len(self.__asm_encoding_list)

        return batch, labels


    def train(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
        tf.reset_default_graph()

        train_dataset = tf.placeholder(tf.int32, shape=[self.__batch_size, 2 * self.__window_size])
        train_labels = tf.placeholder(tf.int32, shape=[self.__batch_size, 1])

        embeddings = tf.Variable(tf.random_uniform([self.__vocabulary_size, self.__embedding_size], -1.0, 1.0, dtype=tf.float32))
        softmax_weights = tf.Variable(tf.truncated_normal([self.__vocabulary_size, self.__embedding_size], stddev=0.5 / math.sqrt(self.__embedding_size), dtype=tf.float32))
        softmax_biases = tf.Variable(tf.random_uniform([self.__vocabulary_size], 0.0, 0.01))
        valid_examples = np.array(random.sample(range(self.__valid_window), self.__valid_size))
        valid_examples = np.append(valid_examples, random.sample(range(0, 0 + self.__valid_window), self.__valid_size), axis=0)
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # 降维运算 [batch_size, embedding_size, 2 * window_size] -> [batch_size, embedding_size]
        stacked_embedings = None
        for i in range(2 * self.__window_size):
            embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
            x_size, y_size = embedding_i.get_shape().as_list()
            if stacked_embedings is None:
                stacked_embedings = tf.reshape(embedding_i, [x_size, y_size, 1])
            else:
                stacked_embedings = tf.concat(axis=2, values=[stacked_embedings, tf.reshape(embedding_i, [x_size, y_size, 1])])
        mean_embeddings = tf.reduce_mean(stacked_embedings, 2, keepdims=False)

        # unigram 负采样机制
        word_count_dictionary = {}
        unigrams = [0 for _ in range(self.__vocabulary_size)]
        for word, w_count in self.__asm_count_list:
            w_idx = self.__asm_encoding_dict[word]
            unigrams[w_idx] = w_count * 1.0 / self.__vocabulary_num
            word_count_dictionary[w_idx] = w_count
        candidate_sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes=tf.cast(train_labels, dtype=tf.int64),
                                                                  num_true=1, num_sampled=self.__num_sampled,
                                                                  unique=True, range_max=self.__vocabulary_size,
                                                                  distortion=0.75, num_reserved_ids=0,
                                                                  unigrams=unigrams, name='unigram_sampler')
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings, labels=train_labels,
                                                         num_sampled=self.__num_sampled, num_classes=self.__vocabulary_size, sampled_values=candidate_sampler))
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        # 降采样机制
        subsampled_data = []
        drop_count = 0
        drop_examples = []
        for w_i in self.__asm_encoding_list:
            p_w_i = 1 - np.sqrt(1e5 / word_count_dictionary[w_i])
            if np.random.random() < p_w_i:
                drop_count += 1
                drop_examples.append(reverse_dictionary[w_i])
            else:
                subsampled_data.append(w_i)
        print('Dropped %d%% words (%d words) in total...' % (drop_count * 100.0 / len(self.__asm_encoding_list), drop_count))
        self.__asm_encoding_list = subsampled_data
        print(len(self.__asm_encoding_list))

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

        average_loss = 0
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
            tf.global_variables_initializer().run()
            print('Initialized')

            for step in range(self.__train_step_num):
                batch_data, batch_labels = self.__generate_batch_cbow(self.__batch_size, self.__window_size)
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l

                # if (step + 1) % self.__cbow_loss_interval == 0:
                if (step + 1) % 5000 == 0:
                    if step > 0:
                        # average_loss = average_loss / self.__cbow_loss_interval
                        average_loss = average_loss / 5000
                    self.__cbow_loss_list.append(average_loss)
                    print('Average loss at step %d: %f' % (step + 1, average_loss))
                    average_loss = 0

                # if (step + 1) % 200081 == 0:
                # if (step + 1) % 135000 == 0:
                if (step + 1) % 137500 == 0:
                    print(time.strftime("%Y%m%d %H%M%S", time.localtime()))
                    sim = similarity.eval()
                    for i in range(self.__valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 5
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to %s:' % valid_word.strip('\n')
                        for k in range(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)
            cbow_final_embeddings = normalized_embeddings.eval()
        # 词汇编码保存
        self.__cbow_embeddings = cbow_final_embeddings
        np.save(self.__cbow_embeddings_path, cbow_final_embeddings)

        with open(self.__cbow_loss_csv_path, 'wt') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(self.__cbow_loss_list)

        return


    def plot_loss(self):
        # with open(self.__cbow_loss_csv_path, 'rt') as f:
        with open('./cbow_losses_20210327020829.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for r_i, row in enumerate(reader):
                if r_i == 0:
                    cow_loss = [float(s) for s in row]
        pylab.figure(figsize=(15, 5))
        print(len(cow_loss))
        x = np.arange(len(cow_loss)) * 20000
        pylab.plot(x, cow_loss, label="cbow model", linewidth=1)
        # pylab.title('cbow', fontsize=24)
        pylab.xlabel('Iterations', fontsize=22)
        pylab.ylabel('Loss', fontsize=22)
        pylab.xlim(0, 25 * 1e5)
        pylab.legend(loc=1, fontsize=22)
        pylab.savefig(self.__cbow_loss_png_path)
        pylab.show()

        # x = np.arange(len(cow_loss_path))
        # plt.plot(x, self.__cbow_loss_list, '--', label='cbow', linewidth=1)
        # plt.xlabel('Epoch', fontsize=20)
        # plt.ylabel('Loss', fontsize=20)
        # plt.legend(loc=1, fontsize=20)
        # x_locator = plt.MultipleLocator(5)
        # y_locator = plt.MultipleLocator(0.1)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_locator)
        # ax.yaxis.set_major_locator(y_locator)
        # plt.xlim(0, 50)
        # plt.ylim(0, 1.0)
        # plt.savefig(self.__cbow_loss_png_path)
        # plt.show()

        return


    def __find_clustered_embeddings(self, embeddings, distance_threshold, sample_threshold):
        '''
        Find only the closely clustered embeddings.
        This gets rid of more sparsly distributed word embeddings and make the visualization clearer
        This is useful for t-SNE visualization

        distance_threshold: maximum distance between two points to qualify as neighbors
        sample_threshold: number of neighbors required to be considered a cluster
        '''

        # calculate cosine similarity
        cosine_sim = np.dot(embeddings, np.transpose(embeddings))
        norm = np.dot(np.sum(embeddings ** 2, axis=1).reshape(-1, 1),
                      np.sum(np.transpose(embeddings) ** 2, axis=0).reshape(1, -1))
        assert cosine_sim.shape == norm.shape
        cosine_sim /= norm

        # make all the diagonal entries zero otherwise this will be picked as highest
        np.fill_diagonal(cosine_sim, -1.0)

        argmax_cos_sim = np.argmax(cosine_sim, axis=1)
        mod_cos_sim = cosine_sim
        # find the maximums in a loop to count if there are more than n items above threshold
        for _ in range(sample_threshold - 1):
            argmax_cos_sim = np.argmax(cosine_sim, axis=1)
            mod_cos_sim[np.arange(mod_cos_sim.shape[0]), argmax_cos_sim] = -1

        max_cosine_sim = np.max(mod_cos_sim, axis=1)

        return np.where(max_cosine_sim > distance_threshold)[0]


    # 150 0.25 3
    # 125 0.25 2
    def plot_embeddings(self):
        # self.__cbow_embeddings = np.load(self.__cbow_embeddings_path)
        num_points = 60
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000) # 5000
        selected_embeddings = self.__cbow_embeddings[:num_points, :]
        two_d_embeddings = tsne.fit_transform(selected_embeddings)
        selected_ids = self.__find_clustered_embeddings(selected_embeddings, 0.25, 2)

        embeddings = two_d_embeddings[selected_ids, :]
        labels = [self.__asm_encoding_rdict[i] for i in selected_ids]

        n_clusters = 5
        # label_colors = [pylab.cm.Spectral(float(i) / n_clusters) for i in range(n_clusters)]
        label_colors = [plt.cm.Spectral(float(i) / n_clusters) for i in range(n_clusters)]
        assert embeddings.shape[0] >= len(labels)

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(embeddings)
        kmeans_labels = kmeans.labels_

        x, y = np.transpose(embeddings)
        fig, ax = plt.subplots(figsize=(30, 20))
        ax.scatter(x, y, s=200)
        texts = [plt.text(x_, y_, text, fontsize=40) for x_, y_, text in zip(x, y, labels)]
        adjust_text(texts)
        plt.savefig(self.__cbow_embeddings_tsne_path)
        plt.show()
        return

        # pylab.figure(figsize=(20, 20))
        # for i, (label, klabel) in enumerate(zip(labels, kmeans_labels)):
        #     x, y = embeddings[i, :]
        #     pylab.scatter(x, y, c=label_colors[klabel])
        #     pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=30)
        #
        # pylab.savefig(self.__cbow_embeddings_tsne_path)
        # pylab.show()
        # return


if __name__ == '__main__':
    asm_ins_list = data_preprocess.read_asm_ins('./words/datasets_ppc')
    asm_encoding_list, asm_count_list, dictionary, reverse_dictionary = data_preprocess.build_asm_dataset(asm_ins_list)
    cbow = CbowModel(input_data = asm_encoding_list, input_count = asm_count_list,
                     input_dict = dictionary, input_rdict = reverse_dictionary, vocabulary_num = len(asm_ins_list))
    cbow.train()
    # cbow.plot_loss()
    cbow.plot_embeddings()

