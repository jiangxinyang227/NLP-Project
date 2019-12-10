import os
import json
import copy
import random
from collections import Counter
from itertools import chain

import jieba
import gensim
import numpy as np


class SiameseLstmData(object):
    def __init__(self, config):
        self.__output_path = config["output_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.__stop_word_path = config["stop_word_path"]
        self.__embedding_size = config["embedding_size"]
        self.__word_vector_path = config["word_vector_path"]
        self.__low_freq = config["low_freq"]

        self.vocab_size = None
        self.word_vectors = None

    @staticmethod
    def load_data(file_path):
        """

        :param file_path:
        :return:
        """
        queries = []
        sims = []
        labels = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                try:
                    query, sim, label = line.strip().split("\t")
                    queries.append(jieba.lcut(query))
                    sims.append(jieba.lcut(sim))
                    labels.append(label)
                except:
                    continue

        return queries, sims, labels

    def remove_stop_word(self, inputs):
        """
        去除低频词/字和停用词/字, 构建一个项目的vocab词/字汇表
        :param inputs: 输入
        :return:
        """
        all_words = list(chain(*inputs))

        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sort_word_count if item[1] > self.__low_freq]

        # 如果传入了停用词表，则去除停用词
        if self.__stop_word_path:
            with open(self.__stop_word_path, "r", encoding="utf8") as fr:
                stop_words = [line.strip() for line in fr.readlines()]
            words = [word for word in words if word not in stop_words]

        return words

    def get_word_vectors(self, vocab):
        """
        加载词/字向量，并获得相应的词/字向量矩阵
        :param vocab: 训练集所含有的单词/字
        :return:
        """
        pad_vector = np.zeros(self.__embedding_size)  # 将pad字符的向量置为0
        word_vectors = (1 / np.sqrt(len(vocab) - 1) * (2 * np.random.rand(len(vocab) - 1, self.__embedding_size) - 1))
        if os.path.splitext(self.__word_vector_path)[-1] == ".bin":
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self.__word_vector_path, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self.__word_vector_path)

        for i in range(1, len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_vectors[i, :] = vector
            except:
                print(vocab[i] + "不存在于字向量中")
        word_vectors = np.vstack((pad_vector, word_vectors))
        return word_vectors

    def gen_vocab(self, words, labels):
        """
        生成词汇，标签等映射表
        :param words: 训练集所含有的单词/字
        :return:
        """
        vocab = ["<PAD>", "<UNK>"] + words

        # 若vocab的长读小于设置的vocab_size，则选择vocab的长度作为真实的vocab_size
        self.vocab_size = len(vocab)

        if self.__word_vector_path:
            word_vectors = self.get_word_vectors(vocab)
            self.word_vectors = word_vectors
            # 将本项目的词向量保存起来
            np.save(os.path.join(self.__output_path, "word_vectors.npy"), self.word_vectors)

        word_to_index = dict(zip(vocab, list(range(len(vocab)))))

        # 将词汇-索引映射表保存为pkl数据，之后做inference时直接加载来处理数据
        with open(os.path.join(self.__output_path, "word_to_index.json"), "w", encoding="utf8") as f:
            json.dump(word_to_index, f, ensure_ascii=False, indent=0)

        unique_labels = list(set(labels))
        label_to_index = dict(zip(unique_labels, list(range(len(unique_labels)))))

        with open(os.path.join(self.__output_path, "label_to_index.json"), "w", encoding="utf8") as fw:
            json.dump(label_to_index, fw, ensure_ascii=False)

        return word_to_index, label_to_index

    @staticmethod
    def trans_to_index(queries, word_to_index):
        """

        :param queries:
        :param word_to_index:
        :return:
        """
        query_ids = [[word_to_index.get(word, word_to_index["<UNK>"]) for word in query]
                     for query in queries]

        return query_ids

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        label_ids = [label_to_index[label] for label in labels]
        return label_ids

    @staticmethod
    def padding(query_ids, sim_ids, label_ids):
        """
        对输入的句子进行补全
        :param query_ids:
        :param sim_ids:
        :param label_ids:
        :return:
        """
        sim_length = [len(sim_id) for sim_id in sim_ids]
        sim_max_len = max(sim_length)

        query_length = [len(query_id) for query_id in query_ids]
        query_max_len = max(query_length)
        max_len = max(sim_max_len, query_max_len)

        sim_ids_pad = [sim_id + [0] * (max_len - len(sim_id)) for sim_id in sim_ids]
        query_ids_pad = [query_id + [0] * (max_len - len(query_id)) for query_id in query_ids]

        return dict(query=query_ids_pad, query_length=query_length, sim=sim_ids_pad, sim_length=sim_length,
                    label=label_ids)

    def gen_data(self, file_path, is_training=True):
        """

        :param file_path:
        :param is_training:
        :return:
        """
        queries, sims, labels = self.load_data(file_path)
        words = self.remove_stop_word(queries + sims)
        if is_training:
            word_to_index, label_to_index = self.gen_vocab(words, labels)
        else:
            with open(os.path.join(self.__output_path, "word_to_index.json"), "r", encoding="utf8") as fr:
                word_to_index = json.load(fr)
            with open(os.path.join(self.__output_path, "label_to_index.json"), "r") as fr:
                label_to_index = json.load(fr)

        query_ids = self.trans_to_index(queries, word_to_index)
        sim_ids = self.trans_to_index(sims, word_to_index)
        label_ids = self.trans_label_to_index(labels, label_to_index)

        return query_ids, sim_ids, label_ids

    def next_batch(self, x, y, label, batch_size):
        """
        生成batch 数据
        :param x:
        :param y:
        :param label:
        :param batch_size:
        :return:
        """
        z = list(zip(x, y, label))
        random.shuffle(z)
        x, y, label = zip(*z)
        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = x[start: end]
            batch_y = y[start: end]
            batch_label = label[start: end]
            yield self.padding(batch_x, batch_y, batch_label)
