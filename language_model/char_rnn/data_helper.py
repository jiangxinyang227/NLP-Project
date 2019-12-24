import os
import json
import copy
from collections import Counter
import gensim
import numpy as np


class TrainData(object):
    def __init__(self, config):
        self._output_path = config["output_path"]
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

        self._word_vectors_path = config["word_vectors_path"] if config["word_vectors_path"] else None

        self._embedding_size = config["embedding_size"]  # 字向量的长度
        self.word_vectors = None
        self.vocab_size = config["vocab_size"]

    @staticmethod
    def read_data(file_path):
        """
        读取数据
        :return:
        """
        with open(file_path, "r", encoding="utf8") as fq:
            datas = fq.read()

        return datas

    def get_word_vectors(self, vocab):
        """
        加载字向量，并获得相应的字向量矩阵
        :param vocab: 字汇表
        :return:
        """
        word_vectors = (1 / np.sqrt(len(vocab)) * (2 * np.random.rand(len(vocab), self._embedding_size) - 1))
        if os.path.splitext(self._word_vectors_path)[-1] == ".bin":
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._word_vectors_path, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._word_vectors_path)

        for i in range(len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_vectors[i, :] = vector
            except:
                print(vocab[i] + "不存在于字向量中")

        return word_vectors

    def gen_vocab(self, datas):
        """
        生成词汇映射表
        :param datas: 问题
        :return:
        """

        all_words = [word for word in datas]

        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        words = ["<UNK>"] + [item[0] for item in sort_word_count]
        vocab = words[: self.vocab_size]
        self.vocab_size = len(vocab)
        if self._word_vectors_path:
            word_vectors = self.get_word_vectors(vocab)
            self.word_vectors = word_vectors
            # 将本项目的词向量保存起来
            np.save(os.path.join(self._output_path, "word_vectors.npy"), self.word_vectors)

        word_to_index = dict(zip(vocab, list(range(len(vocab)))))

        # 将词汇-索引映射表保存为pkl数据，之后做inference时直接加载来处理数据
        with open(os.path.join(self._output_path, "word_to_index.json"), "w", encoding="utf8") as f:
            json.dump(word_to_index, f, ensure_ascii=False)

        return word_to_index

    def get_vocab(self):
        """
        加载验证数据时，直接用已有的词表
        :return:
        """
        with open(os.path.join(self._output_path, "word_to_index.json"), "r", encoding="utf8") as f:
            word_to_index = json.load(f)
        return word_to_index

    @staticmethod
    def trans_to_index(data, word_to_index):
        """
        将输入转化为索引表示
        :param data: 输入的是questions 和 responses
        :param word_to_index: 词汇-索引映射表
        :return:
        """
        data_ids = [word_to_index.get(word, word_to_index["<UNK>"]) for word in data]

        return data_ids

    def gen_data(self, file_path, is_training=True):
        """
        生成可导入到模型中的数据
        :return:
        """
        # 1，读取原始数据
        datas = self.read_data(file_path)

        if is_training:
            # 3，得到词汇表
            word_to_index = self.gen_vocab(datas)
        else:
            word_to_index = self.get_vocab()

        # 4，输入转索引
        datas_idx = self.trans_to_index(datas, word_to_index)

        return np.array(datas_idx)

    @staticmethod
    def next_batch(data, batch_size, sequence_length):
        """
        生成batch数据集
        :param data: 原始数据
        :param batch_size: 一个batch中序列的数量
        :param sequence_length: 每条序列的长度
        :return:
        """
        data = copy.copy(data)
        total_size = batch_size * sequence_length
        n_batches = int(len(data) / total_size)
        arr = data[:total_size * n_batches]
        arr = arr.reshape((batch_size, -1))

        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], sequence_length):
            x = arr[:, n:n + sequence_length]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield dict(inputs=x, labels=y)