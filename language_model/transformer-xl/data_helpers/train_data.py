import os
import pickle
import copy
from collections import Counter
import gensim
import numpy as np

from .data_base import TrainDataBase


class TrainData(TrainDataBase):
    def __init__(self, config):
        super(TrainData, self).__init__(config)

        self._train_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config["train_data"])
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                         config["output_path"])
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

        self._word_vectors_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                               config["word_vectors_path"]) if config["word_vectors_path"] else None

        self._embedding_size = config["embedding_size"]  # 字向量的长度
        self.word_vectors = None
        self.vocab_size = config["vocab_size"]

    def read_data(self):
        """
        读取数据
        :return:
        """
        with open(self._train_data_path, "r", encoding="utf8") as fq:
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
        # 如果不是第一次处理，则可以直接加载生成好的词汇表和词向量
        if os.path.exists(os.path.join(self._output_path, "word_vectors.npy")):
            print("load word_vectors")
            self.word_vectors = np.load(os.path.join(self._output_path, "word_vectors.npy"))

        if os.path.exists(os.path.join(self._output_path, "word_to_index.pkl")):
            print("load word_to_index")
            with open(os.path.join(self._output_path, "word_to_index.pkl"), "rb") as f:
                word_to_index = pickle.load(f)

            self.vocab_size = len(word_to_index)

            return word_to_index

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
        with open(os.path.join(self._output_path, "word_to_index.pkl"), "wb") as f:
            pickle.dump(word_to_index, f)

        return word_to_index

    def trans_to_index(self, data, word_to_index):
        """
        将输入转化为索引表示
        :param data: 输入的是questions 和 responses
        :param word_to_index: 词汇-索引映射表
        :return:
        """
        data_ids = []
        for word in data:
            if word in word_to_index:
                data_ids.append(word_to_index[word])
            else:
                data_ids.append(word_to_index["<UNK>"])

        return data_ids

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        # 如果不是第一次数据预处理，则直接读取
        if os.path.exists(os.path.join(self._output_path, "train_data.pkl")):
            print("load existed train data")
            with open(os.path.join(self._output_path, "train_data.pkl"), "rb") as f:
                train_data = pickle.load(f)

            if os.path.exists(os.path.join(self._output_path, "word_vectors.npy")):
                self.word_vectors = np.load(os.path.join(self._output_path, "word_vectors.npy"))

            return train_data

        # 1，读取原始数据
        datas = self.read_data()

        # 3，得到词汇表
        word_to_index = self.gen_vocab(datas)

        # 4，输入转索引
        datas_idx = self.trans_to_index(datas, word_to_index)

        with open(os.path.join(self._output_path, "train_data.pkl"), "wb") as fw:
            pickle.dump(datas_idx, fw)

        return np.array(datas_idx)

    def next_batch(self, data, batch_size, sequence_length):
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