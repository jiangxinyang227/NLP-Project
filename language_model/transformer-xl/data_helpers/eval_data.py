import os
import pickle
import copy
import numpy as np

from .data_base import EvalPredictDataBase


class EvalData(EvalPredictDataBase):
    def __init__(self, config):
        super(EvalData, self).__init__(config)

        self._eval_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config["eval_data"])
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                         config["output_path"])

    def read_data(self):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，questions, responses = [[]]
        """
        with open(self._eval_data_path, "r", encoding="utf8") as fq:
            datas = fq.read()

        return datas

    def load_vocab(self):
        """
        加载词汇和标签的映射表
        :return:
        """
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self._output_path, "word_to_index.pkl"), "rb") as f:
            word_to_index = pickle.load(f)

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
        if os.path.exists(os.path.join(self._output_path, "eval_data.pkl")):
            print("load existed train data")
            with open(os.path.join(self._output_path, "eval_data.pkl"), "rb") as f:
                eval_data = pickle.load(f)
            return eval_data

        # 1，读取原始数据
        datas = self.read_data()

        # 2，得到词汇表
        word_to_index = self.load_vocab()

        # 3，输入转索引
        data_idx = self.trans_to_index(datas, word_to_index)
        with open(os.path.join(self._output_path, "eval_data.pkl"), "wb") as fw:
            pickle.dump(data_idx, fw)

        return np.array(data_idx)

    def next_batch(self, data, n_seqs, n_steps):
        """
        生成batch数据集
        :param data: 原始数据
        :param n_seqs: 一个batch中序列的数量
        :param n_steps: 每条序列的长度
        :return:
        """
        data = copy.copy(data)
        batch_size = n_seqs * n_steps
        n_batches = int(len(data) / batch_size)
        arr = data[:batch_size * n_batches]
        arr = arr.reshape((n_seqs, -1))
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield dict(inputs=x, labels=y)