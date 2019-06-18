import os
import pickle
import numpy as np

from .data_base import EvalPredictDataBase


class EvalData(EvalPredictDataBase):
    def __init__(self, config):
        super(EvalData, self).__init__(config)

        self._eval_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config["eval_data"])
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                         config["output_path"])
        self._stop_word_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                            config["stop_word"]) if config["stop_word"] else None

        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长

    def read_data(self):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        inputs = []
        labels = []
        with open(self._eval_data_path, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                try:
                    text, label = line.strip().split("<SEP>")
                    inputs.append(text.strip().split(" "))
                    labels.append(label)
                except:
                    continue

        return inputs, labels

    def remove_stop_words(self, inputs):
        """
        去除停用词
        :param inputs:
        :return:
        """
        if self._stop_word_path:
            with open(self._stop_word_path, "r", encoding="utf8") as fr:
                stop_words = [line.strip() for line in fr.readlines()]
            inputs = [[word for word in sentence if word not in stop_words]for sentence in inputs]
        return inputs

    def load_vocab(self):
        """
        加载词汇和标签的映射表
        :return:
        """
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self._output_path, "word_to_index.pkl"), "rb") as f:
            word_to_index = pickle.load(f)

        # 将标签-索引映射表加载出来
        with open(os.path.join(self._output_path, "label_to_index.pkl"), "rb") as f:
            label_to_index = pickle.load(f)
        return word_to_index, label_to_index

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :param word_to_index: 词汇-索引映射表
        :return: 
        """
        inputs_idx = [[word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence] for sentence in inputs]

        return inputs_idx

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return: 
        """
        labels_idx = [label_to_index[label] for label in labels]
        return labels_idx

    def padding(self, inputs, sequence_length):
        """
        对序列进行截断和补全
        :param inputs: 输入
        :param sequence_length: 预定义的序列长度
        :return: 
        """
        new_inputs = [sentence[:sequence_length]
                      if len(sentence) > sequence_length
                      else sentence + [0] * (sequence_length - len(sentence))
                      for sentence in inputs]

        return new_inputs

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        # 如果不是第一次数据预处理，则直接读取
        if os.path.exists(os.path.join(self._output_path, "eval_data.pkl")):
            print("load existed eval data")
            with open(os.path.join(self._output_path, "eval_data.pkl"), "rb") as f:
                eval_data = pickle.load(f)
            return np.array(eval_data["inputs_idx"]), eval_data["labels_idx"]

        # 1，读取原始数据
        inputs, labels = self.read_data()
        print("read finished")

        # 2，去除停用词
        inputs = self.remove_stop_words(inputs)
        print("word process finished")

        # 3，得到词汇表
        word_to_index, label_to_index = self.load_vocab()
        print("load vocab finished")

        # 4，输入转索引
        inputs_idx = self.trans_to_index(inputs, word_to_index)
        print("input transform finished")

        # 5，对输入做padding
        inputs_idx = self.padding(inputs_idx, self._sequence_length)

        # 6，标签转索引
        labels_idx = self.trans_label_to_index(labels, label_to_index)
        print("label transform index finished")

        eval_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
        with open(os.path.join(self._output_path, "eval_data.pkl"), "wb") as fw:
            pickle.dump(eval_data, fw)

        return np.array(inputs_idx), np.array(labels_idx)

    def next_batch(self, x, y, batch_size):
        """
        生成batch数据集
        :param x: 输入
        :param y: 标签
        :param batch_size: 批量的大小
        :return: 
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = np.array(x[start: end], dtype="int64")
            batch_y = np.array(y[start: end], dtype="float32")

            yield dict(x=batch_x, y=batch_y)