"""
数据预处理类
"""
import os
import json
import random
from collections import Counter
from itertools import chain

import gensim
import numpy as np
import pandas as pd


class TrainingData(object):
    def __init__(self, output_path, sequence_length, stop_word_path=None, embedding_size=None,
                 low_freq=0, word_vector_path=None, is_training=True):
        """

        :param output_path: output file path，主要是用来保存词汇映射表，标签映射表
        :param sequence_length: 序列长度
        :param stop_word_path: 停用词表路径
        :param embedding_size: 嵌入大小
        :param low_freq: 低频词
        :param word_vector_path: 词向量路径
        """

        self.__output_path = output_path
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.__sequence_length = sequence_length
        self.__stop_word_path = stop_word_path
        self.__embedding_size = embedding_size
        self.__low_freq = low_freq
        self.__word_vector_path = word_vector_path
        self.__is_training = is_training

        self.vocab_size = None
        self.word_vectors = None

    @staticmethod
    def load_data(file_path):
        """
        加载数据
        :param file_path: 训练集或验证集的文件路径
        :return:
        """
        df = pd.read_csv(file_path)
        contents = df["token_content"].tolist()
        services = df["service"].tolist()
        environments = df["environment"].tolist()
        hygienes = df["hygiene"].tolist()

        new_contents, new_services, new_environments, new_hygiene = [], [], [], []
        for content, service, environment, hygiene in zip(contents, services, environments, hygienes):
            new_contents.append(content.strip().split(" "))
            new_services.append(str(service))
            new_environments.append(str(environment))
            new_hygiene.append(str(hygiene))

        return new_contents, new_services, new_environments, new_hygiene

    @classmethod
    def load_aspect_word(cls):
        aspect_name_word = {
            "service": "排队 态度 服务员 热情 服务态度 老板 服务 服务生 开车 停车费 停车位 停车场 车位 泊车 很快 催 慢 速度 分钟 上菜 等",
            "environment": "装修 布置 灯光 古色古香 装饰 优雅 情调 安静 环境 氛围 嘈杂 吵闹 音乐 大 宽敞 空间 面积 装修 拥挤 店面",
            "hygiene": "整洁 干净 环境 卫生 苍蝇 不错 脏"
        }
        aspects = [value.strip().split(" ") for key, value in aspect_name_word.items()]
        return aspects

    def remove_stop_word(self, contents):
        """
        去除低频词/字和停用词/字, 构建一个项目的vocab词/字汇表
        :param contents: 输入
        :return:
        """
        all_words = list(chain(*contents))

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
        :param labels: 标签
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

        # 将标签-索引映射表保存为pkl数据
        unique_labels = list(set(labels))
        label_to_index = dict(zip(unique_labels, list(range(len(unique_labels)))))
        with open(os.path.join(self.__output_path, "label_to_index.json"), "w", encoding="utf8") as f:
            json.dump(label_to_index, f, ensure_ascii=False, indent=0)

        return word_to_index, label_to_index

    def get_vocab(self):
        """
        验证时，直接读取词汇映射表和标签映射表
        :return:
        """
        with open(os.path.join(self.__output_path, "word_to_index.json"), "r", encoding="utf8") as f:
            word_to_index = json.load(f)

        with open(os.path.join(self.__output_path, "label_to_index.json"), "r", encoding="utf8") as f:
            label_to_index = json.load(f)

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

    @staticmethod
    def padding(inputs, sequence_length):
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

    @staticmethod
    def max_padding(inputs):
        input_length = [len(sentence) for sentence in inputs]
        max_len = max(input_length)
        inputs_pad = [sentence + [0] * (max_len - len(sentence)) for sentence in inputs]
        return inputs_pad, input_length

    def gen_data(self, file_path):
        """
        生成可导入到模型中的数据
        :return:
        """
        # 1，读取原始数据
        contents, services, environments, hygiene = self.load_data(file_path)
        if self.__is_training:

            # 2，得到去除低频词和停用词的词汇表
            words = self.remove_stop_word(contents)

            # 3，得到词汇表
            word_to_index, label_to_index = self.gen_vocab(words, services)
        else:
            word_to_index, label_to_index = self.get_vocab()

        aspects = self.load_aspect_word()
        aspect_ids = self.trans_to_index(aspects, word_to_index)
        aspect_ids, aspect_length = self.max_padding(aspect_ids)

        # 4，输入转索引
        content_ids = self.trans_to_index(contents, word_to_index)

        # 5，对输入序列按最大长度补全
        content_ids = self.padding(content_ids, self.__sequence_length)

        # 6，标签转索引
        service_ids = self.trans_label_to_index(services, label_to_index)
        environment_ids = self.trans_label_to_index(environments, label_to_index)
        hygiene_ids = self.trans_label_to_index(hygiene, label_to_index)

        if self.__is_training:
            return content_ids, service_ids, environment_ids, hygiene_ids, aspect_ids, aspect_length, label_to_index
        else:
            return content_ids, service_ids, environment_ids, hygiene_ids

    @staticmethod
    def concat_labels(services, environments, hygiene):
        """
        将三个aspect的标签组合在一起输入
        :param services:
        :param environments:
        :param hygiene:
        :return:
        """
        labels = []
        for service, environment, hygiene_ in zip(services, environments, hygiene):
            labels.append([service, environment, hygiene_])
        return labels

    def next_batch(self, contents, services, environments, hygiene, batch_size):
        """
        生成batch 数据集
        :param contents:
        :param services:
        :param environments:
        :param hygiene:
        :param batch_size:
        :return:
        """
        z = list(zip(contents, services, environments, hygiene))
        random.shuffle(z)
        contents, services, environments, hygiene = zip(*z)

        num_batches = len(contents) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_contents = contents[start: end]
            batch_labels = self.concat_labels(services[start: end], environments[start: end], hygiene[start: end])

            yield dict(contents=batch_contents, labels=batch_labels)

