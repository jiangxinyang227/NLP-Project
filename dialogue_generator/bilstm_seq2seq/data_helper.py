import os
import json
import random
from collections import Counter
from itertools import chain

import gensim
import numpy as np


class BilstmSeq2SeqData(object):
    def __init__(self, config):

        self._output_path = config["output_path"]
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        self._word_vectors_path = config["word_vectors_path"]
        self._embedding_size = config["embedding_size"]  # 词向量的长度

        self.vocab_size = config["vocab_size"]
        self.word_vectors = None

        self.pad_token = 0
        self.go_token = 2
        self.eos_token = 3

    @staticmethod
    def read_data(file_path):
        """
        读取数据
        :return: 返回分词后的对话对，questions, responses = [[]]
        """
        with open(file_path, "r", encoding="utf8") as f:
            posts = []
            responses = []
            for line in f.readlines():
                post, response = line.strip().split("<SEP>")
                posts.append(post.strip().split(" "))
                responses.append(response.strip().split(" "))
        return posts, responses

    def get_word_vectors(self, vocab):
        """
        加载词向量，并获得相应的词向量矩阵
        :param vocab: 训练集所含有的单词
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

    def gen_vocab(self, posts, responses):
        """
        生成词汇，标签等映射表
        :param posts
        :param responses
        :return:
        """
        all_words = list(chain(*posts)) + list(chain(*responses))

        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        words = ["<PAD>", "<UNK>", "GO", "EOS"] + [item[0] for item in sort_word_count]

        vocab = words[:self.vocab_size]
        self.vocab_size = len(vocab)

        if self._word_vectors_path:
            word_vectors = self.get_word_vectors(vocab)
            self.word_vectors = word_vectors
            # 将本项目的词向量保存起来
            np.save(os.path.join(self._output_path, "word_vectors.npy"), self.word_vectors)

        word_to_index = dict(zip(vocab, list(range(len(vocab)))))
        with open(os.path.join(self._output_path, "word_to_index.json"), "w", encoding="utf8") as fw:
            json.dump(word_to_index, fw, ensure_ascii=False, indent=0)

        return word_to_index

    def get_vocab(self):
        """
        处理验证集时直接加载该方法得到word to index 的映射表
        :return:
        """
        with open(os.path.join(self._output_path, "word_to_index.json"), "r", encoding="utf8") as fr:
            word_to_index = json.load(fr)

        return word_to_index

    @staticmethod
    def trans_to_index(data, word_to_index):
        """
        将输入转化为索引表示
        :param data: 输入的是questions 和 responses
        :param word_to_index: 词汇-索引映射表
        :return:
        """
        data_ids = [[word_to_index.get(token, word_to_index["<UNK>"]) for token in sentence] for sentence in data]
        return data_ids

    def padding(self, batch):
        """
        对每个batch数据按数据集中最大长度的句子进行补全
        :param batch:
        :return:
        """
        question_length = [len(sample[0]) for sample in batch]
        max_question_length = max(question_length)
        questions = [sample[0] + [self.pad_token] * (max_question_length - len(sample[0]))
                     for sample in batch]

        # 在这里先对response加上一个终止符<eos>
        responses = [sample[1] + [self.eos_token] for sample in batch]
        response_length = [len(response) for response in responses]
        max_response_length = max(response_length)

        # 对response按最大长度补齐
        pad_responses = [response + [self.pad_token] * (max_response_length - len(response)) for response in responses]

        return dict(questions=questions, responses=pad_responses,
                    question_length=question_length, response_length=response_length)

    def gen_data(self, file_path, is_training=True):
        """
        生成可导入到模型中的数据
        :param file_path
        :param is_training
        :return:
        """

        # 1，读取原始数据
        posts, responses = self.read_data(file_path)

        # 2, 得到词汇表
        if is_training:
            # 3，得到词汇表
            word_to_index = self.gen_vocab(posts, responses)

        else:
            word_to_index = self.get_vocab()

        # 3，输入转索引
        post_ids = self.trans_to_index(posts, word_to_index)
        response_ids = self.trans_to_index(responses, word_to_index)

        # 生成数据并保存下来
        train_data = [(post, response) for post, response in zip(post_ids, response_ids)]

        return train_data

    def next_batch(self, data, batch_size):
        """
        生成batch数据集
        :param data: 输入
        :param batch_size: 批量的大小
        :return:
        """
        random.shuffle(data)
        batch_num = len(data) // batch_size

        for i in range(batch_num):
            batch_data = data[batch_size * i: batch_size * (i + 1)]
            new_batch = self.padding(batch_data)
            yield new_batch
