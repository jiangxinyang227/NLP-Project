import os
import pickle
import random
from collections import Counter
import gensim
import numpy as np
from itertools import chain

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

        self._embedding_size = config["embedding_size"]  # 词向量的长度

        self.vocab_size = config["vocab_size"]
        self.word_vectors = None

        self.pad_token = 0
        self.go_token = 2
        self.eos_token = 3

    def read_data(self):
        """
        读取数据
        :return: 返回分词后的对话对，questions, responses = [[]]
        """
        with open(self._train_data_path, "r", encoding="utf8") as f:
            requests = []
            responses = []
            for line in f.readlines():
                request, response = line.strip().split("<SEP>")
                requests.append(request.strip().split(" "))
                responses.append(response.strip().split(" "))
        return requests, responses

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

    def gen_vocab(self, questions, responses):
        """
        生成词汇，标签等映射表
        :param questions: 对话的输入
        :param responses: 对话的回复
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

        all_words = list(chain(*questions)) + list(chain(*responses))
        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        words = ["<PAD>", "<UNK>", "<GO>", "<EOS>"] + [item[0] for item in sort_word_count]

        vocab = words[:self.vocab_size]

        # 若vocab的长读小于设置的vocab_size，则选择vocab的长度作为真实的vocab_size
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

    @staticmethod
    def trans_to_index(data, word_to_index):
        """
        将输入转化为索引表示
        :param data: 输入的是questions 和 responses
        :param word_to_index: 词汇-索引映射表
        :return:
        """
        data_idx = [[word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence] for sentence in data]

        return data_idx

    def padding(self, batch):
        """
        对每个batch数据按数据集中最大长度的句子进行补全
        :param batch:
        :return:
        """
        # 对question添加结束符
        questions = [sample[0] + [self.eos_token] for sample in batch]
        question_length = [len(question) for question in questions]
        max_question_length = max(question_length)
        encoder_inputs = [question + [self.pad_token] * (max_question_length - len(question))
                          for question in questions]

        # 在这里先对response加上一个开始符和结束符<eos>
        responses = [[self.go_token] + sample[1] + [self.eos_token] for sample in batch]
        decoder_inputs = [response[:-1] for response in responses]
        decoder_outputs = [response[1:] for response in responses]

        decoder_inputs_length = [len(response) for response in decoder_inputs]
        max_decoder_inputs_length = max(decoder_inputs_length)

        decoder_outputs_length = [len(response) for response in decoder_outputs]
        max_decoder_outputs_length = max(decoder_outputs_length)

        # 按最大长度补齐
        decoder_inputs = [response + [self.pad_token] * (max_decoder_inputs_length - len(response))
                          for response in decoder_inputs]
        decoder_outputs = [response + [self.pad_token] * (max_decoder_outputs_length - len(response))
                           for response in decoder_outputs]

        return dict(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, decoder_outputs=decoder_outputs,
                    encoder_max_len=max_question_length, decoder_max_len=max_decoder_inputs_length)

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        # 如果不是第一次数据预处理，则直接读取
        if os.path.exists(os.path.join(self._output_path, "train_data.pkl")) and \
                os.path.exists(os.path.join(self._output_path, "word_to_index.pkl")):
            print("load existed train data")
            with open(os.path.join(self._output_path, "train_data.pkl"), "rb") as f:
                train_data = pickle.load(f)

            with open(os.path.join(self._output_path, "word_to_index.pkl"), "rb") as f:
                word_to_index = pickle.load(f)
                self.vocab_size = len(word_to_index)

            if os.path.exists(os.path.join(self._output_path, "word_vectors.npy")):
                self.word_vectors = np.load(os.path.join(self._output_path, "word_vectors.npy"))

            return train_data

        # 1，读取原始数据
        questions, responses = self.read_data()

        # 3，得到词汇表
        word_to_index = self.gen_vocab(questions, responses)

        # 4，输入转索引
        questions_idx = self.trans_to_index(questions, word_to_index)
        responses_idx = self.trans_to_index(responses, word_to_index)

        # 生成数据并保存下来
        train_data = [[questions_idx[i], responses_idx[i]] for i in range(len(questions_idx))]
        with open(os.path.join(self._output_path, "train_data.pkl"), "wb") as fw:
            pickle.dump(train_data, fw)
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
