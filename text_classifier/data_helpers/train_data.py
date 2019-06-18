import os
import pickle
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
        self._stop_word_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                            config["stop_word"]) if config["stop_word"] else None

        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长
        self._batch_size = config["batch_size"]
        self._embedding_size = config["embedding_size"]  # 词向量的长度

        self.vocab_size = config["vocab_size"]
        self.word_vectors = None

    def read_data(self):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        inputs = []
        labels = []
        with open(self._train_data_path, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                try:
                    text, label = line.strip().split("<SEP>")
                    inputs.append(text.strip().split(" "))
                    labels.append(label)
                except:
                    continue

        return inputs, labels

    def remove_stop_word(self, inputs):
        """
        去除低频词和停用词
        :param inputs: 输入
        :return:
        """
        all_words = [word for data in inputs for word in data]

        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sort_word_count]

        # 如果传入了停用词表，则去除停用词
        if self._stop_word_path:
            with open(self._stop_word_path, "r", encoding="utf8") as fr:
                stop_words = [line.strip() for line in fr.readlines()]
            words = [word for word in words if word not in stop_words]

        return words

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

    def gen_vocab(self, words, labels):
        """
        生成词汇，标签等映射表
        :param words: 训练集所含有的单词
        :param labels: 标签
        :return:
        """
        # 如果不是第一次处理，则可以直接加载生成好的词汇表和词向量
        if os.path.exists(os.path.join(self._output_path, "word_vectors.npy")):
            print("load word_vectors")
            self.word_vectors = np.load(os.path.join(self._output_path, "word_vectors.npy"))

        if os.path.exists(os.path.join(self._output_path, "word_to_index.pkl")) and \
                os.path.exists(os.path.join(self._output_path, "label_to_index.pkl")):
            print("load word_to_index")
            with open(os.path.join(self._output_path, "word_to_index.pkl"), "rb") as f:
                word_to_index = pickle.load(f)

            with open(os.path.join(self._output_path, "label_to_index.pkl"), "rb") as f:
                label_to_index = pickle.load(f)

            self.vocab_size = len(word_to_index)

            return word_to_index, label_to_index

        words = ["<PAD>", "<UNK>"] + words
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

        # 将标签-索引映射表保存为pkl数据
        unique_labels = list(set(labels))
        label_to_index = dict(zip(unique_labels, list(range(len(unique_labels)))))
        with open(os.path.join(self._output_path, "label_to_index.pkl"), "wb") as f:
            pickle.dump(label_to_index, f)

        return word_to_index, label_to_index

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :param word_to_index: 词汇-索引映射表
        :return:
        """
        inputs_idx = [[word_to_index.get(word, word_to_index["<UNK>"])for word in sentence] for sentence in inputs]

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
        if os.path.exists(os.path.join(self._output_path, "train_data.pkl")) and \
                os.path.exists(os.path.join(self._output_path, "label_to_index.pkl")) and \
                os.path.exists(os.path.join(self._output_path, "word_to_index.pkl")):
            print("load existed train data")
            with open(os.path.join(self._output_path, "train_data.pkl"), "rb") as f:
                train_data = pickle.load(f)

            with open(os.path.join(self._output_path, "word_to_index.pkl"), "rb") as f:
                word_to_index = pickle.load(f)

            self.vocab_size = len(word_to_index)

            with open(os.path.join(self._output_path, "label_to_index.pkl"), "rb") as f:
                label_to_index = pickle.load(f)

            if os.path.exists(os.path.join(self._output_path, "word_vectors.npy")):
                self.word_vectors = np.load(os.path.join(self._output_path, "word_vectors.npy"))

            return np.array(train_data["inputs_idx"]), np.array(train_data["labels_idx"]), label_to_index

        # 1，读取原始数据
        inputs, labels = self.read_data()
        print("read finished")

        # 2，得到去除低频词和停用词的词汇表
        words = self.remove_stop_word(inputs)
        print("word process finished")

        # 3，得到词汇表
        word_to_index, label_to_index = self.gen_vocab(words, labels)
        print("vocab process finished")

        # 4，输入转索引
        inputs_idx = self.trans_to_index(inputs, word_to_index)
        print("index transform finished")

        # 5，对输入做padding
        inputs_idx = self.padding(inputs_idx, self._sequence_length)
        print("padding finished")

        # 6，标签转索引
        labels_idx = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")

        train_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
        with open(os.path.join(self._output_path, "train_data.pkl"), "wb") as fw:
            pickle.dump(train_data, fw)

        return np.array(inputs_idx), np.array(labels_idx), label_to_index

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