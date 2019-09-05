import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import numpy as np
import tensorflow as tf
from predict_base import PredictorBase
from models import CharRNNModel


class Predictor(PredictorBase):
    def __init__(self, config):
        super(Predictor, self).__init__(config)
        self.config = config
        self.model = None
        self.sess = None

        # 加载词汇表
        self.word_vectors = None
        self.word_to_index = None
        self.vocab_size = None

        self.load_vocab()
        self.idx_to_label = {value: key for key, value in self.word_to_index.items()}

        # 初始化模型
        self.create_model()
        print("load model finished")
        # 加载计算图
        self.load_graph()
        print("load graph finished")

    def load_vocab(self):
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self.output_path, "word_to_index.pkl"), "rb") as f:
            self.word_to_index = pickle.load(f)
            self.vocab_size = len(self.word_to_index)

        if os.path.exists(os.path.join(self.output_path, "word_vectors.npy")):
            self.word_vectors = np.load(os.path.join(self.output_path, "word_vectors.npy"))

    def word_to_encode(self, words):
        """
        创建数据对象
        :return:
        """
        if not words:
            return None

        word_idx = [self.word_to_index.get(word, self.word_to_idx["<UNK>"]) for word in words]

        return word_idx

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config["model_name"] == "char_rnn":
            self.model = CharRNNModel(config=self.config, vocab_size=self.vocab_size,
                                      word_vectors=self.word_vectors, is_training=False)

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                          self.config["ckpt_model_path"]))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))

    def pick_top_n(self, preds, vocab_size, top_n=5):
        """
        取出softmax后最大的top个，然后从中随机选择一个作为输出
        :param preds:
        :param vocab_size:
        :param top_n:
        :return:
        """
        p = np.squeeze(preds)
        # 将除了top_n个预测值的位置都置为0
        p[np.argsort(p)[:-top_n]] = 0
        # 归一化概率
        p = p / np.sum(p)
        # 随机选取一个字符
        while True:
            c = np.random.choice(vocab_size, 1, p=p)[0]
            token_c = self.idx_to_label[c]
            if token_c != "<UNK>":
                return token_c

    def predict(self, start, num_sample):
        """
         给定一条句子，预测结果
        :return:
        """
        state = self.sess.run(self.model.initial_state)
        samples = []
        start_ids = self.word_to_encode(start)
        start_id = None
        for item in start_ids:
            start_id = item
            prediction, state = self.model.sample(self.sess, [[start_id]], state)
            token = self.pick_top_n(prediction, self.vocab_size)
            samples.append(token)

        for i in range(num_sample):
            prediction, state = self.model.sample(self.sess, [[start_id]], state)
            token = self.pick_top_n(prediction, self.vocab_size)
            samples.append(token)
            start_id = self.word_to_encode(token)[0]
        return "".join(samples)

