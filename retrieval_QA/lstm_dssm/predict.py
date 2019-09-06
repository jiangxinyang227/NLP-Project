import json
import os

import numpy as np
import tensorflow as tf
from model import LstmDssmModel


class TextCnnPredictor(object):
    def __init__(self, config):

        self.config = config
        self.word_to_index, self.word_vectors = self.load_vocab()
        self.vocab_size = len(self.word_to_index)
        self.sequence_length = self.config["sequence_length"]

        # 创建模型
        self.model = self.create_model()
        # 加载计算图
        self.sess = self.load_graph()

    def load_vocab(self):
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self.config["output_path"], "word_to_index.json"), "r") as f:
            word_to_index = json.load(f)

        if os.path.exists(os.path.join(self.config["output_path"], "word_vectors.npy")):
            word_vectors = np.load(os.path.join(self.config["output_path"], "word_vectors.npy"))
        else:
            word_vectors = None

        return word_to_index, word_vectors

    def sentence_to_idx(self, sentence):
        """
        将分词后的句子转换成idx表示
        :param sentence:
        :return:
        """
        sentence_ids = [self.word_to_index.get(token, self.word_to_index["<UNK>"]) for token in sentence]
        sentence_pad = sentence_ids[: self.sequence_length] if len(sentence_ids) > self.sequence_length \
            else sentence_ids + [0] * (self.sequence_length - len(sentence_ids))
        return sentence_pad

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                          self.config["ckpt_model_path"]))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))
        return sess

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """

        model = LstmDssmModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        return model

    def predict(self, sentence):
        """
        给定分词后的句子，预测其分类结果
        :param sentence:
        :return:
        """
        sentence_ids = self.sentence_to_idx(sentence)
        prediction = self.model.infer(self.sess, [sentence_ids]).tolist()[0]
        label = self.index_to_label[prediction]
        return label
