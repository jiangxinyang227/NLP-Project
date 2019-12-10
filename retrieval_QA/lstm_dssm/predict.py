import os
import json

import jieba
import numpy as np
import tensorflow as tf
from model import LstmDssmModel


class Predictor(object):
    def __init__(self, config, samples, batch_size=1):

        self.config = config
        self.batch_size = batch_size
        self.samples = samples
        self.output_path = config["output_path"]

        self.word_to_index, self.word_vectors = self.get_vocab()
        self.vocab_size = len(self.word_to_index)

        # 创建模型
        self.model = self.create_model()
        # 加载计算图
        self.sess = self.load_graph()

    def get_vocab(self):
        with open(os.path.join(self.output_path, "word_to_index.json"), "r", encoding="utf8") as fr:
            word_to_index = json.load(fr)

        if os.path.exists(os.path.join(self.output_path, "word_vectors.npy")):
            word_vectors = np.load(os.path.join(self.output_path, "word_vectors.npy"))
        else:
            word_vectors = None

        return word_to_index, word_vectors

    def sentence_to_idx(self, sentences):
        """

        :param sentences:
        :return:
        """
        sentences = [jieba.lcut(sentence) for sentence in sentences]
        sentences = [[self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in sentence] for sentence in sentences]
        sequence_len = [len(sentence) for sentence in sentences]
        max_len = max(sequence_len)
        sentences = [sentence + [0] * (max_len - len(sentence)) for sentence in sentences]

        return sentences, sequence_len

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
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

        model = LstmDssmModel(self.config, self.vocab_size, self.word_vectors, batch_size=self.batch_size,
                              samples=self.samples)
        return model

    def predict(self, query, candidates):
        """

        :param query:
        :param candidates:
        :return:
        """
        candidate_ids, candidate_len = self.sentence_to_idx(candidates)
        query_ids, query_len = self.sentence_to_idx([query])
        batch = dict(query=query_ids, sim=candidate_ids, query_length=query_len, sim_length=candidate_len)
        prediction = self.model.infer(self.sess, batch).tolist()[0]
        return prediction
