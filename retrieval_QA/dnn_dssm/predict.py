import json
import os

import jieba
import numpy as np
import tensorflow as tf
from model import DnnDssmModel


class TextCnnPredictor(object):
    def __init__(self, config, init_size, batch_size, samples):

        self.config = config
        self.init_size = init_size
        self.batch_size = batch_size
        self.samples = samples

        # 创建模型
        self.model = self.create_model()
        # 加载计算图
        self.sess = self.load_graph()

    def sentence_to_idx(self, sentences, tf_idf_model, dictionary):
        """

        :param sentences:
        :param tf_idf_model:
        :param dictionary:
        :return:
        """
        sentences = [jieba.lcut(sentence) for sentence in sentences]
        question_ids = []
        for question in sentences:
            bow_vec = dictionary.doc2bow(question)
            tfidf_vec = tf_idf_model[bow_vec]
            vec = [0] * self.init_size
            for item in tfidf_vec:
                vec[item[0]] = item[1]
            question_ids.append(vec)
        return question_ids

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

        model = DnnDssmModel(config=self.config, init_size=self.init_size, batch_size=self.batch_size,
                             samples=self.samples, is_training=False)
        return model

    def predict(self, query, candidates, tf_idf_model, dictionary):
        """

        :param query:
        :param candidates:
        :param tf_idf_model:
        :param dictionary:
        :return:
        """
        candidate_ids = self.sentence_to_idx(candidates, tf_idf_model, dictionary)
        query_ids = self.sentence_to_idx([query], tf_idf_model, dictionary)
        batch = dict(query=query_ids, sim=candidate_ids)
        prediction = self.model.infer(self.sess, batch).tolist()[0]
        return prediction
