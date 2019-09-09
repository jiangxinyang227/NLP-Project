import json
import os

import jieba
import tensorflow as tf
import numpy as np
from model import LstmClassifier


class Predictor(object):
    def __init__(self, config):

        self.config = config
        self.word_to_index, self.label_to_index, self.word_vectors = self.load_vocab()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.vocab_size = len(self.word_to_index)
        self.sequence_length = self.config["sequence_length"]
        self.aspect_ids, self.aspect_length = self.load_aspect_word()
        # 创建模型
        self.model = self.create_model()
        # 加载计算图
        self.sess = self.load_graph()

    def load_vocab(self):
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self.config["output_path"], "word_to_index.json"), "r", encoding="utf8") as f:
            word_to_index = json.load(f)

        with open(os.path.join(self.config["output_path"], "label_to_index.json"), "r", encoding="utf8") as f:
            label_to_index = json.load(f)

        word_vectors = np.load(os.path.join(self.config["output_path"], "word_vectors.npy"))

        return word_to_index, label_to_index, word_vectors

    def load_aspect_word(self):
        aspect_name_word = {
            "service": "排队 态度 服务员 热情 服务态度 老板 服务 服务生 开车 停车费 停车位 停车场 车位 泊车 很快 催 慢 速度 分钟 上菜 等",
            "environment": "装修 布置 灯光 古色古香 装饰 优雅 情调 安静 环境 氛围 嘈杂 吵闹 音乐 大 宽敞 空间 面积 装修 拥挤 店面",
            "hygiene": "整洁 干净 环境 卫生 苍蝇 不错 脏"
        }
        aspects = [value.strip().split(" ") for key, value in aspect_name_word.items()]
        aspect_ids = [[self.word_to_index.get(token, self.word_to_index["<UNK>"]) for token in aspect] for aspect in aspects]
        aspect_length = [len(sentence) for sentence in aspect_ids]
        max_len = max(aspect_length)
        aspect_pad = [aspect + [0] * (max_len - len(aspect)) for aspect in aspect_ids]
        return aspect_pad, aspect_length

    def sentence_to_idx(self, sentence):
        """
        将分词后的句子转换成idx表示
        :param sentence:
        :return:
        """
        sentence = jieba.lcut(sentence)
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
        ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
        print(ckpt)
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

        model = LstmClassifier(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        return model

    def predict(self, sentence):
        """
        给定分词后的句子，预测其分类结果
        :param sentence:
        :return:
        """
        sentence_ids = self.sentence_to_idx(sentence)
        prediction = self.model.infer(self.sess, [sentence_ids], self.aspect_ids, self.aspect_length).tolist()[0]
        label = [self.index_to_label[pred] for pred in prediction]
        return label
