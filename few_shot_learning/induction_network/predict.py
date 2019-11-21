#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name ：     predict.py
   Author ：        王枫
   date ：          2019/11/13 18:13
-------------------------------------------------
   Description ： 
                   
-------------------------------------------------
"""

import json
import os
import tensorflow as tf
from model import InductionModel
import numpy as np


def online_predict():
    with open("config.json", "r") as fr:
        config = json.load(fr)
    with open("output/induction/word_to_index.json", "r") as f:
        word2id = json.load(f)
    word_vectors = np.load("output/induction/word_vectors.npy")
    config["num_classes"] = 3
    model = InductionModel(config=config, vocab_size=len(word2id), word_vectors=word_vectors)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
    max_lens = config["sequence_length"]
    with tf.Session(config=sess_config) as sess:
        save_path = os.path.join(os.path.abspath(os.getcwd()), config["ckpt_model_path"])
        checkpoint_prefix = os.path.join(save_path, config["model_name"] + "-500")
        model.saver.restore(sess, checkpoint_prefix)
        pos_sentence = ["i like it", "i love it", "i get it make me so happy", "happy", "so love it"]
        neu_sentence = ["hello world", "hello python", "what is it", "are you ok", "hello c++"]
        neg_sentence = ["it's do bad", "rubbish", "i don't like it", "i don't love it", "shit"]
        pos_ids = []
        for sentence in pos_sentence:
            ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
            if len(ids) < max_lens:
                ids = ids + [0] * (max_lens - len(ids))
            ids = ids[:max_lens]
            pos_ids.append(ids)
        neg_ids = []
        for sentence in neg_sentence:
            ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
            if len(ids) < max_lens:
                ids = ids + [0] * (max_lens - len(ids))
            ids = ids[:max_lens]
            neg_ids.append(ids)

        neu_ids = []
        for sentence in neu_sentence:
            ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
            if len(ids) < max_lens:
                ids = ids + [0] * (max_lens - len(ids))
            ids = ids[:max_lens]
            neu_ids.append(ids)

        support = [pos_ids, neu_ids, neg_ids]

        while True:
            queries = []
            for i in range(1):
                sentence = input("输入测试句子：")
                ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
                if len(ids) < max_lens:
                    ids = ids + [0] * (max_lens - len(ids))
                ids = ids[:max_lens]
                queries.append(ids)
            batch1 = {"queries": queries, "support": support}
            predict, scores = model.infer(sess, batch1)

            print("===============================")
            print(predict, scores)


if __name__ == "__main__":
    online_predict()
