import json
import os
import argparse
import random

import tensorflow as tf
from data_helper import DnnDssmData
from model import DnnDssmModel
from metrics import mean, accuracy

RATE = 0.2


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r", encoding="utf8") as fr:
            self.config = json.load(fr)

        self.data_obj = self.load_data()
        self.queries = self.data_obj.gen_data(self.config["train_data"])
        self.init_size = len(self.data_obj.dictionary.token2id)

        # 初始化模型对象
        self.model = self.create_model()

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        data_obj = DnnDssmData(self.config)

        return data_obj

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """

        model = DnnDssmModel(config=self.config,
                             init_size=self.init_size)
        return model

    def train(self):
        """
        训练模型
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())
            current_step = 0

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.queries, self.config["batch_size"]):
                    loss, predictions = self.model.train(sess, batch, self.config["keep_prob"])

                    acc = accuracy(predictions)

                    print("train: step: {}, loss: {}, acc: {}".format(current_step, loss, acc))

                    current_step += 1
                    if current_step % self.config["checkpoint_every"] == 0:

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.path.abspath(os.getcwd()),
                                                     self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
