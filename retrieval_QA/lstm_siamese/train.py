import json
import os
import argparse
import random

import tensorflow as tf
from data_helper import SiameseLstmData
from model import SiameseLstmModel
from metrics import mean, get_binary_metrics


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r", encoding="utf8") as fr:
            self.config = json.load(fr)

        # 加载数据集
        self.data_obj = self.load_data()
        self.t_queries, self.t_sims, self.t_labels = self.data_obj.gen_data(self.config["train_data"])
        self.e_queries, self.e_sims, self.e_labels = self.data_obj.gen_data(self.config["eval_data"], False)
        print("vocab size: ", self.data_obj.vocab_size)

        # 初始化模型对象
        self.model = self.create_model()

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        data_obj = SiameseLstmData(self.config)

        return data_obj

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """

        model = SiameseLstmModel(config=self.config,
                                 vocab_size=self.data_obj.vocab_size,
                                 word_vectors=self.data_obj.word_vectors)

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

                for batch in self.data_obj.next_batch(self.t_queries, self.t_sims, self.t_labels,
                                                      self.config["batch_size"]):
                    loss, predictions = self.model.train(sess, batch, self.config["keep_prob"])

                    acc, rec, pre, f = get_binary_metrics(pred_y=predictions, true_y=batch["label"])

                    print("train: step: {}, loss: {}, acc: {}, rec: {}, pre: {}, f: {}".format(
                        current_step, loss, acc, rec, pre, f))

                    current_step += 1
                    if current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_acc = []
                        eval_rec = []
                        eval_pre = []
                        eval_f = []
                        for eval_batch in self.data_obj.next_batch(self.e_queries, self.e_sims, self.e_labels,
                                                                   self.config["batch_size"]):
                            eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_losses.append(eval_loss)

                            acc, rec, pre, f, = get_binary_metrics(pred_y=eval_predictions,
                                                                   true_y=eval_batch["label"])
                            eval_acc.append(acc)
                            eval_rec.append(rec)
                            eval_pre.append(pre)
                            eval_f.append(f)

                        print("\n")
                        print("eval: , loss: {}, acc: {}, rec: {}, pre: {}, f: {}".format(
                            mean(eval_losses), mean(eval_acc), mean(eval_rec), mean(eval_pre), mean(eval_f)))

                        print("\n")

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
