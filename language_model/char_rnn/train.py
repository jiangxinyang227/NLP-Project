import json
import os
import argparse
import math

import tensorflow as tf
from data_helper import TrainData
from models import CharRNNModel
from metrics import mean


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)

        # self.builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(
        #     os.path.abspath(os.path.dirname(os.getcwd())), self.config["pb_model_path"]))

        # 加载数据集
        self.data_obj = self.load_data()
        self.train_data = self.data_obj.gen_data(self.config["train_data"])
        self.vocab_size = self.data_obj.vocab_size
        self.word_vectors = self.data_obj.word_vectors
        self.eval_data = self.data_obj.gen_data(self.config["eval_data"], is_training=False)

        # 初始化模型对象
        self.model = self.create_model()
        print("model had build")

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        data_obj = TrainData(self.config)
        return data_obj

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """

        model = CharRNNModel(config=self.config, vocab_size=self.vocab_size,
                             word_vectors=self.word_vectors)
        return model

    def train(self):
        """
        训练模型
        :return:
        """
        with tf.Session() as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())
            current_step = 0

            # 创建train和eval的summary路径和写入对象
            train_summary_path = self.config["output_path"] + "/summary/train"
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)

            eval_summary_path = self.config["output_path"] + "/summary/eval"
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)

            state = sess.run(self.model.initial_state)
            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.train_data,
                                                      self.config["batch_size"],
                                                      self.config["sequence_length"]):
                    summary_op, loss, state = self.model.train(sess, batch, state, self.config["keep_prob"])
                    train_summary_writer.add_summary(summary_op)

                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print("train: step: {}, loss: {}, perplexity: {}".format(current_step,
                                                                             loss,
                                                                             perplexity))

                    current_step += 1
                    if current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_perplexities = []
                        eval_state = sess.run(self.model.initial_state)
                        for eval_batch in self.data_obj.next_batch(self.eval_data,
                                                                   self.config["batch_size"],
                                                                   self.config["sequence_length"]):
                            eval_summary_op, eval_loss, eval_state = self.model.eval(sess, eval_batch, eval_state)
                            eval_summary_writer.add_summary(eval_summary_op)

                            eval_perplexity = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                            eval_losses.append(eval_loss)
                            eval_perplexities.append(eval_perplexity)

                        print("\n")
                        print("eval: step: {}, loss: {}, perplexity: {}".format(current_step,
                                                                                mean(eval_losses),
                                                                                mean(eval_perplexities)))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = self.config["ckpt_model_path"]
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
