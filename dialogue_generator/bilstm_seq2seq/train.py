import json
import os
import argparse
import math

import tensorflow as tf
from data_helper import BilstmSeq2SeqData
from model import Seq2SeqBiLstmModel
from metrics import get_bleu, mean


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)

        # 加载数据集
        self.data_obj = self.load_data()

        self.train_data = self.data_obj.gen_data(self.config["train_data"])
        self.eval_data = self.data_obj.gen_data(self.config["eval_data"], is_training=False)
        self.word_vectors = self.data_obj.word_vectors
        self.vocab_size = self.data_obj.vocab_size
        print("data load finished")
        print("vocab size: {}".format(self.vocab_size))

        # 初始化模型对象
        self.model = self.create_model()
        print("model had build")

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        data_obj = BilstmSeq2SeqData(self.config)
        return data_obj

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """

        model = Seq2SeqBiLstmModel(config=self.config, vocab_size=self.vocab_size,
                                   word_vectors=None)
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

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.train_data,
                                                      self.config["batch_size"]):

                    loss, predictions, summary = self.model.train(sess, batch, self.config["keep_prob"])

                    # 将train参数加入到tensorboard中
                    train_summary_writer.add_summary(summary, current_step)

                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    bleu = get_bleu(batch["responses"], predictions)
                    print("train: step: {}, loss: {}, perplexity: {}, bleu: {}".format(current_step,
                                                                                       loss,
                                                                                       perplexity,
                                                                                       bleu))

                    current_step += 1
                    if self.eval_data and current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_perplexities = []
                        eval_bleus = []
                        for eval_batch in self.data_obj.next_batch(self.eval_data,
                                                                   self.config["batch_size"]):
                            eval_loss, eval_predictions, eval_summary = self.model.eval(sess, eval_batch)

                            # 将eval参数加入到tensorboard中
                            eval_summary_writer.add_summary(eval_summary, current_step)

                            eval_perplexity = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                            eval_bleu = get_bleu(eval_batch["responses"], eval_predictions)
                            eval_losses.append(eval_loss)
                            eval_perplexities.append(eval_perplexity)
                            eval_bleus.append(eval_bleu)

                        print("\n")
                        print("eval: step: {}, loss: {}, perplexity: {}, bleu: {}".format(current_step,
                                                                                          mean(eval_losses),
                                                                                          mean(eval_perplexities),
                                                                                          mean(eval_bleus)))
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
