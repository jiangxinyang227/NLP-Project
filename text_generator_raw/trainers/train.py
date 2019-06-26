import json
import os
import argparse
import sys
import math

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import tensorflow as tf
from data_helpers import TrainData, EvalData, BpeTrainData, BpeEvalData
from train_base import TrainerBase
from models import Seq2SeqBiLstm, Seq2SeqTransformer
from metrics import get_bleu, mean


class Trainer(TrainerBase):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), args.config_path), "r") as fr:
            self.config = json.load(fr)

        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None

        # self.builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(
        #     os.path.abspath(os.path.dirname(os.getcwd())), self.config["pb_model_path"]))

        # 加载数据集
        self.load_data()

        self.train_data = self.train_data_obj.gen_data()

        self.word_vectors = self.train_data_obj.word_vectors
        self.vocab_size = self.train_data_obj.vocab_size
        print("data load finished")
        print("vocab size: {}".format(self.vocab_size))

        self.eval_data = self.eval_data_obj.gen_data()

        # 初始化模型对象
        self.create_model()
        print("model had build")

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        if self.config["use_bpe"]:
            self.train_data_obj = BpeTrainData(self.config)
            self.eval_data_obj = BpeEvalData(self.config)
            print("use bpe to segment")
        else:
            # 生成训练集对象并生成训练数据
            self.train_data_obj = TrainData(self.config)
            # 生成验证集对象和验证集数据
            self.eval_data_obj = EvalData(self.config)

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config["model_name"] == "seq2seq_transformer":
            self.model = Seq2SeqTransformer(config=self.config, vocab_size=self.vocab_size,
                                            word_vectors=self.word_vectors)

        if self.config["model_name"] == "seq2seq_bilstm":
            self.model = Seq2SeqBiLstm(config=self.config, vocab_size=self.vocab_size,
                                       word_vectors=self.word_vectors)

    @staticmethod
    def schedule_sample(current_step, k=0.9, min_prob=0.4, mode="linear"):
        """
        用来做计划采样，主要时在训练时控制在解码的过程中是使用真实值还是预测值。
        :param current_step:
        :param k:
        :param min_prob:
        :param mode:
        :return:
        """
        if mode == "linear":
            sample_prob = 1 - max(min_prob, (k - current_step * 5e-5))
            return sample_prob

        elif mode == "exponential":
            sample_prob = 1 - pow(k, current_step / 1000)
            return sample_prob

        elif mode == "sigmoid":
            sample_prob = 1 - 1300 / (1300 + math.exp(current_step / 1300))
            return sample_prob

    def train(self):
        """
        训练模型
        :return:
        """
        with tf.Session() as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())

            # 创建train和eval的summary路径和写入对象
            train_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                              self.config["output_path"] + "/summary/train")
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)

            eval_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                             self.config["output_path"] + "/summary/eval")
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)

            current_step = 0
            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.train_data_obj.next_batch(self.train_data,
                                                            self.config["batch_size"]):

                    loss, prediction = self.model.train(sess, batch)

                    # 将train参数加入到tensorboard中
                    # train_summary_writer.add_summary(summary, global_step)

                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print("train: step: {}, loss: {}, perplexity: {}".format(current_step, loss, perplexity))
                    current_step += 1

                    if current_step % self.config["checkpoint_every"] == 0:
                        if self.eval_data:
                            eval_losses = []
                            eval_perplexities = []
                            for eval_batch in self.eval_data_obj.next_batch(self.eval_data,
                                                                            self.config["batch_size"]):
                                eval_loss_, eval_summary_ = self.model.eval(sess, eval_batch)
                                eval_loss, eval_summary = sess.run([eval_loss_, eval_summary_])
                                # 将eval参数加入到tensorboard中
                                # eval_summary_writer.add_summary(eval_summary, global_step)

                                eval_perplexity = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                                eval_losses.append(eval_loss)
                                eval_perplexities.append(eval_perplexity)

                            print("\n")
                            print("eval: step: {}, loss: {}, perplexity: {}".format(current_step,
                                                                                    mean(eval_losses),
                                                                                    mean(eval_perplexities)))
                            print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                     self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

            # inputs = {"inputs": tf.saved_model.utils.build_tensor_info(self.train_model.encoder_inputs),
            #           "inputs_length": tf.saved_model.utils.build_tensor_info(self.train_model.encoder_inputs_length),
            #           "keep_prob": tf.saved_model.utils.build_tensor_info(self.train_model.keep_prob)}
            #
            # outputs = {"predictions": tf.saved_model.utils.build_tensor_info(self.train_model.predictions)}
            #
            # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
            #                                                                               outputs=outputs,
            #                                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
            # self.builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
            #                                           signature_def_map={"dialogue": prediction_signature},
            #                                           legacy_init_op=legacy_init_op)
            #
            # self.builder.save()


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
