import json
import os
import argparse

import tensorflow as tf
from data_helper import TrainingData
from model import LstmClassifier
from gate_conv import GateConvClassifier
from metrics import mean, get_aspect_metrics

RATE = 0.2


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r", encoding="utf8") as fr:
            self.config = json.load(fr)

        # self.builder = tf.saved_model.builder.SavedModelBuilder("../pb_model/weibo/bilstm/savedModel")
        # 加载数据集
        self.train_data_obj, self.eval_data_obj = self.load_data()
        self.cont, self.ser, self.env, self.hyg, self.asp, self.asp_len, label_to_idx = self.train_data_obj.gen_data(
            self.config["train_data"])
        print("vocab size: ", self.train_data_obj.vocab_size)
        self.eval_cont, self.eval_ser, self.eval_env, self.eval_hyg = self.eval_data_obj.gen_data(
            self.config["eval_data"])
        self.label_list = [value for key, value in label_to_idx.items()]

        # 初始化模型对象
        self.model = self.create_model()

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        train_data_obj = TrainingData(output_path=self.config["output_path"],
                                      sequence_length=self.config["sequence_length"],
                                      stop_word_path=self.config["stop_word_path"],
                                      embedding_size=self.config["embedding_size"],
                                      low_freq=self.config["low_freq"],
                                      word_vector_path=self.config["word_vector_path"])
        eval_data_obj = TrainingData(output_path=self.config["output_path"],
                                     sequence_length=self.config["sequence_length"],
                                     stop_word_path=self.config["stop_word_path"],
                                     embedding_size=self.config["embedding_size"],
                                     low_freq=self.config["low_freq"],
                                     word_vector_path=self.config["word_vector_path"],
                                     is_training=False)
        return train_data_obj, eval_data_obj

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """

        model = LstmClassifier(config=self.config,
                               vocab_size=self.train_data_obj.vocab_size,
                               word_vectors=self.train_data_obj.word_vectors)

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

                for batch in self.train_data_obj.next_batch(self.cont, self.ser, self.env, self.hyg,
                                                            self.config["batch_size"]):
                    loss, predictions = self.model.train(sess, batch, self.asp, self.asp_len, self.config["keep_prob"])

                    ser_acc, ser_rec, ser_pre, ser_f, env_acc, env_rec, env_pre, env_f, h_acc, h_rec, h_pre, h_f = \
                        get_aspect_metrics(pred_ys=predictions, true_ys=batch["labels"],
                                           labels=self.label_list)
                    if current_step % 50 == 0:
                        print("train: step: {}, loss: {}, ser_acc: {}, ser_rec: {}, ser_pre: {}, ser_f: {}".format(
                            current_step, loss, ser_acc, ser_rec, ser_pre, ser_f))
                        print("train: step: {}, loss: {}, env_acc: {}, env_rec: {}, env_pre: {}, env_f: {}".format(
                            current_step, loss, env_acc, env_rec, env_pre, env_f))
                        print("train: step: {}, loss: {}, h_acc: {}, h_rec: {}, h_pre: {}, h_f: {}".format(
                            current_step, loss, h_acc, h_rec, h_pre, h_f))

                    current_step += 1
                    if current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        ser_eval_acc = []
                        ser_eval_rec = []
                        ser_eval_pre = []
                        ser_eval_f = []
                        env_eval_acc = []
                        env_eval_rec = []
                        env_eval_pre = []
                        env_eval_f = []
                        h_eval_acc = []
                        h_eval_rec = []
                        h_eval_pre = []
                        h_eval_f = []
                        for eval_batch in self.train_data_obj.next_batch(self.eval_cont, self.eval_ser, self.eval_env,
                                                                         self.eval_hyg, self.config["batch_size"]):
                            eval_loss, eval_predictions = self.model.eval(sess, eval_batch, self.asp, self.asp_len)
                            eval_losses.append(eval_loss)

                            ser_acc, ser_rec, ser_pre, ser_f, env_acc, env_rec, env_pre, env_f, h_acc, h_rec, h_pre, h_f = \
                                get_aspect_metrics(pred_ys=eval_predictions,
                                                   true_ys=eval_batch["labels"],
                                                   labels=self.label_list)
                            ser_eval_acc.append(ser_acc)
                            ser_eval_rec.append(ser_rec)
                            ser_eval_pre.append(ser_pre)
                            ser_eval_f.append(ser_f)
                            env_eval_acc.append(env_acc)
                            env_eval_rec.append(env_rec)
                            env_eval_pre.append(env_pre)
                            env_eval_f.append(env_f)
                            h_eval_acc.append(h_acc)
                            h_eval_pre.append(h_pre)
                            h_eval_rec.append(h_rec)
                            h_eval_f.append(h_f)
                        print("\n")
                        print("eval: , loss: {}, ser_acc: {}, ser_rec: {}, ser_pre: {}, ser_f: {}".format(
                            mean(eval_losses), mean(ser_eval_acc), mean(ser_eval_rec), mean(ser_eval_pre),
                            mean(ser_eval_f)))
                        print("eval: , loss: {}, env_acc: {}, env_rec: {}, env_pre: {}, env_f: {}".format(
                            mean(eval_losses), mean(env_eval_acc), mean(env_eval_rec), mean(env_eval_pre),
                            mean(env_eval_f)))
                        print("eval: , loss: {}, h_acc: {}, h_rec: {}, h_pre: {}, h_f: {}".format(
                            mean(eval_losses), mean(h_eval_acc), mean(h_eval_rec), mean(h_eval_pre),
                            mean(h_eval_f)))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.path.abspath(os.getcwd()),
                                                     self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

            # inputs = {"inputs": tf.saved_model.utils.build_tensor_info(self.model.inputs),
            #           "keep_prob": tf.saved_model.utils.build_tensor_info(self.model.keep_prob)}
            #
            # outputs = {"predictions": tf.saved_model.utils.build_tensor_info(self.model.predictions)}
            #
            # # method_name决定了之后的url应该是predict还是classifier或者regress
            # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
            #                                                                               outputs=outputs,
            #                                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
            # self.builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
            #                                           signature_def_map={"classifier": prediction_signature},
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
