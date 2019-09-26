import json
import os
import argparse

import tensorflow as tf
from data_helper import PrototypicalData
from model import PrototypicalModel
from metrics import get_multi_metrics, mean


class PrototypicalTrainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)

        # self.builder = tf.saved_model.builder.SavedModelBuilder("../pb_model/weibo/bilstm/savedModel")
        # load date set
        self.train_data_obj = self.load_data()
        self.eval_data_obj = self.load_data(is_training=False)
        self.train_tasks = self.train_data_obj.gen_data(self.config["train_data"])
        self.eval_tasks = self.eval_data_obj.gen_data(self.config["eval_data"])

        print("vocab size: ", self.train_data_obj.vocab_size)

        self.model = self.create_model()

    def load_data(self, is_training=True):
        """
        init data object
        :return:
        """
        data_obj = PrototypicalData(self.config["output_path"], sequence_length=self.config["sequence_length"],
                                    num_classes=self.config["num_classes"], num_support=self.config["num_support"],
                                    num_queries=self.config["num_queries"], num_tasks=self.config["num_tasks"],
                                    num_eval_tasks=self.config["num_eval_tasks"],
                                    is_training=is_training)
        return data_obj

    def create_model(self):
        """
        init model object
        :return:
        """
        model = PrototypicalModel(config=self.config, vocab_size=self.train_data_obj.vocab_size,
                                  word_vectors=self.train_data_obj.word_vectors)
        return model

    def train(self):
        """
        train model
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # init all variable in graph
            sess.run(tf.global_variables_initializer())
            current_step = 0

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))
                for batch in self.train_data_obj.next_batch(self.train_tasks):
                    loss, predictions = self.model.train(sess, batch, self.config["keep_prob"])
                    label_list = list(set(batch["labels"]))
                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch["labels"],
                                                                  labels=label_list)
                    print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        current_step, loss, acc, recall, prec, f_beta))

                    current_step += 1
                    if current_step % self.config["checkpoint_every"] == 0:
                        eval_losses = []
                        eval_accs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.train_data_obj.next_batch(self.eval_tasks):
                            eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_losses.append(eval_loss)
                            eval_label_list = list(set(eval_batch["labels"]))
                            acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                                                                          true_y=eval_batch["labels"],
                                                                          labels=eval_label_list)
                            eval_accs.append(acc)
                            eval_recalls.append(recall)
                            eval_precs.append(prec)
                            eval_f_betas.append(f_beta)
                        print("\n")
                        print("eval:  loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
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
    # Read the input information by the user on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    trainer = PrototypicalTrainer(args)
    trainer.train()
